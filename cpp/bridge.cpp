#include <array>
#include <algorithm>
#include <string>
#include <cstring>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>

#include "llama.h"
#include "common.h"
#include "sampling.h"

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-backend.h"

#include "llama-impl.h"
#include "llama-vocab.h"
// #include "llama-grammar.h"
// #include "llama-sampling.h"

#include "bridge.h"
#include "janus.h"

char * debug; // debug level = "cuda|tokenizer", etc

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static gpt_params               * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting = false;

// FIXME ASAP - do not allow longer context when reading session file

// do_inference: PROMPT [ 2386 ] tokens
// do_inference: SESSION [ 0 ] tokens
// do_inference: error: prompt is too long (2386 tokens, max 2044)
// ggml_new_tensor_impl: not enough space in the scratch memory pool (needed 877775360, available 536870912)
// fatal error: unexpected signal during runtime execution

// Shared map for storing pairs of [UUID] -> [Output] while processing within C++ side
// After returning final result to Go side, we can safely remove the current result from the map

std::shared_mutex mutex;

// NB! Always use mutex to access map thread-safe way

// https://www.geeksforgeeks.org/map-vs-unordered_map-c/
// https://github.com/bdasgupta02/dynamap/issues/1
// https://github.com/tsixta/tsmap
// https://github.com/kshk123/hashMap

std::unordered_map<std::string, std::string> jobs;

// Map of vectors storing PROMPT token evaluation timings [ in milliseconds ]
std::unordered_map<std::string, int64_t> promptEvals;

// Map of vectors storing OUTPUT token evaluation timings [ in milliseconds ]
std::unordered_map<std::string, int64_t> timings;

// Map of vectors storing seeds for RNG
std::unordered_map<std::string, uint32_t> seeds;

// Map of vectors storing PROMPT token count
std::unordered_map<std::string, int64_t> promptTokenCount;

// Map of vectors storing OUTPUT token count
std::unordered_map<std::string, int64_t> outputTokenCount;

// Suspend stdout / stderr messaging
// https://stackoverflow.com/questions/70371091/silencing-stdout-stderr

static void log_nothing(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
    ///// fputs(text, stderr);
    ///// fflush(stderr);
}

void hide() {
    llama_log_set(log_nothing, NULL); // disable logging

    ///// (void) !freopen(NULL_DEVICE, "w", stdout);
    ///// (void) !freopen(NULL_DEVICE, "w", stderr);
}    

void show() {
    llama_log_set(NULL, NULL); // enable default logger
    ///// (void) !freopen(TTY_DEVICE, "w", stdout);
    ///// (void) !freopen(TTY_DEVICE, "w", stderr);
}

// --- Globals for all pods. Do anyone needs more than 8 pods per machine?

gpt_params params[8];             // main params 
llama_sampling_params sparams[8]; // sampling params
janus_params jparams[8];          // Janus Sampling params
llama_model * models[8];          // models
llama_context * contexts[8];      // contexts

// Flags to stop particular inference thread from the Go code

bool stopInferenceFlags[8];

// Directory where session data files will be held. Emtpy string if sessions are disabled

std::string path_session;

// -- init_context

struct llama_context * init_context(int idx) {

    auto modelName = params[idx].model.c_str();

    // -- initialize the model
    // TODO: Use llama_init_from_gpt_params :: llama_load_model_from_hf

    auto mparams = llama_model_params_from_gpt_params((const gpt_params &) ::params[idx]);

    mparams.main_gpu     = ::params[idx].main_gpu;
    mparams.n_gpu_layers = ::params[idx].n_gpu_layers;
    mparams.tensor_split = ::params[idx].tensor_split;

    llama_model * model = llama_load_model_from_file(modelName, mparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, modelName);
        // return std::make_tuple(nullptr, nullptr);
        return NULL;
    }

    models[idx] = model;

    // -- initialize the context

    auto cparams = llama_context_params_from_gpt_params(params[idx]);

    cparams.n_ctx           = ::params[idx].n_ctx;
    cparams.seed            = ::params[idx].seed;
    cparams.n_threads       = ::params[idx].n_threads;
    cparams.n_threads_batch = ::params[idx].n_threads_batch;

    // TODO: Determine best batch size for GPU (and maybe different depending on VRAM size)
    // NB! It crashes with batch of 32/64 and go loop with 128. So use batching of 256 or more

    bool isGPU = ::params[idx].n_gpu_layers > 0 ? true : false;

    if (::params[idx].n_batch > 0 && ::params[idx].n_batch <= ::params[idx].n_ctx) {
        cparams.n_batch = ::params[idx].n_batch;
    } else if (isGPU) {
        cparams.n_batch = 512;
    } else {
        cparams.n_batch = ::params[idx].n_ctx;
    }

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, modelName);
        llama_free_model(model);
        return NULL;
    }

    contexts[idx] = ctx;
    return ctx;
}

// Process prompt and compute output, return total number of tokens processed
// idx - index of pod / context / params to do processing within
int64_t do_inference(

    int idx, 
    struct llama_context * ctx, 
    const std::string & jobID, 
    const std::string & sessionID, 
    const std::string & prompt

) {

    llama_reset_timings(ctx);
    stopInferenceFlags[idx] = false;

    bool isGPU     = ::params[idx].n_gpu_layers > 0 ? true : false;
    auto model     = models[idx];
    ///// auto vocabSize = llama_n_vocab(model);

    gpt_params & params = ::params[idx];
    llama_sampling_params & sparams = ::sparams[idx];
    janus_params & jparams = ::jparams[idx];

    initJanus(ctx, jparams, debug); // NB!

    [[maybe_unused]] llama_context * ctx_guidance = NULL;
    g_model = &model;
    g_ctx = &ctx;
    g_params = &params;
    const int n_ctx = llama_n_ctx(ctx);
    std::string path_session = ::params[idx].path_prompt_cache;
    std::vector<llama_token> session_tokens;

    std::string sessionFile;
    if (!isGPU &&
        !path_session.empty() && 
        !sessionID.empty()) {

        sessionFile = path_session + '/' + sessionID;
    }

    // TODO: Do not always use RANDOM seed ?!
    // if (params.seed == LLAMA_DEFAULT_SEED) {
    auto seed = time(NULL);
    llama_set_rng_seed(ctx, seed);
    mutex.lock();
    ::params[idx].seed = seed;
    ::seeds[jobID] = seed;
    mutex.unlock();
    
    // --- SESSIONS ---
/*
    //std::string path_session = "./session.data.bin";
    //std::string path_session = "./"; // FIXME: params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    // NB! Do not store sessions for fast GPU-machines
    if (!isGPU && 
        !sessionFile.empty()) {

        fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, sessionFile.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(sessionFile.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(params.n_ctx);
            //fprintf(stderr, "%s: session_tokens capacity = %d tokens\n", __func__, (int) session_tokens.capacity());

            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, sessionFile.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, sessionFile.c_str());
                //return 1;

                // FIXME: The possible problem - mismatch between models
                // llama_load_session_file_internal : model hparams didn't match from session file!
                // Just ignore the problem for now, session file will be rewritten soon within C++ 
                // and then possibly again (if Go code has another value of token counter) after 2K limit will be reached
                // The better solution is to create new file and sync this event between Go / C++ parts
            }
            session_tokens.resize(n_token_count_out);

            //fprintf(stderr, "%s: %d tokens were restored\n", __func__, n_token_count_out);

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }
*/

    // tokenize the prompt
    const bool add_bos = llama_should_add_bos_token(model);
    // fprintf(stderr, "\n\nADD_BOS: %d\n\n", add_bos);
    // FIXME: Check out what this assert actually means
    ///// if (!llama_model_has_encoder(model)) {
        ///// GGML_ASSERT(llama_add_eos_token(model) != 1);
    /////    fprintf(stderr, "\ndo_inference :: [ 3-1 ] "); // DEBUG
    /////    return NULL;
    ///// }

    bool add_special = false;
    bool parse_special = true;
    std::vector<llama_token> embd_inp;
    embd_inp = ::llama_tokenize(model, prompt, add_special, parse_special);

    // Should not run without any tokens
    ///// if (embd_inp.empty()) {
    /////     embd_inp.push_back(llama_token_bos(model)); // FIXME
    ///// }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() /* || params.instruct || params.chatml */ ) {
        params.n_keep = (int) embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }
/*
    // DEBUG
    fprintf(stderr, "\n\nADD_BOS: %d\n\n", add_bos);
    fprintf(stderr, "\n\nTOKENS: [ ");
    for(int i = 0; i < embd_inp.size(); i++) {
         fprintf(stderr, "%d, ", embd_inp.data()[i]);
    }
    fprintf(stderr, "]");
*/

/*
    auto tokens = ::llama_tokenize(ctx, prompt, true); 
    std::unordered_map<llama_token, size_t> tokmap;
    for(int i = 0; i < tokens.size(); i++) {
        auto id = tokens[i];
        tokmap[id]++;
    }
    std::vector<llama_token_data> candidates;
    candidates.clear();
    for(const auto& elem : tokmap)
    {
        //fprintf(stderr, "\nID %d | \"%s\" | P %d", elem.first, llama_token_to_str(ctx, elem.first).c_str(), elem.second);
        //std::cout << elem.first << " " << elem.second.first << " " << elem.second.second << "\n";
        candidates.emplace_back(llama_token_data{elem.first, (float)elem.second, 0.0f});
    }
    std::sort(
        candidates.data(), 
        candidates.data() + candidates.size(), 
        [](const llama_token_data & a, const llama_token_data & b) { 
            return a.logit > b.logit; 
        }
    );
    fprintf(stderr, "\n\n==== TOKENS %d ====", tokens.size());
    fprintf(stderr, "\n\n==== TOKMAP ====\n");
    for (size_t i = 0; i < candidates.size(); i++) {
        auto id =candidates.data()[i].id;
        auto logit = candidates.data()[i].logit;
        fprintf(stderr, 
            "\n  -- %5d [ %.1f ~ %.2f ] \"%s\"", 
            id,
            logit,
            logit / tokens.size(),
            llama_token_to_piece(ctx, id).c_str()
        );
    }    
    exit(1);
*/    
    
    // -- DEBUG

    if (strstr(::debug, "tokenizer")) {

        // DEBUG JANUS
        // fprintf(stderr, "\n\n === JANUS ===");
        // fprintf(stderr, "\n * janus = %d", ::sparams[idx].janus);
        // fprintf(stderr, "\n * depth = %d", ::sparams[idx].depth);
        // fprintf(stderr, "\n * scale = %f", ::sparams[idx].scale);
        // fprintf(stderr, "\n * lo = %f", ::sparams[idx].lo);
        // fprintf(stderr, "\n * hi = %f", ::sparams[idx].hi);

        fprintf(stderr, "\n\n=== ADD_BOS = %d ===", add_bos);
        fprintf(stderr, "\n\n=== PROMPT ===\n\n%s", prompt.c_str());

        fprintf(stderr, "\n\n=== IDS ===\n\n");
        for(size_t i = 0; i < embd_inp.size(); i++) {
            fprintf(stderr, "%d, ", embd_inp.data()[i]);
        }

        fprintf(stderr, "\n\n=== TOKENS ===\n\n");
        for(size_t i = 0; i < embd_inp.size(); i++) {
            auto id = embd_inp.data()[i];
            fprintf(stderr, " #%d ", id);
            if (id == 13) fprintf(stderr, "{\\n}\n");
            else if (id == 271) fprintf(stderr, "{\\n\\n}\n\n");
            else if (id == 1) fprintf(stderr, "{ BOS #1 }");
            else if (id == 2) fprintf(stderr, "{ EOS #2 }");
            else if (id == 128000) fprintf(stderr, "<|begin_of_text|>");
            else if (id == 128001) fprintf(stderr, "<|end_of_text|>");
            else if (id == 128006) fprintf(stderr, "<|start_header_id|>");
            else if (id == 128007) fprintf(stderr, "<|end_header_id|>");
            else if (id == 128009) fprintf(stderr, "<|eot_id|>");
            else if (id >= 128000) fprintf(stderr, "{ #%d }", id);
            else fprintf(stderr, "{%s}",  llama_token_to_piece(ctx, embd_inp.data()[i]).c_str());
        }
    }

    // const int n_ctx = llama_n_ctx(ctx); // TODO: Set it from ::params[idx] ?
    promptTokenCount[jobID] = embd_inp.size();

    // FIXME: Process the longer context properly and return some meaningful HTTP code to the front-end

    if ((int) embd_inp.size() > (n_ctx - 4)) {
    //if (sessionFile.empty() && ((int) embd_inp.size() > n_ctx - 4)) {  
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 0;
    }
/*
    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!isGPU && 
        session_tokens.size()) {

        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params[idx].prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            fprintf(stderr, "%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        // WAS: llama_kv_cache_tokens_rm(ctx, n_matching_session_tokens, -1);
        llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);
    }

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!isGPU && 
        !embd_inp.empty() && 
        n_matching_session_tokens == embd_inp.size() &&
        session_tokens.size() > embd_inp.size()) {

        session_tokens.resize(embd_inp.size() - 1);
    }
*/
    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = ::params[idx].grp_attn_n;
    const int ga_w = ::params[idx].grp_attn_w;

    if (ga_n != 1 && ga_n <= 0) return 0; // ERR: grp_attn_n must be positive
    if (ga_n != 1 && (ga_w % ga_n != 0)) return 0; // ERR: grp_attn_w must be a multiple of grp_attn_n

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_tokens(n_ctx);
    std::fill(last_tokens.begin(), last_tokens.end(), 0);

    int n_past             = 0;
    int n_consumed         = 0;
    ///// int n_session_consumed = 0;
    ///// int n_past_guidance    = 0;
    int guidance_offset    = 0; // TODO: Implement guidance

    ///// int n_batch            = ::params[idx].n_batch;
    int n_remain           = ::params[idx].n_predict;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    ///// FIXME std::ostringstream output_ss;     g_output_ss     = &output_ss;

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;

    struct llama_sampling_context * ctx_sampling = llama_sampling_init((const struct llama_sampling_params &) sparams);

    // -- fix hallucinations from previously polluted cache 
    llama_kv_cache_clear(ctx);

    // -- MAIN LOOP --

    // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
    // --prompt or --file which uses the same value.
    int max_embd_size = n_ctx - 4;

    while (n_remain && 
        n_past < max_embd_size &&
        !stopInferenceFlags[idx]) { 

        // predict
        if (!embd.empty()) {

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                ///// const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);
            }

            if (ga_n == 1) {

                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                if (n_past + (int) embd.size() + std::max<int>(0, guidance_offset) > n_ctx) {

                    if (params.n_predict == -2) {
                        break;
                    }

                    // WAS: const int n_left    = n_past - params.n_keep - 1;
                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;

                    llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    ///// if (ctx_guidance) {
                    /////    n_past_guidance -= n_discard;
                    ///// }

                    path_session.clear();
                }

            } else {    

                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    llama_kv_cache_seq_add(ctx, 0, ga_i,                n_past,              ib*bd);
                    llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;
                }
            }
/*
            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }
*/ 
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                    return 1;
                }

                n_past += n_eval;

                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                }
            }
/*
            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            } */
        }

        embd.clear();
        embd_guidance.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
/*            
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG("saved session to %s\n", path_session.c_str());
            }
*/
            llama_token id;
            // FIXME: Allow standard samplings
            ///// if (sparams.janus) {
                id = sample_janus_token(
                    ctx,
                    sparams,
                    jparams,
                    last_tokens,
                    embd_inp.size(),
                    n_past,
                    ::params[idx].n_predict);
            ///// } else {
            /////     id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
            ///// }

            // we still need to maintain this for Janus Sampling
            last_tokens.erase(last_tokens.begin());
            last_tokens.push_back(id);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            embd.push_back(id); // add it to the context
            --n_remain; // decrement remaining sampling budget

        } else {

            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // -- update job text buffer
        mutex.lock();
        for (auto id : embd) {
            jobs[jobID] = jobs[jobID] + llama_token_to_piece(ctx, id);
        }
        mutex.unlock();

        //llama_token_eos(model) // DEBUG
        // fprintf(stderr, "\n\nllama_token_eos = %d\n\n", llama_token_eos(model));
        // fprintf(stderr, "\n\nllama_token_eot = %d\n\n", llama_token_eot(model));

        // end of text token
        // WAS: if (!embd.empty() && llama_token_is_eog(model, embd.back())) {
        if (llama_token_is_eog(model, embd.back())) {
            break;
        }
    }
/*
    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG_TEE("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }
*/ 
    const llama_timings timings = llama_get_timings(ctx);

    mutex.lock();
    promptEvals[jobID] = timings.t_p_eval_ms / timings.n_p_eval;
    ::timings[jobID] = timings.t_eval_ms / timings.n_eval;
    mutex.unlock();

    return timings.n_p_eval + timings.n_eval;
}

// lock() is safer that lock_shared() - https://stackoverflow.com/questions/59809405/shared-mutex-in-c

const char * statusCPP(const std::string & jobID) {
    mutex.lock();
    const char * res = jobs[jobID].c_str();
    mutex.unlock();
    return res;
}

int64_t promptEvalCPP(const std::string & jobID) {
    mutex.lock();
    int64_t res = promptEvals[jobID];
    mutex.unlock();
    return res;
}

int64_t getPromptTokenCountCPP(const std::string & jobID) {
    mutex.lock();
    int64_t res = promptTokenCount[jobID];
    mutex.unlock();
    return res;
}

int64_t timingCPP(const std::string & jobID) {
    mutex.lock();
    int64_t res = ::timings[jobID];
    mutex.unlock();
    return res;
}

uint32_t getSeedCPP(const std::string & jobID) {
    mutex.lock();
    uint32_t res = ::seeds[jobID];
    mutex.unlock();
    return res;
}

extern "C" { // ------------------------------------------------------

/*
// numa strategies
enum ggml_numa_strategy {
    GGML_NUMA_STRATEGY_DISABLED   = 0,
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
    GGML_NUMA_STRATEGY_ISOLATE    = 2,
    GGML_NUMA_STRATEGY_NUMACTL    = 3,
    GGML_NUMA_STRATEGY_MIRROR     = 4,
    GGML_NUMA_STRATEGY_COUNT
}; 
*/

void init(char * swap, char * debug) {
    ::debug = debug;
    ::path_session = swap;
    // fprintf(stderr, "\n\nDEBUG: %s\n\n", debug);
    // fprintf(stderr, "\n\nLEN: %d\n\n", strlen(debug));
    if (strlen(debug) < 2) hide();
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED); // TODO: NUMA = params.numa
    show();
}

// TODO: support n_threads_batch
void * initContext(
    int idx, 
    char * modelName, 
    int threads, 
    int batch_size, 
    int gpu1, int gpu2, int gpu3, int gpu4, 
    int context, int predict,
    int32_t mirostat, float mirostat_tau, float mirostat_eta,
    float temperature, int top_k, float top_p,
    float typical_p, 
    float repetition_penalty, int penalty_last_n,
    int32_t janus, int32_t depth, float scale, float hi, float lo,
    uint32_t seed,
    char * debug) {

    ::debug = debug;
    
    ::params[idx].model           = modelName;
    ::params[idx].n_threads       = threads;
    ::params[idx].n_batch         = batch_size;
    ::params[idx].n_threads_batch = ::params[idx].n_threads_batch == -1 ? threads : ::params[idx].n_threads_batch;

    ::params[idx].main_gpu        = 0; // TODO: Main GPU depending on tensor split
    ::params[idx].n_gpu_layers    = gpu1 + gpu2 + gpu3 + gpu4; // TODO: variable number of GPUs
    ::params[idx].tensor_split[0] = gpu1;
    ::params[idx].tensor_split[1] = gpu2;
    ::params[idx].tensor_split[2] = gpu3;
    ::params[idx].tensor_split[3] = gpu4;

    ::params[idx].n_ctx           = context;
    ::params[idx].n_predict       = predict;

    // -- Janus sampling

    ::jparams[idx].janus          = janus;
    ::jparams[idx].depth          = depth;
    ::jparams[idx].scale          = scale;
    ::jparams[idx].hi             = hi;
    ::jparams[idx].lo             = lo;

    // -- other samplings

    ::sparams[idx].mirostat       = mirostat;
    ::sparams[idx].mirostat_tau   = mirostat_tau; 
    ::sparams[idx].mirostat_eta   = mirostat_eta;

    ::sparams[idx].temp           = temperature;
    ::sparams[idx].top_k          = top_k;
    ::sparams[idx].top_p          = top_p;

    ::sparams[idx].typical_p      = typical_p > 0 ? typical_p : 1.0f;

    ::sparams[idx].penalty_repeat  = repetition_penalty;
    ::sparams[idx].penalty_last_n  = penalty_last_n;
    
    ::params[idx].seed            = seed;

    bool showFlag = false;
    if (strstr(debug, "cuda") == NULL) { hide(); showFlag = true; }
    auto res = init_context(idx);
    if (showFlag) { show(); }

    return res;
}

int64_t doInference(
    int idx, 
    void * ctx, 
    char * jobID, 
    char * sessionID, 
    char * prompt) {  
    
    std::string id = jobID;
    std::string text = prompt;
    std::string session = sessionID;
    
    return do_inference(idx, (struct llama_context *)ctx, id, session, text);
}

void stopInference(int idx) {
    ::stopInferenceFlags[idx] = true;
}

// return current result of processing
const char * status(char * jobID) {
    std::string id = jobID;
    return statusCPP(id);
}

// return average PROMPT token processing timing from context
int64_t promptEval(char * jobID) {
    std::string id = jobID;
    return promptEvalCPP(id);
}

// return average PROMPT token processing timing from context
int64_t getPromptTokenCount(char * jobID) {
    std::string id = jobID;
    return getPromptTokenCountCPP(id);
}

// return average OUTPUT token processing timing from context
int64_t timing(char * jobID) {
    std::string id = jobID;
    return timingCPP(id);
}

uint32_t getSeed(char * jobID) {
    std::string id = jobID;
    return getSeedCPP(id);
}

}  // ------------------------------------------------------

/*
// does not write null-terminator to buf
int32_t llama_token_to_piece(const struct llama_model * model, llama_token token, char * buf, int32_t length, bool special) {
    // ref: https://github.com/ggerganov/llama.cpp/pull/7587#discussion_r1620983843
    if (!special && llama_is_control_token(model->vocab, token)) {
        return 0;
    }

    // if we have a cache - use it
    {
        const auto & cache = model->vocab.cache_token_to_piece;

        if (!cache.empty()) {
            const auto & res = cache.at(token);
            if (length < (int) res.size()) {
                return -(int) res.size();
            }
            memcpy(buf, res.c_str(), res.size());
            return res.size();
        }
    }

    if (0 <= token && token < llama_n_vocab(model)) {
        switch (llama_vocab_get_type(model->vocab)) {
            case LLAMA_VOCAB_TYPE_WPM:
            case LLAMA_VOCAB_TYPE_SPM: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (llama_is_normal_token(model->vocab, token)) {
                    std::string result = model->vocab.id_to_token[token].text;
                    llama_unescape_whitespace(result);
                    if (length < (int) result.length()) {
                        return -(int) result.length();
                    }
                    memcpy(buf, result.c_str(), result.length());
                    return result.length();
                } else if (
                        (llama_is_user_defined_token(model->vocab, token)) ||
                        (llama_is_control_token     (model->vocab, token) && special)) {
                    std::string result = model->vocab.id_to_token[token].text;
                    if (length < (int) result.length()) {
                        return -(int) result.length();
                    }
                    memcpy(buf, result.c_str(), result.length());
                    return result.length();
                } else if (llama_is_unknown_token(model->vocab, token)) { // NOLINT
                    if (length < 3) {
                        return -3;
                    }
                    memcpy(buf, "\xe2\x96\x85", 3);
                    return 3;
                } else if (llama_is_byte_token(model->vocab, token)) {
                    if (length < 1) {
                        return -1;
                    }
                    buf[0] = llama_token_to_byte(model->vocab, token);
                    return 1;
                }
                break;
            }
            case LLAMA_VOCAB_TYPE_BPE: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (llama_is_normal_token(model->vocab, token)) {
                    std::string result = model->vocab.id_to_token[token].text;
                    result = llama_decode_text(result);
                    if (length < (int) result.length()) {
                        return -(int) result.length();
                    }
                    memcpy(buf, result.c_str(), result.length());
                    return result.length();
                } else if (
                        (llama_is_user_defined_token(model->vocab, token)) ||
                        (llama_is_control_token     (model->vocab, token) && special)) {
                    std::string result = model->vocab.id_to_token[token].text;
                    if (length < (int) result.length()) {
                        return -(int) result.length();
                    }
                    memcpy(buf, result.c_str(), result.length());
                    return result.length();
                }
                break;
            }
            default:
                GGML_ASSERT(false);
        }
    }
    return 0;
}


// NOTE: avoid ever using this except for building the token_to_piece caches
std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}
*/


/*
std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}
*/

// This works fast and allows for 100% deterministic sampling
llama_token sample_top_token(/*struct llama_context * ctx,*/ const float * logits, const int size) {
      
    //const int64_t t_start_sample_us = ggml_time_us();

    llama_token id = 0;
    float prob = 0;

    for (llama_token i = 1; i < size; i++) {
        if (logits[i] > prob) {
            id = i;
            prob = logits[i];
        }
    }

    //if (ctx) {
    //    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    //}

    return id;
}



// ------------------------------------------------------
/*
struct llama_sampling_context * llama_sampling_init(const struct llama_sampling_params & params) {
    struct llama_sampling_context * result = new llama_sampling_context();

    result->params  = params;
    result->grammar = nullptr;

    // // if there is a grammar, parse it
    // if (!params.grammar.empty()) {
    //     result->parsed_grammar = grammar_parser::parse(params.grammar.c_str());

    //     // will be empty (default) if there are parse errors
    //     if (result->parsed_grammar.rules.empty()) {
    //         fprintf(stderr, "%s: failed to parse grammar\n", __func__);
    //         delete result;
    //         return nullptr;
    //     }

    //     std::vector<const llama_grammar_element *> grammar_rules(result->parsed_grammar.c_rules());

    //     result->grammar = llama_grammar_init(
    //             grammar_rules.data(),
    //             grammar_rules.size(), result->parsed_grammar.symbol_ids.at("root"));
    // }

    result->prev.resize(params.n_prev);

    return result;
} */

// -- FIXME: DUP BRIDGE + JANUS
/*
// no reasons to expose this function in header
static void sampler_queue(
                   struct llama_context * ctx_main,
            const llama_sampling_params & params,
                 llama_token_data_array & cur_p,
                                 size_t   min_keep) {
    const float         temp              = params.temp;
    const float         dynatemp_range    = params.dynatemp_range;
    const float         dynatemp_exponent = params.dynatemp_exponent;
    const int32_t       top_k             = params.top_k;
    const float         top_p             = params.top_p;
    const float         min_p             = params.min_p;
    const float         tfs_z             = params.tfs_z;
    const float         typical_p         = params.typical_p;
    const std::vector<llama_sampler_type> & samplers_sequence = params.samplers_sequence;

    for (auto sampler_type : samplers_sequence) {
        switch (sampler_type) {
            case llama_sampler_type::TOP_K    : llama_sample_top_k    (ctx_main, &cur_p, top_k,     min_keep); break;
            case llama_sampler_type::TFS_Z    : llama_sample_tail_free(ctx_main, &cur_p, tfs_z,     min_keep); break;
            case llama_sampler_type::TYPICAL_P: llama_sample_typical  (ctx_main, &cur_p, typical_p, min_keep); break;
            case llama_sampler_type::TOP_P    : llama_sample_top_p    (ctx_main, &cur_p, top_p,     min_keep); break;
            case llama_sampler_type::MIN_P    : llama_sample_min_p    (ctx_main, &cur_p, min_p,     min_keep); break;
            case llama_sampler_type::TEMPERATURE:
                if (dynatemp_range > 0) {
                    float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                    float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                    llama_sample_entropy(ctx_main, &cur_p, dynatemp_min, dynatemp_max, dynatemp_exponent);
                } else {
                    llama_sample_temp(ctx_main, &cur_p, temp);
                }
                break;
            default : break;
        }
    }
}
*/
/*
static llama_token_data_array llama_sampling_prepare_impl(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool apply_grammar,
                  std::vector<float> * original_logits) {
    const llama_sampling_params & params = ctx_sampling->params;

    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));

    const int32_t penalty_last_n  = params.penalty_last_n < 0 ? params.n_prev : params.penalty_last_n;
    const float   penalty_repeat  = params.penalty_repeat;
    const float   penalty_freq    = params.penalty_freq;
    const float   penalty_present = params.penalty_present;

    const bool    penalize_nl     = params.penalize_nl;

    auto & prev = ctx_sampling->prev;
    auto & cur  = ctx_sampling->cur;

    // Get a pointer to the logits
    float * logits = llama_get_logits_ith(ctx_main, idx);

    if (ctx_sampling->grammar != NULL && !apply_grammar) {
        GGML_ASSERT(original_logits != NULL);
        // Only make a copy of the original logits if we are not applying grammar checks, not sure if I actually have to do this.
        *original_logits = {logits, logits + llama_n_vocab(llama_get_model(ctx_main))};
    }

    // apply params.logit_bias map
    for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
        logits[it->first] += it->second;
    }

    if (ctx_cfg) {
        float * logits_guidance = llama_get_logits_ith(ctx_cfg, idx);
        llama_sample_apply_guidance(ctx_main, logits, logits_guidance, params.cfg_scale);
    }

    cur.clear();

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), false };

    // apply penalties
    const auto& penalty_tokens = params.use_penalty_prompt_tokens ? params.penalty_prompt_tokens : prev;
    const int penalty_tokens_used_size = std::min((int)penalty_tokens.size(), penalty_last_n);
    if (penalty_tokens_used_size) {
        const float nl_logit = logits[llama_token_nl(llama_get_model(ctx_main))];

        llama_sample_repetition_penalties(ctx_main, &cur_p,
                penalty_tokens.data() + penalty_tokens.size() - penalty_tokens_used_size,
                penalty_tokens_used_size, penalty_repeat, penalty_freq, penalty_present);

        if (!penalize_nl) {
            for (size_t idx = 0; idx < cur_p.size; idx++) {
                if (cur_p.data[idx].id == llama_token_nl(llama_get_model(ctx_main))) {
                    cur_p.data[idx].logit = nl_logit;
                    break;
                }
            }
        }
    }

    // apply grammar checks before sampling logic
    if (apply_grammar && ctx_sampling->grammar != NULL) {
        llama_sample_grammar(ctx_main, &cur_p, ctx_sampling->grammar);
    }

    return cur_p;
}
*/
/*
llama_token_data_array llama_sampling_prepare(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool apply_grammar,
                  std::vector<float> * original_logits) {
    return llama_sampling_prepare_impl(ctx_sampling,ctx_main, ctx_cfg, idx, apply_grammar, original_logits);
}

static llama_token llama_sampling_sample_impl(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool is_resampling) {
    const llama_sampling_params & params = ctx_sampling->params;

    const float   temp            = params.temp;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;

    std::vector<float> original_logits;
    auto cur_p = llama_sampling_prepare(ctx_sampling, ctx_main, ctx_cfg, idx, / * apply_grammar= * / is_resampling, &original_logits);
    if (ctx_sampling->grammar != NULL && !is_resampling) {
        GGML_ASSERT(!original_logits.empty());
    }
    llama_token id = 0;
    // Get a pointer to the logits
    float * logits = llama_get_logits_ith(ctx_main, idx);

    if (temp < 0.0) {
        // greedy sampling, with probs
        llama_sample_softmax(ctx_main, &cur_p);
        id = cur_p.data[0].id;
    } else if (temp == 0.0) {
        // greedy sampling, no probs
        id = llama_sample_token_greedy(ctx_main, &cur_p);
    } else {
        if (mirostat == 1) {
            const int mirostat_m = 100;
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat(ctx_main, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &ctx_sampling->mirostat_mu);
        } else if (mirostat == 2) {
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat_v2(ctx_main, &cur_p, mirostat_tau, mirostat_eta, &ctx_sampling->mirostat_mu);
        } else {
            // temperature sampling
            size_t min_keep = std::max(1, params.min_keep);

            sampler_queue(ctx_main, params, cur_p, min_keep);

            id = llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);

            //{
            //    const int n_top = 10;
            //    LOG("top %d candidates:\n", n_top);

            //    for (int i = 0; i < n_top; i++) {
            //        const llama_token id = cur_p.data[i].id;
            //        (void)id; // To avoid a warning that id is unused when logging is disabled.
            //        LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(ctx_main, id).c_str(), cur_p.data[i].p);
            //    }
            //}

            //LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx_main, id).c_str());
        }
    }

    if (ctx_sampling->grammar != NULL && !is_resampling) {
        // Create an array with a single token data element for the sampled id
        llama_token_data single_token_data = {id, logits[id], 0.0f};
        llama_token_data_array single_token_data_array = { &single_token_data, 1, false };

        // Apply grammar constraints to the single token
        llama_sample_grammar(ctx_main, &single_token_data_array, ctx_sampling->grammar);

        // Check if the token is valid according to the grammar by seeing if its logit has been set to -INFINITY
        bool is_valid = single_token_data_array.data[0].logit != -INFINITY;

        // If the token is not valid according to the grammar, perform resampling
        if (!is_valid) {
            ///// LOG("Resampling because token %d: '%s' does not meet grammar rules\n", id, llama_token_to_piece(ctx_main, id).c_str());

            // Restore logits from the copy
            std::copy(original_logits.begin(), original_logits.end(), logits);

            return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, / * is_resampling= * / true);
        }
    }

    ctx_sampling->n_valid = temp == 0.0f ? 0 : cur_p.size;

    return id;
}
*/
