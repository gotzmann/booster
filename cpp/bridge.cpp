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

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
//#include "llama.cpp"

#include "bridge.h"
#include "janus.h"

char * debug; // debug level = "" | "full" | "cuda"

// NEW
static llama_context           ** g_ctx;
static llama_model             ** g_model;
static gpt_params               * g_params;
static std::vector<llama_token> * g_input_tokens;
///// static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting = false;


// -- NB! llama_sample_token() is an older implementation of newer janus.cpp::llama_sampling_sample()

// pos => index of current position within generation window [ 0 .. max )
// max => how many tokens were generated via the last iteration?
//        remember, that sessions might have one or multiple iterations
//        before reaching context limit of 4K tokens

llama_token llama_sample_token(
                  struct llama_context * ctx,
                  struct llama_context * ctx_guidance,
                  struct llama_grammar * grammar,
          struct llama_sampling_params & params,
        const std::vector<llama_token> & last_tokens,
         std::vector<llama_token_data> & candidates,
                            const size_t promptLen,
                            const size_t pos,
                            const size_t max) {                              

    auto model = llama_get_model(ctx);

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));

    const int32_t top_k          = params.top_k <= 0 ? n_vocab : params.top_k;
    const int32_t penalty_last_n = params.penalty_last_n < 0 ? n_ctx : params.penalty_last_n;

/* 
    // DEBUG GQA
    auto hparams = model->hparams;
    fprintf(stderr, "\n\n === GQA HPARAMS ===");
    fprintf(stderr, "\n * n_embd = %d", hparams.n_embd);
    fprintf(stderr, "\n * n_head = %d", hparams.n_head);
    fprintf(stderr, "\n * n_head_kv = %d", hparams.n_head_kv);
    fprintf(stderr, "\n * n_gqa() = n_head/n_head_kv = %d", hparams.n_gqa());
    fprintf(stderr, "\n * n_embd_head() = n_embd/n_head = %d", hparams.n_embd_head());
    fprintf(stderr, "\n * n_embd_gqa() = n_embd/n_gqa() = %d", hparams.n_embd_gqa());

    // DEBUG HPARAMS
    fprintf(stderr, "\n\n === HPARAMS ===");
    fprintf(stderr, "\n * n_ctx = %d", n_ctx);
    fprintf(stderr, "\n * n_vocab = %d", n_vocab);
    fprintf(stderr, "\n * temp = %f", params.temp);
    fprintf(stderr, "\n * top_k = %d", top_k);
    fprintf(stderr, "\n * top_p = %f", params.top_p);
    fprintf(stderr, "\n * penalty_last_n = %d", penalty_last_n);
    fprintf(stderr, "\n * penalty_repeat = %f", params.penalty_repeat);
    fprintf(stderr, "\n * mirostat = %d", params.mirostat);
    fprintf(stderr, "\n * mirostat_eta = %f", params.mirostat_eta);
    fprintf(stderr, "\n * mirostat_tau = %f", params.mirostat_tau); 
*/

    //llama_token id = 0;
    //float * logits = llama_get_logits(ctx);
    //candidates.clear();

    // Experimental sampling both creative for text and pedantic for math / coding
    if (params.janus > 0) {
        return sample_janus_token(ctx, params, last_tokens, promptLen, pos, max);
    }

    llama_token id = 0;
    float * logits = llama_get_logits(ctx);
    ///// candidates.clear();

    // Deterministic sampling with great performance
    if (top_k == 1) {
        return sample_top_token(logits, n_vocab);
    }

    // Apply params.logit_bias map
    //for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
    //    logits[it->first] += it->second;
    //}

    candidates.clear();
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array cur_p = { candidates.data(), candidates.size(), false };

    if (ctx_guidance) {
    /////     llama_sample_classifier_free_guidance(ctx, &cur_p, ctx_guidance, params.cfg_scale);
    }

    // apply penalties
    if (!last_tokens.empty()) {

        const float nl_logit = logits[llama_token_nl(model)];
        const int last_n = std::min(
            std::min(
                (int)last_tokens.size(), 
                penalty_last_n), 
            n_ctx);

        llama_sample_repetition_penalties(
                ctx, 
                &cur_p,
                last_tokens.data() + last_tokens.size() - last_n, 
                last_n, 
                params.penalty_repeat,
                params.penalty_freq, 
                params.penalty_present
            );

        if (!params.penalize_nl) {
            for (size_t idx = 0; idx < cur_p.size; idx++) {
                if (cur_p.data[idx].id == llama_token_nl(model)) {
                    cur_p.data[idx].logit = nl_logit;
                    break;
                }
            }
        }
    }

    if (grammar != NULL) {
        llama_sample_grammar(ctx, &cur_p, grammar);
    }

    if (params.temp <= 0) {
        // Greedy sampling
        id = llama_sample_token_greedy(ctx, &cur_p);
    } else {
        if (params.mirostat == 1) {
            static float mirostat_mu = 2.0f * params.mirostat_tau;
            const int mirostat_m = 100;
            llama_sample_temp(ctx, &cur_p, params.temp);
            id = llama_sample_token_mirostat(ctx, &cur_p, params.mirostat_tau, params.mirostat_eta, mirostat_m, &mirostat_mu);
        } else if (params.mirostat == 2) {
            static float mirostat_mu = 2.0f * params.mirostat_tau;
            // Experimental step!
            if (top_k > 0) {
                llama_sample_top_k(ctx, &cur_p, top_k, 1);
            }
            llama_sample_temp(ctx, &cur_p, params.temp);
            id = llama_sample_token_mirostat_v2(ctx, &cur_p, params.mirostat_tau, params.mirostat_eta, &mirostat_mu);
        } else {
            // Temperature sampling
            llama_sample_top_k      (ctx, &cur_p, top_k, 1);
            //llama_sample_tail_free  (ctx, &cur_p, tfs_z, 1);
            //llama_sample_typical    (ctx, &cur_p, params.typical_p, 1);
            llama_sample_top_p      (ctx, &cur_p, params.top_p, 1);
            llama_sample_temp       (ctx, &cur_p, params.temp);
            id = llama_sample_token (ctx, &cur_p);
        }
    }

    if (grammar != NULL) {
        llama_grammar_accept_token(ctx, grammar, id);
    }

    return id;
}


// FIXME ASAP - do not allow longer context when reading session file

// do_inference: PROMPT [ 2386 ] tokens
// do_inference: SESSION [ 0 ] tokens
// do_inference: error: prompt is too long (2386 tokens, max 2044)
// ggml_new_tensor_impl: not enough space in the scratch memory pool (needed 877775360, available 536870912)
// fatal error: unexpected signal during runtime execution

// Shared map for storing pairs of [UUID] -> [Output] while processing within C++ side
// After returning final result to Go side, we can safely remove the current result from the map

/*mutable*/ std::shared_mutex mutex;

// NB! Always use mutex to access map  thread-safe way

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

void hide() {
    (void) !freopen(NULL_DEVICE, "w", stdout);
    (void) !freopen(NULL_DEVICE, "w", stderr);
}    

void show() {
    (void) !freopen(TTY_DEVICE, "w", stdout);
    (void) !freopen(TTY_DEVICE, "w", stderr);
}

// --- Globals for all pods. Do anyone needs more than 8 pods per machine?

gpt_params params[8];             // main params 
llama_sampling_params sparams[8]; // sampling params
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

    llama_model_params settings = llama_model_default_params();

    settings.main_gpu     = ::params[idx].main_gpu;
    settings.n_gpu_layers = ::params[idx].n_gpu_layers;
    settings.tensor_split = ::params[idx].tensor_split;

    llama_model * model = llama_load_model_from_file(modelName, settings);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, modelName);
        // return std::make_tuple(nullptr, nullptr);
        return NULL;
    }

    models[idx] = model;

    // -- initialize the context

    auto defaults = llama_context_default_params();

    defaults.n_ctx           = ::params[idx].n_ctx;
    defaults.seed            = ::params[idx].seed;
    defaults.n_threads       = ::params[idx].n_threads;
    defaults.n_threads_batch = ::params[idx].n_threads_batch;

    // TODO: Determine best batch size for GPU (and maybe different depending on VRAM size)
    // NB! It crashes with batch of 32/64 and go loop with 128. So use batching of 256 or more

    bool isGPU = ::params[idx].n_gpu_layers > 0 ? true : false;

    if (::params[idx].n_batch > 0 && ::params[idx].n_batch <= ::params[idx].n_ctx) {
        defaults.n_batch = ::params[idx].n_batch;
    } else if (isGPU) {
        defaults.n_batch = 512;
    } else {
        defaults.n_batch = ::params[idx].n_ctx;
    }

    llama_context * ctx = llama_new_context_with_model(model, defaults);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, modelName);
        llama_free_model(model);
        // return std::make_tuple(nullptr, nullptr);
        return NULL;
    }

    contexts[idx] = ctx;

    // return std::make_tuple(model, lctx);
    return ctx;
}

// Process prompt and compute output, return total number of tokens processed
// idx - index of pod / context / params to do processing within
int64_t do_inference(

    int idx, 
    struct llama_context * ctx, 
    const std::string & jobID, 
    const std::string & sessionID, 
    const std::string & text

) {

    llama_reset_timings(ctx);
    stopInferenceFlags[idx] = false;

    bool isGPU     = ::params[idx].n_gpu_layers > 0 ? true : false;
    auto model     = models[idx];
    ///// auto vocabSize = llama_n_vocab(model);

    gpt_params & params = ::params[idx];
    llama_sampling_params & sparams = ::sparams[idx];

    // NEW
    ///// llama_model * model;
    ///// llama_context * ctx;
    llama_context * ctx_guidance = NULL;
    g_model = &model;
    g_ctx = &ctx;
    g_params = &params;
    ///// const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    std::string path_session = ::params[idx].path_prompt_cache;
    std::vector<llama_token> session_tokens;
    // END
    
    // llama_token BOS = llama_token_bos(ctx); 
    // llama_token EOS = llama_token_eos(ctx);

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
    //std::vector<llama_token> embd_inp;
    // WAS: const bool add_bos = llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM;
    //const bool add_bos = llama_should_add_bos_token(model);
    //fprintf(stderr, "\n\n add_bos = %d\n\n", add_bos);
    const bool add_bos = llama_should_add_bos_token(model);
    std::vector<llama_token> embd_inp;
    //embd_inp = llama_tokenize(model, text, add_bos, true);
    embd_inp = ::llama_tokenize(ctx, /*params.prompt*/ text, add_bos, true);
    // ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", add_bos, true);

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model)); // FIXME
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() /* || params.instruct || params.chatml */ ) {
        params.n_keep = (int) embd_inp.size();
    }

    // DEBUG
    // fprintf(stderr, "\n\nTOKENS: [ ");
    // for(int i = 0; i < embd_inp.size(); i++) {
    //     fprintf(stderr, "%d, ", embd_inp.data()[i]);
    // }
    // fprintf(stderr, "]");

/*
    auto tokens = ::llama_tokenize(ctx, doc, true); 
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
            llama_token_to_str(ctx, id).c_str()
        );
    }    
    exit(1);
*/    
    
/*
    // -- DEBUG
    fprintf(stderr, "\n\nTOKENS: [ ");
    for(int i = 0; i < embd_inp.size(); i++) {
        fprintf(stderr, "%d, ", embd_inp.data()[i]);
    }
    fprintf(stderr, "]");

    // -- DEBUG
    fprintf(stderr, "\n\nPARTS: [ ");
    for(int i = 0; i < embd_inp.size(); i++) {
        fprintf(stderr, "<%s>, ", llama_token_to_str(ctx, embd_inp.data()[i]).c_str());
    }
    fprintf(stderr, "]");
*/
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

    struct llama_sampling_context * ctx_sampling = llama_sampling_init((const struct llama_sampling_params) sparams);

    // -- fix hallucinations from previously polluted cache 
    llama_kv_cache_clear(ctx);

    // -- MAIN LOOP --

    while (n_remain && 
        // n_past < (n_ctx - 4) && // FIXME
        !stopInferenceFlags[idx]) { 

        // predict
        if (!embd.empty()) {

            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                ///// const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                // console::set_display(console::error);
                // printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                // console::set_display(console::reset);
                // fflush(stdout);
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

                    const int n_left    = n_past - params.n_keep - 1;
                    const int n_discard = n_left/2;

                    llama_kv_cache_seq_rm   (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
                    llama_kv_cache_seq_shift(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

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

                    llama_kv_cache_seq_shift(ctx, 0, ga_i,                n_past,              ib*bd);
                    llama_kv_cache_seq_div  (ctx, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_kv_cache_seq_shift(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

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
*/ /*
            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            if (ctx_guidance) {
                int input_size = 0;
                llama_token * input_buf = NULL;

                if (n_past_guidance < (int) guidance_inp.size()) {
                    // Guidance context should have the same data with these modifications:
                    //
                    // * Replace the initial prompt
                    // * Shift everything by guidance_offset
                    embd_guidance = guidance_inp;
                    if (embd.begin() + original_prompt_len < embd.end()) {
                        embd_guidance.insert(
                            embd_guidance.end(),
                            embd.begin() + original_prompt_len,
                            embd.end()
                        );
                    }

                    input_buf  = embd_guidance.data();
                    input_size = embd_guidance.size();

                    LOG("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_guidance).c_str());
                } else {
                    input_buf  = embd.data();
                    input_size = embd.size();
                }

                for (int i = 0; i < input_size; i += params.n_batch) {
                    int n_eval = std::min(input_size - i, params.n_batch);
                    if (llama_decode(ctx_guidance, llama_batch_get_one(input_buf + i, n_eval, n_past_guidance, 0))) {
                        LOG_TEE("%s : failed to eval\n", __func__);
                        return 1;
                    }

                    n_past_guidance += n_eval;
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
            if (!sparams.janus) {
                id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
            } else {
                /* Collider
                struct llama_context * guidance = NULL;
                struct llama_grammar * grammar = NULL;
                llama_token id = llama_sample_token(
                    ctx,
                    guidance,
                    grammar,
                    ::sparams[idx],
                    last_tokens,
                    candidates,
                    embd_inp.size(),
                    n_past / * - n_consumed * /,
                    ::params[idx].n_predict); */

                struct llama_context * guidance = NULL;
                struct llama_grammar * grammar = NULL;
                id = llama_sample_token(
                    ctx,
                    guidance,
                    grammar,
                    ::sparams[idx],
                    last_tokens,
                    ctx_sampling->cur, // candidates
                    embd_inp.size(),
                    n_past,
                    ::params[idx].n_predict);    
            }

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
/*
        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = llama_token_to_piece(ctx, id);
                printf("%s", token_str.c_str());

                if (embd.size() > 1) {
                    input_tokens.push_back(id);
                } else {
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
            fflush(stdout);
        }
        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
            display = true;
        }
*/
       // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
/*            
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG("found antiprompt: %s\n", last_output.c_str());
                }
            }
*/ /*
            // deal with end of text token in interactive mode
            if (llama_sampling_last(ctx_sampling) == llama_token_eos(model)) {
                //LOG("found EOS token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    is_interacting = true;
                    printf("\n");
                } else if (params.instruct || params.chatml) {
                    is_interacting = true;
                }
            }
*/ /*
            if (n_past > 0 && is_interacting) {
                LOG("waiting for user input\n");

                if (params.instruct || params.chatml) {
                    printf("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_token_bos(model));
                }

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    LOG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    printf("%s", params.input_prefix.c_str());
                }

                // color user input only
                console::set_display(console::user_input);
                display = params.display_prompt;

                std::string line;
                bool another_line = true;
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);

                // done taking input, reset color
                console::set_display(console::reset);
                display = true;

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty()) {
                        LOG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        printf("%s", params.input_suffix.c_str());
                    }

                    LOG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    // instruct mode: insert instruction prefix
                    if (params.instruct && !is_antiprompt) {
                        LOG("inserting instruction prefix\n");
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }
                    // chatml mode: insert user chat prefix
                    if (params.chatml && !is_antiprompt) {
                        LOG("inserting chatml prefix\n");
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), cml_pfx.begin(), cml_pfx.end());
                    }
                    if (params.escape) {
                        process_escapes(buffer);
                    }

                    const auto line_pfx = ::llama_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = ::llama_tokenize(ctx, buffer,              false, false);
                    const auto line_sfx = ::llama_tokenize(ctx, params.input_suffix, false, true);
                    LOG("input tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, line_inp).c_str());

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    // instruct mode: insert response suffix
                    if (params.instruct) {
                        LOG("inserting instruction suffix\n");
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }
                    // chatml mode: insert assistant chat suffix
                    if (params.chatml) {
                        LOG("inserting chatml suffix\n");
                        embd_inp.insert(embd_inp.end(), cml_sfx.begin(), cml_sfx.end());
                    }

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << llama_token_to_piece(ctx, token);
                    }

                    n_remain -= line_inp.size();
                    LOG("n_remain: %d\n", n_remain);
                } else {
                    LOG("empty line, passing control back\n");
                }

                input_echo = false; // do not echo this again
            }
*/ /*
            if (n_past > 0) {
                if (is_interacting) {
                    llama_sampling_reset(ctx_sampling);
                }
                is_interacting = false;
            } */
        }

        // -- Collider: update job text buffer
        mutex.lock();
        for (auto id : embd) {
            //if (id == BOS || id == EOS) { fprintf(stderr, "\n\n... SKIPPING BOS or EOS ..."); continue; };
            jobs[jobID] = jobs[jobID] + llama_token_to_piece(ctx, id);
            //fprintf(stderr, "\n\n... ADDING [[[%s]]] ...", llama_token_to_str(ctx, id).c_str());
        }
        mutex.unlock();

        // end of text token
        // if (!embd.empty() && embd.back() == llama_token_eos(model) && !(params.instruct || params.interactive || params.chatml)) {
        if (!embd.empty() && embd.back() == llama_token_eos(model)) {    
            break;
        }
/*
        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        } */
    }
/*
    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG_TEE("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }
*/ /*
    llama_print_timings(ctx);
    write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);

    if (ctx_guidance) { llama_free(ctx_guidance); }
    llama_free(ctx);
    llama_free_model(model);

    llama_sampling_free(ctx_sampling);
    llama_backend_free();

#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log end\n");
#endif // LOG_DISABLE_LOGS

    return 0;
} */

    const llama_timings timings = llama_get_timings(ctx);

    mutex.lock();
    promptEvals[jobID] = timings.t_p_eval_ms / timings.n_p_eval;
    ::timings[jobID] = timings.t_eval_ms / timings.n_eval;
    mutex.unlock();

    return timings.n_p_eval + timings.n_eval;
}

// TODO: Safer lock/unlock - https://stackoverflow.com/questions/59809405/shared-mutex-in-c

const char * statusCPP(const std::string & jobID) {
    mutex.lock_shared();
    const char * res = jobs[jobID].c_str();
    mutex.unlock_shared();
    return res;
}

int64_t promptEvalCPP(const std::string & jobID) {
    mutex.lock_shared();
    int64_t res = promptEvals[jobID];
    mutex.unlock_shared();
    return res;
}

int64_t getPromptTokenCountCPP(const std::string & jobID) {
    mutex.lock_shared();
    int64_t res = promptTokenCount[jobID];
    mutex.unlock_shared();
    return res;
}

int64_t timingCPP(const std::string & jobID) {
    mutex.lock_shared();
    int64_t res = ::timings[jobID];
    mutex.unlock_shared();
    return res;
}

uint32_t getSeedCPP(const std::string & jobID) {
    mutex.lock_shared();
    uint32_t res = ::seeds[jobID];
    mutex.unlock_shared();
    return res;
}

extern "C" { // ------------------------------------------------------

void init(char * swap, char * debug) {
    ::debug = debug;
    ::path_session = swap;
    bool showFlag = false;
    if (strstr(debug, "cuda") != NULL) { hide(); showFlag = true; }
    llama_backend_init(false); // NUMA = false
    if (showFlag) { show(); }
}

// TODO: support n_threads_batch
void * initContext(
    int idx, 
    char * modelName, 
    int threads, 
    int batch_size, 
    int gpu1, int gpu2, 
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
    ::params[idx].n_gpu_layers    = gpu1 + gpu2;
    ::params[idx].tensor_split[0] = gpu1;
    ::params[idx].tensor_split[1] = gpu2;

    ::params[idx].n_ctx           = context;
    ::params[idx].n_predict       = predict;

    // -- Janus sampling

    ::sparams[idx].janus          = janus;
    ::sparams[idx].depth          = depth;
    ::sparams[idx].scale          = scale;
    ::sparams[idx].hi             = hi;
    ::sparams[idx].lo             = lo;

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
    if (strstr(debug, "cuda") != NULL) { hide(); showFlag = true; }
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

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos,
                        bool   special) {
    return llama_tokenize(llama_get_model(ctx), text, add_bos, special);
}

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_bos,
                        bool   special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

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

bool llama_should_add_bos_token(const llama_model * model) {
    const int add_bos = llama_add_bos_token(model);

    return add_bos != -1 ? bool(add_bos) : (llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM);
}

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

void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos,
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

// ------------------------------------------------------

struct llama_sampling_context * llama_sampling_init(const struct llama_sampling_params & params) {
    struct llama_sampling_context * result = new llama_sampling_context();

    result->params  = params;
    result->grammar = nullptr;
/*
    // if there is a grammar, parse it
    if (!params.grammar.empty()) {
        result->parsed_grammar = grammar_parser::parse(params.grammar.c_str());

        // will be empty (default) if there are parse errors
        if (result->parsed_grammar.rules.empty()) {
            fprintf(stderr, "%s: failed to parse grammar\n", __func__);
            delete result;
            return nullptr;
        }

        std::vector<const llama_grammar_element *> grammar_rules(result->parsed_grammar.c_rules());

        result->grammar = llama_grammar_init(
                grammar_rules.data(),
                grammar_rules.size(), result->parsed_grammar.symbol_ids.at("root"));
    }
*/
    result->prev.resize(params.n_prev);

    return result;
}

// -- FIXME: DUP

// no reasons to expose this function in header
static void sampler_queue(
                   struct llama_context * ctx_main,
            const llama_sampling_params & params,
                 llama_token_data_array & cur_p,
                                 size_t & min_keep) {
    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));

    const float         temp              = params.temp;
    const float         dynatemp_range    = params.dynatemp_range;
    const float         dynatemp_exponent = params.dynatemp_exponent;
    const int32_t       top_k             = params.top_k <= 0 ? n_vocab : params.top_k;
    const float         top_p             = params.top_p;
    const float         min_p             = params.min_p;
    const float         tfs_z             = params.tfs_z;
    const float         typical_p         = params.typical_p;
    const std::string & samplers_sequence = params.samplers_sequence;

    for (auto s : samplers_sequence) {
        switch (s){
            case 'k': llama_sample_top_k    (ctx_main, &cur_p, top_k,     min_keep); break;
            case 'f': llama_sample_tail_free(ctx_main, &cur_p, tfs_z,     min_keep); break;
            case 'y': llama_sample_typical  (ctx_main, &cur_p, typical_p, min_keep); break;
            case 'p': llama_sample_top_p    (ctx_main, &cur_p, top_p,     min_keep); break;
            case 'm': llama_sample_min_p    (ctx_main, &cur_p, min_p,     min_keep); break;
            case 't':
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

static llama_token llama_sampling_sample_impl(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool is_resampling) {  // Add a parameter to indicate if we are resampling
    const llama_sampling_params & params = ctx_sampling->params;

    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));

    const float   temp            = params.temp;
    const int32_t penalty_last_n  = params.penalty_last_n < 0 ? params.n_prev : params.penalty_last_n;
    const float   penalty_repeat  = params.penalty_repeat;
    const float   penalty_freq    = params.penalty_freq;
    const float   penalty_present = params.penalty_present;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;
    const bool    penalize_nl     = params.penalize_nl;

    auto & prev = ctx_sampling->prev;
    auto & cur  = ctx_sampling->cur;

    llama_token id = 0;

    // Get a pointer to the logits
    float * logits = llama_get_logits_ith(ctx_main, idx);

    // Declare original_logits at the beginning of the function scope
    std::vector<float> original_logits;

    if (!is_resampling) {
        // Only make a copy of the original logits if we are not in the resampling phase, not sure if I actually have to do this.
        original_logits = std::vector<float>(logits, logits + llama_n_vocab(llama_get_model(ctx_main)));
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

    // If we are in the resampling phase, apply grammar checks before sampling logic
    if (is_resampling && ctx_sampling->grammar != NULL) {
        llama_sample_grammar(ctx_main, &cur_p, ctx_sampling->grammar);
    }

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
            size_t min_keep = std::max(1, params.n_probs);

            sampler_queue(ctx_main, params, cur_p, min_keep);

            id = llama_sample_token(ctx_main, &cur_p);

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
            //LOG("Resampling because token %d: '%s' does not meet grammar rules\n", id, llama_token_to_piece(ctx_main, id).c_str());

            // Restore logits from the copy
            std::copy(original_logits.begin(), original_logits.end(), logits);

            return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, true);  // Pass true for is_resampling
        }
    }

    return id;
}

llama_token llama_sampling_sample(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx) {
    // Call the implementation function with is_resampling set to false by default
    return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, false);
}

void llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar) {
    ctx_sampling->prev.erase(ctx_sampling->prev.begin());
    ctx_sampling->prev.push_back(id);

    if (ctx_sampling->grammar != NULL && apply_grammar) {
        llama_grammar_accept_token(ctx_main, ctx_sampling->grammar, id);
    }
}

llama_token llama_sampling_last(llama_sampling_context * ctx) {
    return ctx->prev.back();
}


// ------------------------------------------------------

int32_t get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32)
    //TODO: Implement
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

