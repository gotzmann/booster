#include "bridge.h"
#include "llama.h"
#include "ggml.h"

#include <string>
#include <vector>
#include <random>
#include <thread>
#include <shared_mutex>
#include <unordered_map>
#include <tuple>

std::vector<llama_token> llama_tokenize(
        struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(ctx, text.c_str(), result.data(), result.size(), add_bos);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(ctx, text.c_str(), result.data(), result.size(), add_bos);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_str(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(ctx, token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(ctx, token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
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

// Map of vectors storing PROMPT token count
std::unordered_map<std::string, int64_t> promptTokenCount;

// Map of vectors storing OUTPUT token count
std::unordered_map<std::string, int64_t> outputTokenCount;

// Suspend stdout / stderr messaging
// https://stackoverflow.com/questions/70371091/silencing-stdout-stderr

// FIXME: Redirect C++ stderr into log file 
void hide() {
    freopen(NULL_DEVICE, "w", stdout);
    freopen(NULL_DEVICE, "w", stderr);
}    

// FIXME: Doesn't work for MacOS ?
void show() {
    freopen(TTY_DEVICE, "w", stdout);
    freopen(TTY_DEVICE, "w", stderr);
}

// --- Global params for all pods. Do anyone needs more than 8 pods per machine?

gpt_params params[8];
llama_model * models[8];
llama_context * contexts[8];

// Flags to stop particular inference thread from the Go code

bool stopInferenceFlags[8];

// Directory where session data files will be held. Emtpy string if sessions are disabled

std::string path_session;

struct llama_context * init_context(int idx) {

    bool isGPU = params[idx].n_gpu_layers > 0 ? true : false;

    auto lparams = llama_context_default_params();

    if (isGPU) {
        lparams.mul_mat_q = true; // FIXME: Experimental, move to config!
    }

    // NB! [lparams] is of type llama_context_params and have no all parameters from bigger gpt_params
    //     [params]  is of type gpt_params and has n_threads parameter

    lparams.n_ctx        = params[idx].n_ctx;
    lparams.seed         = params[idx].seed;

    // TODO: Determine best batch size for GPU (and maybe different depending on VRAM size)
    // NB! It crashes with batch of 32/64 and go loop with 128. So use batching of 256 or more
    lparams.n_batch = isGPU ? 512 : params[idx].n_ctx;

    // -- Init GPU inference params right

    // 100% model layers should be placed into the one GPU
    // and main_gpu (for computing scratch buffers) is always 
    // the same as GPU for big tensors compute

    // int32_t n_gpu_layers;                    // number of layers to store in VRAM
    // int32_t main_gpu;                        // the GPU that is used for scratch and small tensors
    // float   tensor_split[LLAMA_MAX_DEVICES]; // how to split layers across multiple GPUs

    lparams.main_gpu = params[idx].main_gpu;
    lparams.n_gpu_layers = params[idx].n_gpu_layers;

    //for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
    //    lparams.tensor_split[i] = 0.0f;
    //}

    //lparams.tensor_split[0] = params[idx].tensor_split[0];
    //lparams.tensor_split[1] = params[idx].tensor_split[1];

    lparams.tensor_split = params[idx].tensor_split;

    fprintf(stderr, "== %s: n_ctx = %d\n", __func__, (int) lparams.n_ctx);
    fprintf(stderr, "== %s: n_batch = %d\n", __func__, (int) lparams.n_batch);
    //fprintf(stderr, "\n== %s: params[%d].main_gpu = %d\n", __func__, (int) idx, (int) params[idx].main_gpu);
    //fprintf(stderr, "== %s: params[%d].gpu_layers = %d\n\n", __func__, (int) idx, (int) params[idx].n_gpu_layers);

    llama_model * model  = llama_load_model_from_file(params[idx].model.c_str(), lparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params[idx].model.c_str());
        // return std::make_tuple(nullptr, nullptr);
        return NULL;
    }

    models[idx] = model;

    llama_context * lctx = llama_new_context_with_model(model, lparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params[idx].model.c_str());
        llama_free_model(model);
        // return std::make_tuple(nullptr, nullptr);
        return NULL;
    }

    contexts[idx] = lctx;

    // return std::make_tuple(model, lctx);
    return lctx;
}

// Process prompt and compute output, return total number of tokens processed
// idx - index of pod / context / params to do processing within
int64_t do_inference(
    int idx, 
    struct llama_context * ctx, 
    const std::string & jobID, 
    const std::string & sessionID, 
    const std::string & text) {

    stopInferenceFlags[idx] = false;
    bool isGPU = params[idx].n_gpu_layers > 0 ? true : false;
    // llama_token BOS = llama_token_bos(ctx); 
    // llama_token EOS = llama_token_eos(ctx);

    llama_reset_timings(ctx);

    std::string sessionFile;
    if (!isGPU &&
        !path_session.empty() && 
        !sessionID.empty()) {

        sessionFile = path_session + '/' + sessionID;
    }

    if (::params[idx].seed <= 0) {
        ::params[idx].seed = time(NULL);
    }

    llama_set_rng_seed(ctx, ::params[idx].seed);

    // --- SESSIONS ---

    //std::string path_session = "./session.data.bin";
    //std::string path_session = "./"; // FIXME: params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    // NB! Do not store sessions for fast GPU-machines
    if (!isGPU && 
        !sessionFile.empty()) {

        fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, /*path_session*/sessionFile.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(/*path_session*/sessionFile.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            // FIXME: Allow to store 2x context size to allow experiments with context swap, etc...
            // session_tokens.resize(2 * params[idx].n_ctx);

            session_tokens.resize(params[idx].n_ctx);
            fprintf(stderr, "%s: session_tokens capacity = %d tokens\n", __func__, (int) session_tokens.capacity());

            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, /*path_session*/sessionFile.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, /*path_session*/sessionFile.c_str());
                //return 1;

                // FIXME: The possible problem - mismatch between models
                // llama_load_session_file_internal : model hparams didn't match from session file!
                // Just ignore the problem for now, session file will be rewritten soon within C++ 
                // and then possibly again (if Go code has another value of token counter) after 2K limit will be reached
                // The better solution is to create new file and sync this event between Go / C++ parts
            }
            session_tokens.resize(n_token_count_out);
            // FIXME: llama_set_rng_seed(ctx, params.seed);

            //fprintf(stderr, "%s: %d tokens were restored\n", __func__, n_token_count_out);

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    // tokenize the prompt
    std::vector<llama_token> embd_inp;
    embd_inp = ::llama_tokenize(ctx, text, true); // leading space IS already there thanks Go preprocessing
    // embd_inp = ::llama_tokenize(ctx, text, false); // leading space IS already there thanks Go preprocessing
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
    const int n_ctx = llama_n_ctx(ctx);
    promptTokenCount[jobID] = embd_inp.size();

    // FIXME: Process the longer context properly and return some meaningful HTTP code to the front-end

    if ((int) embd_inp.size() > (n_ctx - 4)) {
    //if (sessionFile.empty() && ((int) embd_inp.size() > n_ctx - 4)) {  
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        //return 1;
        return 0;
    }

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
    }

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!isGPU && 
        !embd_inp.empty() && 
        n_matching_session_tokens == embd_inp.size() &&
        session_tokens.size() > embd_inp.size()) {

        session_tokens.resize(embd_inp.size() - 1);
    }

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    int n_past             = 0;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    int n_batch            = ::params[idx].n_batch;
    int n_remain           = ::params[idx].n_predict;

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance; // TODO: Investigate

    // -- MAIN LOOP --

    while (n_remain && 
        n_past < (n_ctx - 4) &&
        !stopInferenceFlags[idx]) { 

        // predict
        if (embd.size() > 0) {  

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (!isGPU && n_session_consumed < (int) session_tokens.size()) {

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

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always

            // if (ctx_guidance) { ... } // TODO: Investigate

            for (int i = 0; i < (int) embd.size(); i += n_batch) {

                int n_eval = (int) embd.size() - i;
                if (n_eval > n_batch) {
                    n_eval = n_batch;
                }

                if (llama_eval(ctx, &embd[i], n_eval, n_past, ::params[idx].n_threads)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return 0;
                }

                n_past += n_eval;
            }

            if (!isGPU && embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();
        //embd_guidance.clear(); // -- new feature

        //fprintf(stderr, "%s === embd_inp.size() = %d | n_consumed = %d | n_remain = %d \n", __func__, (int) embd_inp.size(), (int) n_consumed, (int) n_remain); // DEBUG

        if ((int) embd_inp.size() <= n_consumed) {

            // --- out of user input, sample next token

            const float   temp            = ::params[idx].temp;
            const int32_t top_k           = ::params[idx].top_k <= 0 ? llama_n_vocab(ctx) : ::params[idx].top_k;
            const float   top_p           = ::params[idx].top_p;

            const int32_t mirostat        = ::params[idx].mirostat;
            const float   mirostat_tau    = ::params[idx].mirostat_tau;
            const float   mirostat_eta    = ::params[idx].mirostat_eta;

            const float   repeat_penalty  = ::params[idx].repeat_penalty;
            const int32_t repeat_last_n   = ::params[idx].repeat_last_n < 0 ? n_ctx : ::params[idx].repeat_last_n;

            //const float   tfs_z           = ::params.tfs_z;
            //const float   typical_p       = ::params.typical_p;
            //const float   alpha_presence  = ::params.presence_penalty;
            //const float   alpha_frequency = ::params.frequency_penalty;
            ////const bool    penalize_nl     = ::params[idx].penalize_nl;

            llama_token id = 0;

            {
                auto logits  = llama_get_logits(ctx); // FIXME: Are there problem if logits not cleared after another request ??
                auto n_vocab = llama_n_vocab(ctx);

                // FIXME: local logit_bias VS gpt_params context bias
                // Apply params.logit_bias map
                ////for (auto it = /*::params.*/logit_bias.begin(); it != /*::params.*/logit_bias.end(); it++) {
                ////    logits[it->first] += it->second;
                ////}

                // FIXME: Do we always need to copy logits into candidates ??
                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // if (ctx_guidance) {
                //     llama_sample_classifier_free_guidance(ctx, &candidates_p, ctx_guidance, params.cfg_scale);
                // }

                // --- Apply penalties

                //float nl_logit = logits[llama_token_nl(ctx)];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);

                llama_sample_repetition_penalty(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                //llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                //    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                //    last_n_repeat, alpha_frequency, alpha_presence);
                //if (!penalize_nl) {
                //    logits[llama_token_nl()] = nl_logit;
                //}

                //if (grammar != NULL) {
                //    llama_sample_grammar(ctx, &candidates_p, grammar);
                //}

                //if (temp <= 0) {
                    // Greedy sampling
                //    id = llama_sample_token_greedy(ctx, &candidates_p);
                //} else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        //llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        //llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                //}
                // printf("`%d`", candidates_p.size);

                //if (grammar != NULL) {
                //    llama_grammar_accept_token(ctx, grammar, id);
                //}

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            } 

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;

        } else { 

            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= n_batch) {
                    break;
                }
            }
        }

        // -- update LLaMAZoo job text buffer
        mutex.lock();
        for (auto id : embd) {
            //if (id == BOS || id == EOS) { fprintf(stderr, "\n\n... SKIPPING BOS or EOS ..."); continue; };
            jobs[jobID] = jobs[jobID] + llama_token_to_str(ctx, id);
            //fprintf(stderr, "\n\n... ADDING [[[%s]]] ...", llama_token_to_str(ctx, id).c_str());
        }
        mutex.unlock();

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos(ctx)) {
                break;
        }
    }

    if (!isGPU && !sessionFile.empty()) {
        fprintf(stderr, "\n%s: saving final output [ %d tokens ] to session file '%s'\n", __func__, (int) session_tokens.size(), sessionFile.c_str());
        llama_save_session_file(ctx, sessionFile.c_str(), session_tokens.data(), session_tokens.size());
    }

    const llama_timings timings = llama_get_timings(ctx);
/*
    load time = %8.2f ms\n", timings.t_load_ms);
    sample time = %8.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            timings.t_sample_ms, timings.n_sample, timings.t_sample_ms / timings.n_sample, 1e3 / timings.t_sample_ms * timings.n_sample);
    prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
    eval time = %8.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval);
    total time = %8.2f ms\n", (timings.t_end_ms - timings.t_start_ms));
*/

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

// TODO: Safer lock/unlock - https://stackoverflow.com/questions/59809405/shared-mutex-in-c
int64_t promptEvalCPP(const std::string & jobID) {
    mutex.lock_shared();
    int64_t res = promptEvals[jobID];
    mutex.unlock_shared();
    return res;
}

// TODO: Safer lock/unlock - https://stackoverflow.com/questions/59809405/shared-mutex-in-c
int64_t getPromptTokenCountCPP(const std::string & jobID) {
    mutex.lock_shared();
    int64_t res = promptTokenCount[jobID];
    mutex.unlock_shared();
    return res;
}

// TODO: Safer lock/unlock - https://stackoverflow.com/questions/59809405/shared-mutex-in-c
int64_t timingCPP(const std::string & jobID) {
    mutex.lock_shared();
    int64_t res = ::timings[jobID];
    mutex.unlock_shared();
    return res;
}

extern "C" { // ------------------------------------------------------

void init(char * sessionPath) {
    ::path_session = sessionPath; 
    llama_backend_init(false); // NUMA = false
}

void * initContext(
    int idx, 
    char * modelName, 
    int threads, 
    int gpu1, int gpu2, 
    int context, int predict,
    int32_t mirostat, float mirostat_tau, float mirostat_eta,
    float temp, int top_k, float top_p, 
    float repeat_penalty, int repeat_last_n,
    int32_t seed) {
    
    ::params[idx].model          = modelName;
    ::params[idx].n_threads      = threads;

    ::params[idx].main_gpu       = 0; // TODO: Main GPU depending on tensor split
    ::params[idx].n_gpu_layers   = gpu1 + gpu2;
    ::params[idx].tensor_split[0] = gpu1;
    ::params[idx].tensor_split[1] = gpu2;

    ::params[idx].n_ctx          = context;
    ::params[idx].n_predict      = predict;

    ::params[idx].mirostat       = mirostat;
    ::params[idx].mirostat_tau   = mirostat_tau; 
    ::params[idx].mirostat_eta   = mirostat_eta;

    ::params[idx].temp           = temp;
    ::params[idx].top_k          = top_k;
    ::params[idx].top_p          = top_p;

    ::params[idx].repeat_penalty = repeat_penalty;
    ::params[idx].repeat_last_n  = repeat_last_n;
    
    ::params[idx].seed           = seed;
    
    hide();
    auto res = init_context(idx);
    show();

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

}  // ------------------------------------------------------
