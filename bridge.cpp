#include <array>
#include <algorithm>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <shared_mutex>
#include <unordered_map>
#include <tuple>

#include "ggml.h"
#include "bridge.h"
#include "janus.h"
#include "llama.h"
#include "llama.cpp"

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

    /* DEBUG GQA
    auto hparams = model->hparams;
    fprintf(stderr, "\n\n === GQA HPARAMS ===");
    fprintf(stderr, "\n * n_embd = %d", hparams.n_embd);
    fprintf(stderr, "\n * n_head = %d", hparams.n_head);
    fprintf(stderr, "\n * n_head_kv = %d", hparams.n_head_kv);
    fprintf(stderr, "\n * n_gqa() = n_head/n_head_kv = %d", hparams.n_gqa());
    fprintf(stderr, "\n * n_embd_head() = n_embd/n_head = %d", hparams.n_embd_head());
    fprintf(stderr, "\n * n_embd_gqa() = n_embd/n_gqa() = %d", hparams.n_embd_gqa()); */

    /* DEBUG HPARAMS
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
    fprintf(stderr, "\n * mirostat_tau = %f", params.mirostat_tau); */

    llama_token id = 0;
    float * logits = llama_get_logits(ctx);
    candidates.clear();

    // Experimental sampling both creative for text and pedantic for math / coding
    if (params.janus > 0) {
        return sample_janus_token(ctx, params, last_tokens, promptLen, pos, max);
    }

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
        llama_sample_classifier_free_guidance(ctx, &cur_p, ctx_guidance, params.cfg_scale);
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
    freopen(NULL_DEVICE, "w", stdout);
    freopen(NULL_DEVICE, "w", stderr);
}    

void show() {
    freopen(TTY_DEVICE, "w", stdout);
    freopen(TTY_DEVICE, "w", stderr);
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
    const std::string & text) {

    stopInferenceFlags[idx] = false;

    llama_reset_timings(ctx);

    bool isGPU     = params[idx].n_gpu_layers > 0 ? true : false;
    auto model     = models[idx];
    auto vocabSize = llama_n_vocab(model);
    
    // llama_token BOS = llama_token_bos(ctx); 
    // llama_token EOS = llama_token_eos(ctx);

    std::string sessionFile;
    if (!isGPU &&
        !path_session.empty() && 
        !sessionID.empty()) {

        sessionFile = path_session + '/' + sessionID;
    }

    // TODO: Do not always use RANDOM seed ?!
    //if (::params[idx].seed <= 0) {
    auto seed = time(NULL);
    llama_set_rng_seed(ctx, seed);
    mutex.lock();
    ::params[idx].seed = seed;
    ::seeds[jobID] = seed;
    mutex.unlock();
    //}
    
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

            session_tokens.resize(params[idx].n_ctx);
            //fprintf(stderr, "%s: session_tokens capacity = %d tokens\n", __func__, (int) session_tokens.capacity());

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

            //fprintf(stderr, "%s: %d tokens were restored\n", __func__, n_token_count_out);

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    // tokenize the prompt
    std::vector<llama_token> embd_inp;
    const bool add_bos = llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM;
    //fprintf(stderr, "\n\n add_bos = %d\n\n", add_bos);
    embd_inp = llama_tokenize(model, text, add_bos, true);

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

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_tokens_rm(ctx, n_matching_session_tokens, -1);
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
    std::vector<llama_token> last_tokens(n_ctx);
    std::fill(last_tokens.begin(), last_tokens.end(), 0);

    int n_past             = 0;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    int n_batch            = ::params[idx].n_batch;
    int n_remain           = ::params[idx].n_predict;

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;

    // llama_sampling_context ctx_sampling = llama_sampling_context_init(params, grammar);

    // -- fix hallucinations from previously polluted cache 
    llama_kv_cache_tokens_rm(ctx, -1, -1);
    //fprintf(stderr, "\nllama_kv_cache_tokens_rm(ctx, -1, -1)");
    // llama_kv_cache_tokens_rm(ctx, n_past, -1);
    //fprintf(stderr, "\nllama_kv_cache_tokens_rm(ctx, %d, -1)", n_past);

    // -- batching
/*
    llama_batch batch = llama_batch_init(n_batch, 0);
    batch.n_tokens = embd_inp.size();

    for (int32_t i = 0; i < batch.n_tokens; i++) {
        batch.token[i]  = embd_inp[i];
        batch.pos[i]    = i;
        batch.seq_id[i] = 0;
        batch.logits[i] = false;
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;
*/
    // -- MAIN LOOP --

/* NEWER

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    // total length of the sequence including the prompt
    const int n_len = 32;
    while (n_cur <= n_len) {
*/
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

                // remove any "future" tokens that we might have inherited from the session from the KV cache
                llama_kv_cache_tokens_rm(ctx, n_past, -1); // FIXME: Not needed?
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always

            // if (ctx_guidance) { ... } // TODO: Investigate

            for (int i = 0; i < (int) embd.size(); i += n_batch) {

                int n_eval = (int) embd.size() - i;
                if (n_eval > n_batch) {
                    n_eval = n_batch;
                }

                // WAS: if (llama_eval(ctx, &embd[i], n_eval, n_past, ::params[idx].n_threads)) {
                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
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
        //embd_guidance.clear();

        //fprintf(stderr, "%s === embd_inp.size() = %d | n_consumed = %d | n_remain = %d \n", __func__, (int) embd_inp.size(), (int) n_consumed, (int) n_remain); // DEBUG

        if ((int) embd_inp.size() <= n_consumed) {

            // --- out of user input, sample next token

            std::vector<llama_token_data> candidates;
            candidates.reserve(vocabSize);

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
                n_past /* - n_consumed*/,
                ::params[idx].n_predict);

            // FIXME: New format
            // TODO: Learn sampling.h / sampling.cpp
            // const llama_token id = llama_sampling_sample(ctx, ctx_guidance, ctx_sampling, last_tokens, candidates);

            last_tokens.erase(last_tokens.begin());
            last_tokens.push_back(id);

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;

        } else { 

            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_tokens.erase(last_tokens.begin());
                last_tokens.push_back(embd_inp[n_consumed]);
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
        if (!embd.empty() && embd.back() == llama_token_eos(model)) {
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

void init(char * sessionPath) {
    ::path_session = sessionPath;
    hide();
    llama_backend_init(false); // NUMA = false
    show();
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
    float temp, int top_k, float top_p,
    float typical_p, 
    float penalty_repeat, int penalty_last_n,
    int32_t janus, int32_t depth, float scale, float hi, float lo,
    uint32_t seed) {
    
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

    ::sparams[idx].temp           = temp;
    ::sparams[idx].top_k          = top_k;
    ::sparams[idx].top_p          = top_p;

    ::sparams[idx].typical_p      = typical_p > 0 ? typical_p : 1.0f;

    ::sparams[idx].penalty_repeat = penalty_repeat;
    ::sparams[idx].penalty_last_n = penalty_last_n;
    
    ::params[idx].seed            = seed;
    
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

uint32_t getSeed(char * jobID) {
    std::string id = jobID;
    return getSeedCPP(id);
}

}  // ------------------------------------------------------

//
// Vocab utils
//

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

std::string llama_token_to_str(const struct llama_context * ctx, llama_token token) {
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
