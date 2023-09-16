// Various helper functions and utilities

//#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <string>
#include <vector>
#include <random>
#include <thread>
#include <shared_mutex>
#include <unordered_map>
#include <tuple>

#if !defined (_WIN32)
#include <stdio.h>
#include <termios.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

// --- Some (possibly be wrong) observations

// Temp is used both for mirostat and TopK-TopP sampling, but it better to keep lower temp for mirostat

// Mirostat is more chatty and fluid, looks like real human being. At the same time is jumps wild between topics
// It favors bigger models and becomes crazy on smaller ones like 7B
// Mirostat v2 looks better than v1
// Results with tau = 5 norm, better with lower values down to 1.0 and wild bad up to 9.0
// Not sure how eta plays here

// TopK over 40 (up to 100) might produce crazy irrelevant results. But usually it safe to lower about 10-20
// Seems like TopP sweet spot is arount 0.95
// Not sure about Temp, but any value around 0.5 .. 0.8 is good enough (lower == more steady output, higher = more creative)

// RepeatPenalty is good around 1.1 and bad around 1.0 (switched off). 
// Not sure why bad values above 1.1 are shuffling and muffling the output completely (with mirostat at least)

struct gpt_params {
    uint32_t seed                           = -1;   // RNG seed
    int32_t n_threads                       = 1;    // get_num_physical_cores();
    int32_t n_predict                       = -1;   // new tokens to predict
    int32_t n_ctx                           = 4096; // 512;  // context size
    int32_t n_batch                         = 512;  // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                          = 0;    // number of tokens to keep from initial prompt
    int32_t n_draft                         = 16;   // number of tokens to draft during speculative decoding
    int32_t n_chunks                        = -1;   // max number of chunks to process (-1 = unlimited)
    int32_t n_gpu_layers                    = -1;   // number of layers to store in VRAM (-1 - use default)
    int32_t n_gpu_layers_draft              = -1;   // number of layers to store in VRAM for the draft model (-1 - use default)
    int32_t main_gpu                        = 0;    // the GPU that is used for scratch and small tensors
    float   tensor_split[ 8 /*LLAMA_MAX_DEVICES*/ ] = {0};  // how split tensors should be distributed across GPUs
    int32_t n_probs                         = 0;    // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t n_beams                         = 0;    // if non-zero then use beam search of given width.
    float   rope_freq_base                  = 10000.0f; // RoPE base frequency
    float   rope_freq_scale                 = 1.0f;     // RoPE frequency scaling factor

    // sampling parameters
    int32_t top_k             = 8;     // 40;    // <= 0 to use vocab size
    float   top_p             = 0.1;   // 95f; // 1.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typical_p         = 1.00f; // 1.0 = disabled
    float   temp              = 0.1;   // 0.80f; // 1.0 = disabled
    float   repeat_penalty    = 1.10f; // 1.0 = disabled
    int32_t repeat_last_n     = -1;    // 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   frequency_penalty = 0.00f; // 0.0 = disabled
    float   presence_penalty  = 0.00f; // 0.0 = disabled
    int32_t mirostat          = 2;     // 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 0.1;   // 5.00f; // target entropy
    float   mirostat_eta      = 0.1;   // 0.10f; // learning rate

    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    std::string cfg_negative_prompt;       // string to help guidance
    float       cfg_scale         = 1.f;   // How strong is guidance

    std::string model             = "models/7B/ggml-model-f16.gguf"; // model path
    std::string model_draft       = "";                              // draft model for speculative decoding
    std::string model_alias       = "unknown"; // model alias
    std::string prompt            = "";
    std::string path_prompt_cache = "";  // path to file for saving/loading prompt eval state
    std::string input_prefix      = "";  // string to prefix user inputs with
    std::string input_suffix      = "";  // string to suffix user inputs with
    std::string grammar           = "";  // optional BNF-like grammar to constrain sampling
    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted
    std::string logdir            = "";  // directory in which to save YAML log files

    std::string lora_adapter = "";  // lora adapter path
    std::string lora_base    = "";  // base model path for the lora adapter

    int  ppl_stride        = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int  ppl_output_type   = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                    //                                       (which is more convenient to use for plotting)
                                    //
    bool hellaswag         = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks = 400;   // number of tasks to use when computing the HellaSwag score

    bool low_vram          = false; // if true, reduce VRAM usage at the cost of performance
    bool mul_mat_q         = true;  // if true, use mul_mat_q kernels instead of cuBLAS
    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool embedding         = false; // get only sentence embedding
    bool escape            = false; // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool interactive_first = false; // wait for user input immediately
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool ignore_eos        = false; // ignore generated EOS tokens
    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool penalize_nl       = true;  // consider newlines as a repeatable token
    bool perplexity        = false; // compute perplexity over the prompt
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool mem_test          = false; // compute maximum memory usage
    bool numa              = false; // attempt optimizations that help on some NUMA systems
    bool export_cgraph     = false; // export the computation graph
    bool verbose_prompt    = false; // print prompt tokens before generation
};

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

static std::string llama_token_to_str(const struct llama_context * ctx, llama_token token) {
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

bool low_vram; // global setting

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

#ifdef _WIN32
#define NULL_DEVICE "NUL:"
#define TTY_DEVICE "COM1:"
#else
#define NULL_DEVICE "/dev/null"
#define TTY_DEVICE "/dev/tty"
#endif

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
    //lparams.f16_kv     = params.memory_f16;
    //lparams.use_mmap   = params.use_mmap;
    //lparams.use_mlock  = params.use_mlock;
    //lparams.logits_all = params.perplexity;
    //lparams.embedding  = params.embedding;
    lparams.low_vram     = ::low_vram;

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

    ///// llama_context * lctx = llama_init_from_file(params[idx].model.c_str(), lparams);

    // FIXME ^^^
    // bridge.cpp:161:28: warning: 'llama_init_from_file' is deprecated: please use llama_load_model_from_file 
    // combined with llama_new_context_with_model instead [-Wdeprecated-declarations]

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
int64_t do_inference(int idx, struct llama_context * ctx, const std::string & jobID, const std::string & sessionID, const std::string & text) {

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

    //const int32_t n_sample = std::max(1, llama_n_sample(ctx));
    const int32_t n_eval   = std::max(1, llama_n_eval(ctx));
    const int32_t n_p_eval = std::max(1, llama_n_p_eval(ctx));

    //fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, llama_t_load_us(ctx) / 1000.0);
    //fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * llama_t_sample_us(ctx), n_sample, 1e-3 * llama_t_sample_us(ctx) / n_sample);
    //fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__, 1e-3 * llama_t_p_eval_us(ctx), n_p_eval, 1e-3 * llama_t_p_eval_us(ctx) / n_p_eval);
    //fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * llama_t_eval_us(ctx),   n_eval,   1e-3 * llama_t_eval_us(ctx) / n_eval);
    //fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - llama_t_start_us(ctx))/1000.0);

    // TODO: Sum all timings
    // compute average time needed for processing one token
    //const int32_t avg_compute_time = //1e-3 * llama_t_sample_us(ctx) / n_sample + 
                                     //1e-3 * llama_t_p_eval_us(ctx) / n_p_eval + 
    //                                 1e-3 * llama_t_eval_us(ctx) / n_eval;

    mutex.lock();
    promptEvals[jobID] = 1e-3 * llama_t_p_eval_us(ctx) / n_p_eval;
    timings[jobID] = 1e-3 * llama_t_eval_us(ctx) / n_eval; // avg_compute_time;
    mutex.unlock();

    return n_p_eval + n_eval;
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
    int64_t res = timings[jobID];
    mutex.unlock_shared();
    return res;
}

extern "C" { // ------------------------------------------------------

void init(char * sessionPath, bool numa, bool lowVRAM) {
    ::path_session = sessionPath;
    //if (numa) {
    //    ggml_numa_init();
    //}
    ::low_vram = lowVRAM;
    //llama_init_backend(numa); 
    llama_backend_init(numa);
    // TODO: before server exit: llama_backend_free();
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

int64_t doInference(int idx, void * ctx, char * jobID, char * sessionID, char * prompt) {
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

