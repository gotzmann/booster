// Various helper functions and utilities

#include "llama.h"
//#include "examples/common.h"

#include <string>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>

//
// CLI argument parsing
//
////int32_t get_num_physical_cores();

struct gpt_params {
    int32_t seed          = -1;   // RNG seed
    int32_t n_threads     = 6; // FIXME //// get_num_physical_cores();
    int32_t n_predict     = -1;  // new tokens to predict
    int32_t n_parts       = -1;   // amount of model parts (-1 = determine from model dimensions)
    int32_t n_ctx         = 512;  // context size
    int32_t n_batch       = 512;  // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep        = 0;    // number of tokens to keep from initial prompt

    // sampling parameters
    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typical_p         = 1.00f; // 1.0 = disabled
    float   temp              = 0.80f; // 1.0 = disabled
    float   repeat_penalty    = 1.10f; // 1.0 = disabled
    int32_t repeat_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   frequency_penalty = 0.00f; // 0.0 = disabled
    float   presence_penalty  = 0.00f; // 0.0 = disabled
    int     mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate

    std::string model  = "models/lamma-7B/ggml-model.bin"; // model path
    std::string prompt = "";
    std::string path_session = "";       // path to file for saving/loading model eval state
    std::string input_prefix = "";       // string to prefix user inputs with
    std::string input_suffix = "";       // string to suffix user inputs with
    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted

    std::string lora_adapter = "";  // lora adapter path
    std::string lora_base = "";     // base model path for the lora adapter

    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode

    bool embedding         = false; // get only sentence embedding
    bool interactive_first = false; // wait for user input immediately

    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool penalize_nl       = true;  // consider newlines as a repeatable token
    bool perplexity        = false; // compute perplexity over the prompt
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool mem_test          = false; // compute maximum memory usage
    bool verbose_prompt    = false; // print prompt tokens before generation
};

////bool gpt_params_parse(int argc, char ** argv, gpt_params & params);

////void gpt_print_usage(int argc, char ** argv, const gpt_params & params);

////std::string gpt_random_prompt(std::mt19937 & rng);

//
// Vocab utils
//

////std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos);

//
// Model utils
//

//struct llama_context * llama_init_from_gpt_params(const gpt_params & params);

struct llama_context * llama_init_from_gpt_params(const gpt_params & params) {
    auto lparams = llama_context_default_params();

    lparams.n_ctx      = params.n_ctx;
    lparams.n_parts    = params.n_parts;
    lparams.seed       = params.seed;
    lparams.f16_kv     = params.memory_f16;
    lparams.use_mmap   = params.use_mmap;
    lparams.use_mlock  = params.use_mlock;
    lparams.logits_all = params.perplexity;
    lparams.embedding  = params.embedding;

    llama_context * lctx = llama_init_from_file(params.model.c_str(), lparams);

    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return NULL;
    }

    if (!params.lora_adapter.empty()) {
        int err = llama_apply_lora_from_file(lctx,
                                             params.lora_adapter.c_str(),
                                             params.lora_base.empty() ? NULL : params.lora_base.c_str(),
                                             params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            return NULL;
        }
    }

    return lctx;
}

// TODO: not great allocating this every time
std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}

void loopCPP(struct llama_context * ctx, /*std::vector<llama_token> & embd_inp*/ const std::string & text) {

    bool add_bos = true;

    //std::vector<llama_token> embd_inp = llama_tokenize((struct llama_context *)ctx, text, true);

    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> embd_inp(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), embd_inp.data(), embd_inp.size(), add_bos);
    assert(n >= 0);
    embd_inp.resize(n);

    //fprintf(stderr, "\n=== loop 00 | N tokens = %d", n);

    fprintf(stderr, "\n=== TOKENS ===\n");
    for (auto id : embd_inp) {
        printf(" [ %d ] ", id); // DEBUG
    }

    //fprintf(stderr, "\n=== loop 01 ===");

    const int n_ctx = llama_n_ctx(ctx);

    //fprintf(stderr, "\n=== loop 02 ===");

    // sampling parameters
    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    //fprintf(stderr, "\n=== loop 03 ===");

    //const int n_ctx = llama_n_ctx(ctx);

    int n_past             = 0;
    int n_remain           = 64; // FIXME params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<llama_token> embd;

    //fprintf(stderr, "\n=== loop 04 ===");

    while (n_remain != 0 /*|| params.interactive*/) {

        //fprintf(stderr, "\n=== loop 05 ===");

        // predict
        if (embd.size() > 0) {


            //fprintf(stderr, "\n=== loop 06 ===");

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {

                //fprintf(stderr, "\n=== loop 07 ===");

                const int n_left = n_past - /*params.n_keep*/ 16;

                n_past = 16; // params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                // stop saving session if we run out of context
                ////path_session = "";

                //printf("\n---\n");
                //printf("resetting: '");
                //for (int i = 0; i < (int) embd.size(); i++) {
                //    printf("%s", llama_token_to_str(ctx, embd[i]));
                //}
                //printf("'\n");
                //printf("\n---\n");
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            // REVIEW 
            /* if (n_session_consumed < (int) session_tokens.size()) {
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
            } */

            //fprintf(stderr, "\n=== loop 08 ===");

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += /*params.n_batch*/ 16) {

                //fprintf(stderr, "\n=== loop 09 ===");
                int n_eval = (int) embd.size() - i;
                if (n_eval > /*params.n_batch*/ 16) {
                    n_eval = /*params.n_batch*/ 16;
                }
                if (llama_eval(ctx, &embd[i], n_eval, n_past, /*params.n_threads*/ 6)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    ////return 1;
                    

                    //fprintf(stderr, "\n=== loop 10 ===");
                    return;
                }
                n_past += n_eval;
            }

            ////if (embd.size() > 0 && !path_session.empty()) {
            ////    session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
            ////    n_session_consumed = session_tokens.size();
            ////}
        }

        //fprintf(stderr, "\n=== loop 11 ===");

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed /*&& !is_interacting*/) {

            //fprintf(stderr, "\n=== loop 12 ===");

            // out of user input, sample next token
            const float   temp            = 0.8; // params.temp;
            const int32_t top_k           = 40; //params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float   top_p           = 40; //params.top_p;
            const float   tfs_z           = 0; //params.tfs_z;
            const float   typical_p       = 0; //params.typical_p;
            const int32_t repeat_last_n   = 40; //params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = 1; //params.repeat_penalty;
            const float   alpha_presence  = 1;//params.presence_penalty;
            const float   alpha_frequency = 1;//params.frequency_penalty;
            const int     mirostat        = 0;//params.mirostat;
            const float   mirostat_tau    = 0;//params.mirostat_tau;
            const float   mirostat_eta    = 0;//params.mirostat_eta;
            const bool    penalize_nl     = 0;//params.penalize_nl;

            // optionally save the session on first sample (for faster prompt loading next time)
            ////if (!path_session.empty() && need_to_save_session) {
            ////    need_to_save_session = false;
            ////    llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            ////}

            llama_token id = 0;

            {

                //fprintf(stderr, "\n=== loop 13 ===");

                auto logits  = llama_get_logits(ctx);
                auto n_vocab = llama_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = /*params.*/logit_bias.begin(); it != /*params.*/logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[llama_token_nl()];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    logits[llama_token_nl()] = nl_logit;
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                } else {
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
                        llama_sample_top_k(ctx, &candidates_p, top_k);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z);
                        llama_sample_typical(ctx, &candidates_p, typical_p);
                        llama_sample_top_p(ctx, &candidates_p, top_p);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }
                // printf("`%d`", candidates_p.size);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            ////if (id == llama_token_eos() && params.interactive && !params.instruct) {
            ////    id = llama_token_newline.front();
            ////    if (params.antiprompt.size() != 0) {
            ////        // tokenize and inject first reverse prompt
            ////        const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
            ////        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
            ////    }
            ////}

            // add it to the context
            embd.push_back(id);

            //fprintf(stderr, "\n=== loop 20 ===");

            // echo this to console
            ////input_echo = true;

            // decrement remaining sampling budget
            --n_remain;
        } else {

            //fprintf(stderr, "\n=== loop 30 ===");

            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= /*params.n_batch*/ 16) {
                    break;
                }
            }
        }

        // display text
        ////if (input_echo) {
            for (auto id : embd) {
                //printf(" [ %d ] ", id); // DEBUG
                printf("%s", llama_token_to_str(ctx, id));
            }
            fflush(stdout);
        ////}
        // reset color to default if we there is no pending user input
        ////if (input_echo && (int)embd_inp.size() == n_consumed) {
        ////    set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
        ////}

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        /* if (params.interactive && (int) embd_inp.size() <= n_consumed) {

            // check for reverse prompt
            if (params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_str(ctx, id);
                }

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                for (std::string & antiprompt : params.antiprompt) {
                    if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                        is_interacting = true;
                        is_antiprompt = true;
                        set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);
                        fflush(stdout);
                        break;
                    }
                }
            }

            if (n_past > 0 && is_interacting) {
                // potentially set color to indicate we are taking user input
                set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);

                if (params.instruct) {
                    printf("\n> ");
                }

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    buffer += params.input_prefix;
                    printf("%s", buffer.c_str());
                }

                std::string line;
                bool another_line = true;
                do {
#if defined(_WIN32)
                    std::wstring wline;
                    if (!std::getline(std::wcin, wline)) {
                        // input stream is bad or EOF received
                        return 0;
                    }
                    win32_utf8_encode(wline, line);
#else
                    if (!std::getline(std::cin, line)) {
                        // input stream is bad or EOF received
                        return 0;
                    }
#endif
                    if (!line.empty()) {
                        if (line.back() == '\\') {
                            line.pop_back(); // Remove the continue character
                        } else {
                            another_line = false;
                        }
                        buffer += line + '\n'; // Append the line to the result
                    }
                } while (another_line);

                // done taking input, reset color
                set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty()) {
                        buffer += params.input_suffix;
                        printf("%s", params.input_suffix.c_str());
                    }

                    // instruct mode: insert instruction prefix
                    if (params.instruct && !is_antiprompt) {
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }

                    auto line_inp = ::llama_tokenize(ctx, buffer, false);
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    // instruct mode: insert response suffix
                    if (params.instruct) {
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }

                    n_remain -= line_inp.size();
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                is_interacting = false;
            }
        } */

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            ////if (params.instruct) {
            ////    is_interacting = true;
            ////} else {
                fprintf(stderr, " [end of text]\n");
                break;
            ////}
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        ////if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
        ////    n_remain = params.n_predict;
        ////    is_interacting = true;
        ////}
    }
}

extern "C" { // ------------------------------------------------------

void * initFromParams(char * modelName) {
    fprintf(stderr, "\n=== initFromParams ===");
    
    gpt_params params;
    fprintf(stderr, "\ndefaultModel = %s", params.model.c_str());
    params.model = modelName;
    
    return llama_init_from_gpt_params(params);
    //return NULL;
}

void * tokenize(void * ctx, char * prompt) {
    fprintf(stderr, "\n=== tokenize ===");
    std::string text = prompt;
    std::vector<llama_token> tokens = llama_tokenize((struct llama_context *)ctx, text, true);
    return &tokens;
}

void loop(void * ctx, /*void * embd_inp*/ char * prompt) {
    fprintf(stderr, "\n=== loop ===");
    std::string text = prompt;
    loopCPP((struct llama_context *)ctx, text);
}

}  // ------------------------------------------------------

