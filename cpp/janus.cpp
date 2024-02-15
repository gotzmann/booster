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
#include "ggml-backend.h"
#include "llama.h"

#include "bridge.h"
#include "janus.h"

// The Guardian has always been a newspaper for writers, 
// and so a newspaper for readers.

// The introduction to the Guardian stylebook of 1960, 
// which itself was a revision to the initial guide published in 1928, 
// was headed "Neither pedantic nor wild".

// https://www.theguardian.com/info/2000/mar/24/neither-pedantic-nor-wild

bool isJanusInitialized = false;

float * scales; // precomputed scales (penalties) for each token
float * types;  // precoputed types for each token

// -- NB! llama_sampling_sample() is a newer implementation of older bridge.cpp::llama_sample_token() with Janus implementation inside
/*
llama_token llama_sampling_sample(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx) {

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
//    std::vector<float> original_logits;

//    if (!is_resampling) {
        // Only make a copy of the original logits if we are not in the resampling phase, not sure if I actually have to do this.
//        original_logits = std::vector<float>(logits, logits + llama_n_vocab(llama_get_model(ctx_main)));
//    }

    // apply params.logit_bias map
//    for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
//        logits[it->first] += it->second;
//    }

    if (ctx_cfg) {
        ///// float * logits_guidance = llama_get_logits_ith(ctx_cfg, idx);
        ///// llama_sample_apply_guidance(ctx_main, logits, logits_guidance, params.cfg_scale);
    }

    cur.clear();

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), false };

    // -- Experimental sampling both creative for text and pedantic for math / coding
//    if (params.janus > 0) {
        // WAS: return sample_janus_token(ctx, params, last_tokens, promptLen, pos, max);
//        return sample_janus_token(ctx_main, ctx_sampling.params, last_tokens, promptLen, pos, max);
//    }

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
//    if (is_resampling && ctx_sampling->grammar != NULL) {
//        llama_sample_grammar(ctx_main, &cur_p, ctx_sampling->grammar);
//    }

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

            // LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx_main, id).c_str());
        }
    }

//    if (ctx_sampling->grammar != NULL && !is_resampling) {
//        // Create an array with a single token data element for the sampled id
//        llama_token_data single_token_data = {id, logits[id], 0.0f};
//        llama_token_data_array single_token_data_array = { &single_token_data, 1, false };

        // Apply grammar constraints to the single token
//        llama_sample_grammar(ctx_main, &single_token_data_array, ctx_sampling->grammar);

        // Check if the token is valid according to the grammar by seeing if its logit has been set to -INFINITY
//        bool is_valid = single_token_data_array.data[0].logit != -INFINITY;

        // If the token is not valid according to the grammar, perform resampling
//        if (!is_valid) {
//            LOG("Resampling because token %d: '%s' does not meet grammar rules\n", id, llama_token_to_piece(ctx_main, id).c_str());

            // Restore logits from the copy
//            std::copy(original_logits.begin(), original_logits.end(), logits);

//            return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, true);  // Pass true for is_resampling
//        }
//    }

    return id;
}
*/
// -- Experimental approach of Janus Sampling by gotzmann [ paper is coming ]

llama_token sample_janus_token(

        struct llama_context * ctx, 
        struct llama_sampling_params & params,
        const std::vector<llama_token> & last_tokens,
        const size_t promptLen,
        const size_t pos,
        const size_t max) {

    if (!::isJanusInitialized) {
        initJanus(ctx, params);
        ::isJanusInitialized = true;
    }

    // const int64_t t_start_sample_us = ggml_time_us();

    /* DEBUG
    fprintf(stderr, "\n * janus = %d", params.janus);
    fprintf(stderr, "\n * depth = %d", params.depth);
    fprintf(stderr, "\n * scale = %f", params.scale);
    fprintf(stderr, "\n * hi = %f", params.hi);
    fprintf(stderr, "\n * lo = %f", params.lo);
    fprintf(stderr, "\n * pos = %d", pos);
    fprintf(stderr, "\n * promptLen = %d", promptLen);
    //exit(1); */

    printDebug(ctx, pos, 0, "TOP LIST"); // -- DEBUG

    auto model       = llama_get_model(ctx);
    float * logits   = llama_get_logits(ctx);
    size_t vocabSize = llama_n_vocab(model);
    size_t ctxSize   = last_tokens.size();
    // auto scale       = params.scale;

    auto lastToken = last_tokens.data()[ last_tokens.size() - 1 ];
    auto lastType  = ::types[lastToken];
   
    // -- Boost <EOS> token when we are closer to the limit
    //    NB! It looks like it enough just do not penalize it at all [ allowing scale == 1.0 ] ?

    logits[EOS] *= 1.0 + log(1.0 + float(pos - promptLen) / float(max)) * 0.05;

    // -- Smart pessimization for repeated tokens
    //    For better performance we are excluding prompt tokens

    // TODO: This should work right for the first system prompt, but what's about the next ones [ second, third, etc ] ?!
    size_t depth = std::min((size_t) params.depth, pos - promptLen);
    // fprintf(stderr, "\n * depth = %d", depth); // DEBUG
    // fprintf(stderr, "\n * ctxSize = %d", ctxSize); // DEBUG
    for (size_t i = 0; i < depth; i++) {
        //fprintf(stderr, " [ i=%d | pos=%d | depth=%d | len=%d ] ", i, pos, depth, promptLen); // DEBUG
        // WAS auto id = last_tokens.data()[ ctxSize - 1 - i ];
        auto id = last_tokens[ ctxSize - 1 - i ];
        auto curType = ::types[id];
        // fprintf(stderr, "\n [ ID == %d ] ", id); // DEBUG

        // Decrease reperition penalty for word continuation tokens to help prevent wrong wordings in complex languages
        // TODO: Maybe we need to skip the last token itself [ with check of i > 0 ] ?! 
        if ((lastType == SPACE_RU || lastType == LANG_RU) && curType == LANG_RU) {
            // fprintf(stderr, "\n WAS 01 = %f", logits[id]); // DEBUG
            logits[id] *= 1.0 - (1.0 - ::scales[id]) * 0.20;
            // fprintf(stderr, "\n NOW 01 = %f", logits[id]); // DEBUG
            continue;
        }

        // TODO: Should we process negative probabilities by scale division?
        // how it was before: logits[id] /= 1.0 + (penalty - 1.0) * 0.10;
        // fprintf(stderr, "\n SCALE 02 %d = %f", id, ::scales[id]); // DEBUG
        // fprintf(stderr, "\n WAS 02 = %f", logits[id]); // DEBUG
        logits[id] *= ::scales[id];
        // fprintf(stderr, "\n NOW 02 = %f", logits[id]); // DEBUG
    }
   
    // -- Double down incompatible tokens (like word endings in some other language)

    for (size_t id = 0; id < vocabSize; id++) {

        auto curType = ::types[id];

        if (
            ((lastType == SPACE_RU || lastType == LANG_RU) && (curType == LANG_EN || curType == LANG_OTHER))
            // ||
            // ((lastType == LANG_EN || lastType == SPACE_EN) && curType == LANG_RU) // Europeans mix ASCII and UTF-8
        ) {
            // fprintf(stderr, "\n WAS 03 = %f", logits[id]); // DEBUG
            logits[id] *= 0.5; // scale * scale * scale;
            // fprintf(stderr, "\n NOW 03 = %f", logits[id]); // DEBUG
        }
    }        
   
    // -- Sort all logits

    std::vector<llama_token_data> candidates;
    //candidates.reserve(vocabSize);
    candidates.clear();

    for (llama_token id = 0; id < (int) vocabSize; id++) {
        //candidates.data()[id] = llama_token_data{ id, logits[id], 0.0f };
        candidates.emplace_back(
            llama_token_data{id, logits[id], 0.0f}
        );
    }

    std::sort(
        candidates.data(), 
        candidates.data() + candidates.size(), 
        [](const llama_token_data & a, const llama_token_data & b) { 
            return a.logit > b.logit; 
        }
    );           
    
    // -- Final choice [ with experimental cutoff ]
    //    We'll use some general cutoff value for most of tokens
    //    and pedantic cutoff for the sensitive ones

    auto topToken = candidates.data()[0].id;
    auto topType  = types[topToken];
    auto topLogit = candidates.data()[0].logit;

    float cutoff = params.lo;
    if (isPedantic(topToken) || topType == LANG_RU || topType == LANG_EN) {
        cutoff = params.hi;
    }

    for (size_t i = 1; i < candidates.size(); i++) {
        if (candidates.data()[i].logit / topLogit < cutoff) {
            candidates.resize(i);
            break;
        }
    }

    printDebug(ctx, pos, candidates.size(), "SHORTIST"); // -- DEBUG

    llama_token_data_array shortlist = { candidates.data(), candidates.size(), true };

    return llama_sample_token(ctx, &shortlist);
}

// -- initJanus prefills base scaling penalties for each token depending on Janus Sampling euristics

void initJanus(struct llama_context * ctx, struct llama_sampling_params & params) {

    auto model = llama_get_model(ctx);
    auto vocabSize = llama_n_vocab(model);
    ::scales = new float[vocabSize] {};
    ::types = new float[vocabSize] {};

    /* DEBUG
    fprintf(stderr, "\n * janus = %d", params.janus);
    fprintf(stderr, "\n * depth = %d", params.depth);
    fprintf(stderr, "\n * scale = %f", params.scale);
    fprintf(stderr, "\n * hi = %f", params.hi);
    fprintf(stderr, "\n * lo = %f", params.lo);
    exit(1); */

    // -- safe defaults

    //if (params.depth <= 0 || params.depth > params.n_predict) {
    //    params.depth = 200;
    //}

    if (params.scale <= 0.0 || params.scale > 1.0) params.scale = 0.97;
    if (params.hi <= 0.0    || params.hi > 1.0)    params.hi = 0.99;
    if (params.lo <= 0.0    || params.lo > 1.0)    params.lo = 0.96;

    // -- init tokens with some heuristics
    //    how it was before [ with penalty = 1.06 ] : logits[id] /= 1.0 + (penalty - 1.0) * 0.10;

    float scale = params.scale;

    for (llama_token id = 0; id < vocabSize; id++) {

        auto type  = tokType(ctx, id);
        auto lower = isLower(ctx, id);
        size_t len = tokSize(ctx, id);

        ::types[id] = type;

        // -- pedantic tokens

        if (isPedantic(id)) {
            ::scales[id] = 1.0 - (1.0 - scale) * 0.20;
            //fprintf(stderr, "\n SCALE1 %d = %f", id, ::scales[id]);
            //fprintf(stderr, " | \"%s\"", llama_token_to_piece(ctx, id).c_str());
            continue;
        }   

        // -- special case for complex languages like Russian.
        //    penalise the tokens that might be a parts of other words with less scale
        //    NB! Fix the case of tooooo long tokens => SCALE 23831 = 5115.378906 | "printStackTrace"

        float probes[] = {
            0.20, 0.22, 0.25, 0.28, 0.30,
            0.32, 0.33, 0.35, 0.36, 0.38,
            0.40, 0.42, 0.44, 0.45, 0.46,
            0.48, 0.50, 0.52, 0.53, 0.55,
        };

        if (type == LANG_RU && lower) {
            // NB! Size in bytes is 2x of UTF-8 chars for RU
            ::scales[id] = 1.0 - (1.0 - scale) * probes[len/2];
            //fprintf(stderr, "\n SCALE2 %d = %f", id, ::scales[id]);
            //fprintf(stderr, " | \"%s\"", llama_token_to_piece(ctx, id).c_str());
            continue;
        }

        // -- similar hack for EN

        if (type == LANG_EN && lower) {
            ::scales[id] = 1.0 - (1.0 - scale) * probes[len];
            //fprintf(stderr, "\n SCALE3 %d = %f", id, ::scales[id]);
            //fprintf(stderr, " | \"%s\"", llama_token_to_piece(ctx, id).c_str());
            continue;
        }

        // -- full penalization for other tokens

        ::scales[id] = scale;
        //fprintf(stderr, "\n SCALE4 %d = %f", id, ::scales[id]);
        //fprintf(stderr, " | \"%s\"", llama_token_to_piece(ctx, id).c_str());
    }

    // -- Assign manually specific penalties for high-frequency tokens
    // TODO: Need more work with real texts and statistical probabilities

    ::scales[0]     = 1.0;   // just to be safe
    ::scales[EOS]   = scale; // penalize <EOS> in the beginning and allow it to boost over 1.0 later
    
    ::scales[NL]    = 1.0 - (1.0 - scale) * 0.10; // newline

    ::scales[259]   = 1.0 - (1.0 - scale) * 0.20; //   259 => "  "
    ::scales[268]   = 1.0 - (1.0 - scale) * 0.20; //   268 => "    "

    ::scales[29871] = 1.0 - (1.0 - scale) * 0.10; // 29871 => " "
    ::scales[29892] = 1.0 - (1.0 - scale) * 0.10; // 29892 => ","
    ::scales[29889] = 1.0 - (1.0 - scale) * 0.20; // 29889 => "."

    ::scales[813]   = 1.0 - (1.0 - scale) * 0.30; // 813   => " —"
    ::scales[29899] = 1.0 - (1.0 - scale) * 0.30; // 29899 => "-" [ used as bullet point ]
    ::scales[29901] = 1.0 - (1.0 - scale) * 0.30; // 29901 => ":"
    ::scales[29936] = 1.0 - (1.0 - scale) * 0.30; // 29936 => ";"

    ::scales[313]   = 1.0 - (1.0 - scale) * 0.30; // 313   => " ("
    ::scales[467]   = 1.0 - (1.0 - scale) * 0.30; // 467   => ")."
    ::scales[1723]  = 1.0 - (1.0 - scale) * 0.30; // 1723  => " )"
    ::scales[29897] = 1.0 - (1.0 - scale) * 0.30; // 29897 => ")"
    ::scales[29898] = 1.0 - (1.0 - scale) * 0.30; // 29898 => "("
    
    // -- Popular RU parts

    ::scales[490]   = 1.0 - (1.0 - scale) * 0.30; // 490  => " в"
    ::scales[531]   = 1.0 - (1.0 - scale) * 0.30; // 531  => " с"
    ::scales[606]   = 1.0 - (1.0 - scale) * 0.30; // 606  => " и"
    ::scales[614]   = 1.0 - (1.0 - scale) * 0.30; // 614  => " о"
    ::scales[665]   = 1.0 - (1.0 - scale) * 0.35; // 665  => " на"
    ::scales[733]   = 1.0 - (1.0 - scale) * 0.35; // 733  => " по"
    ::scales[863]   = 1.0 - (1.0 - scale) * 0.35; // 863  => " у"
    ::scales[1077]  = 1.0 - (1.0 - scale) * 0.40; // 1077 => " за"
    ::scales[1097]  = 1.0 - (1.0 - scale) * 0.40; // 1097 => " а"
    ::scales[1186]  = 1.0 - (1.0 - scale) * 0.40; // 1186 => " к"
    ::scales[1447]  = 1.0 - (1.0 - scale) * 0.45; // 1447 => " до"
    ::scales[1538]  = 1.0 - (1.0 - scale) * 0.45; // 1538 => " не"
    ::scales[1604]  = 1.0 - (1.0 - scale) * 0.45; // 1604 => " об"
    ::scales[1685]  = 1.0 - (1.0 - scale) * 0.45; // 1685 => " от"
    ::scales[4281]  = 1.0 - (1.0 - scale) * 0.50; // 4281 => " что"

    ::scales[857]   = 1.0 - (1.0 - scale) * 0.50; // 857  => " С"
    ::scales[939]   = 1.0 - (1.0 - scale) * 0.50; // 939  => " В"
    ::scales[1651]  = 1.0 - (1.0 - scale) * 0.50; // 1651 => " О"

    // -- Popular EN parts

    ::scales[263]   = 1.0 - (1.0 - scale) * 0.30; // 263 => " a"
    ::scales[278]   = 1.0 - (1.0 - scale) * 0.30; // 278 => " the"
    ::scales[297]   = 1.0 - (1.0 - scale) * 0.30; // 297 => " in"
    ::scales[304]   = 1.0 - (1.0 - scale) * 0.30; // 304 => " to"
    ::scales[310]   = 1.0 - (1.0 - scale) * 0.30; // 310 => " of"
    ::scales[322]   = 1.0 - (1.0 - scale) * 0.30; // 322 => " and"

    ::scales[363]   = 1.0 - (1.0 - scale) * 0.35; // 363 => " for"
    ::scales[372]   = 1.0 - (1.0 - scale) * 0.35; // 372 => " it"
    ::scales[373]   = 1.0 - (1.0 - scale) * 0.35; // 373 => " on"
    ::scales[385]   = 1.0 - (1.0 - scale) * 0.35; // 385 => " an"
    ::scales[393]   = 1.0 - (1.0 - scale) * 0.35; // 393 => " that"
    ::scales[408]   = 1.0 - (1.0 - scale) * 0.35; // 408 => " as"
    ::scales[411]   = 1.0 - (1.0 - scale) * 0.35; // 411 => " with"
    
    ::scales[470]   = 1.0 - (1.0 - scale) * 0.40; // 470 => " or"
    ::scales[472]   = 1.0 - (1.0 - scale) * 0.40; // 472 => " at"
    ::scales[526]   = 1.0 - (1.0 - scale) * 0.40; // 526 => " are"

    ::scales[319]   = 1.0 - (1.0 - scale) * 0.50; // 319 => " A"

    //exit(1); // DEBUG
}

// Tokens very often used for math, coding and JSON (aka repetitive),
// so we should be care about them and not penalize

llama_token pedanticTokens[] = {

    2,     // <EOS>

    28956, // "```"

    // -- Math

    29900, // "0"
    29896, // "1"
    29906, // "2"
    29941, // "3"
    29946, // "4"
    29945, // "5"
    29953, // "6"
    29955, // "7"
    29947, // "8"
    29929, // "9"

    334,   // " *"
    353,   // " ="
    448,   // " -"
    718,   // " +"

    // -- JSON

    29912, // "{"
    29913, // "}"
    29961, // "["
    29962, // "]"

    426,   // " {"
    500,   // " }"
    518,   // " ["
    4514,  // " ]"

    // 376,   //  " ""
    // 613,   // "","
};

bool isPedantic(llama_token id) {
    size_t len = *(&pedanticTokens + 1) - pedanticTokens;
    for (size_t i = 0; i < len; i++) {
        if (id == pedanticTokens[i]) {
            return true;
        }
    }
    return false;
}

// this function receives any std::string 
// and returns a vector<byte> containing the numerical value of each byte in the string
// TODO: Optimize implementation
// FIXME: Do we need to free buffer after use?

std::vector<std::byte> getBytes(std::string const &s) {
    std::vector<std::byte> bytes;
    bytes.reserve(std::size(s));
    std::transform(std::begin(s), 
                   std::end(s),
                   std::back_inserter(bytes), 
                   [](char const &c){ return std::byte(c);});
    return bytes;
}

int tokType(const llama_context *ctx, const llama_token token) {

    int en = 0;
    int ru = 0;
    int other = 0;
    bool space = 0;

    // WAS std::string in = llama_token_to_str(ctx, token); // vocab.id_to_token[token].text
    std::string in = llama_token_to_piece(ctx, token); // vocab.id_to_token[token].text

    // DEBUG
    //std::string in = "хід";
    // in = "ё"; // 30043 => {209} {145} => {0xD1} {0x91}
    // in = "ë"; // 30083 => {195} {171} => {0xC3} {0xAB}
    //fprintf(stderr, "\n STR SIZE = %d \n", in.size());

    auto buf = getBytes(in);
    if (buf.size() > 0 && buf[0] == std::byte{0x20}) {
        space = true;
    }

    //for(size_t i = 0; i < buf.size(); i ++) { fprintf(stderr, " - %d", buf[i]); } exit(1); // DEBUG

    for(size_t i = 0; i < buf.size(); i ++) {

        // -- Simplified UTF-9 parsing 
        // TODO: Be more strict
        
        // -- ASCII Letters
        if (
            (buf[i] >= std::byte{0x41} && buf[i] <= std::byte{0x5A}) ||
            (buf[i] >= std::byte{0x61} && buf[i] <= std::byte{0x7A})) {
            en++;
            continue;
        }

        // -- ASCII Other
        if (buf[i] < std::byte{0x80}) {
            continue;
        }

        // -- UTF-8 RU 
        // https://www.utf8-chartable.de/unicode-utf8-table.pl?start=1024
        // except {0xD1} {0xE1} CYRILLIC SMALL LETTER IO == "ё"

        if (buf[i] == std::byte{0xD0} && (i + 1 < buf.size())) {
            i++;
            if ((buf[i] >= std::byte{0x90} && buf[i] <= std::byte{0xBF}) || 
                (buf[i] == std::byte{0x81}) // "Ё"
            )    
                ru++;
            else
                other++;     
            continue;
        }

        if (buf[i] == std::byte{0xD1} && (i + 1 < buf.size())) {
            i++;
            if (
                (buf[i] >= std::byte{0x80} && buf[i] <= std::byte{0x8F}) || 
                (buf[i] == std::byte{0x91}) // "ё"
            )
                ru++;
            else
                other++;    
            continue;
        }

        // -- UTF-8 2 bytes (European)
        if (buf[i] >= std::byte{0xC3} && buf[i] < std::byte{0xE3}) {
            i++;
            other++;
            continue;
        }

        // -- UTF-8 3 bytes (Asian)
        if (buf[i] >= std::byte{0xE3} && buf[i] < std::byte{0xF0}) {
            i += 2;
            other++;
            continue;
        }    

        // -- UTF-8 4 bytes (Emojis)
        if (buf[i] >= std::byte{0xF0}) {
            i += 3;
            continue;
        }    
    }

    if (space) { 
        if (other) return SPACE_OTHER;
        if (en) return SPACE_EN;
        if (ru) return SPACE_RU;   
    } 

    if (other) return LANG_OTHER;
    if (en) return LANG_EN;             
    if (ru) return LANG_RU;

    return LANG_ZERO;
}

// NB! isLower works only for RU and EN 
bool isLower(const llama_context *ctx, const llama_token token) {

    // WAS: std::string in = llama_token_to_str(ctx, token);
    std::string in = llama_token_to_piece(ctx, token);
    auto buf = getBytes(in);

    if (buf.size() <= 0) return false; 

    // -- ASCII Letters
    if (buf[0] >= std::byte{0x61} && buf[0] <= std::byte{0x7A}) {
        return true;
    }

    // -- UTF-8 RU 
    // https://www.utf8-chartable.de/unicode-utf8-table.pl?start=1024

    if (buf[0] == std::byte{0xD0} && buf.size() >= 2) {
        if ((buf[1] >= std::byte{0xB0} && buf[1] <= std::byte{0xBF}))
            return true;
    }

    if (buf[0] == std::byte{0xD1} && buf.size() >= 2) {
        if (
            (buf[1] >= std::byte{0x80} && buf[1] <= std::byte{0x8F}) || 
            (buf[1] == std::byte{0x91}) // "ё"
        )
            return true;
    }

    return false;
}

int tokSize(const llama_context *ctx, const llama_token token) {
    // WAS: return llama_token_to_str(ctx, token).size();
    return llama_token_to_piece(ctx, token).size();
}

void printDebug(struct llama_context * ctx, const int pos, const size_t shortlist, const char * text) {
    return; // DEBUG

    auto model = llama_get_model(ctx);
    float * logits = llama_get_logits(ctx);
    const int vocabSize = llama_n_vocab(model);

    std::vector<llama_token_data> candidates;
    candidates.clear();

    for (llama_token id = 0; id < vocabSize; id++) {
        candidates.emplace_back(llama_token_data{id, logits[id], 0.0f});
    }

    std::sort(
        candidates.data(), 
        candidates.data() + candidates.size(), 
        [](const llama_token_data & a, const llama_token_data & b) { 
            return a.logit > b.logit; 
        }
    );

    size_t size = std::min((int)candidates.size(), 8);
    fprintf(stderr, "\n\n==== %05d ==== %s ====", pos, text);

    for (size_t i = 0; i < size; i++) {

        auto id    = candidates.data()[i].id;
        auto logit = candidates.data()[i].logit;
        std::string zero = "";

        if (logit < 10.0) {
            zero = "0";
        }

        if (shortlist > 0 && i == shortlist) {
            fprintf(stderr, "\n  ---------------------------");
        }

        if (id == NL) {
            fprintf(stderr, 
                "\n  --    13 [ %s%.3f * %.3f ] \"\\n\"",
                zero.c_str(),
                logit, 
                ::scales[id]
            );
        } else if (id == EOS) {
            fprintf(stderr, 
                "\n  --     2 [ %s%.3f * %.3f ] \"<EOS>\"",
                zero.c_str(),
                logit, 
                ::scales[id]
            );
        } else {
            fprintf(stderr, 
                "\n  -- %5d [ %s%.3f * %.3f ] \"%s\"", 
                id,
                zero.c_str(),
                logit,
                ::scales[id],
                // WAS: llama_token_to_str(ctx, id).c_str()
                llama_token_to_piece(ctx, id).c_str()
            );
        }
    }
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

