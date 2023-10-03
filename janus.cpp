#include "bridge.h"
#include "llama.h"
#include "ggml.h"
#include "janus.h"

#include <array>
#include <algorithm>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <shared_mutex>
#include <unordered_map>
#include <tuple>

// The Guardian has always been a newspaper for writers, 
// and so a newspaper for readers.

// The introduction to the Guardian stylebook of 1960, 
// which itself was a revision to the initial guide published in 1928, 
// was headed "Neither pedantic nor wild".

// https://www.theguardian.com/info/2000/mar/24/neither-pedantic-nor-wild

bool isJanusInitialized = false;

float * scales; // precomputed scales (penalties) for each token
float * types;  // precoputed types for each token

// -- Experimental approach Janus Sampling by gotzmann [ paper is coming ]

llama_token sample_janus_token(

        struct llama_context * ctx, 
        struct gpt_params & params,
        const std::vector<llama_token> & last_tokens,
        const size_t promptLen,
        const size_t pos,
        const size_t max) {

    if (!::isJanusInitialized) {
        initJanus(ctx, params);
        ::isJanusInitialized = true;
    }

    printDebug(ctx, pos, 0, "TOP LIST"); // -- DEBUG

    /* DEBUG
    fprintf(stderr, "\n * janus = %d", params.janus);
    fprintf(stderr, "\n * depth = %d", params.depth);
    fprintf(stderr, "\n * scale = %f", params.scale);
    fprintf(stderr, "\n * hi = %f", params.hi);
    fprintf(stderr, "\n * lo = %f", params.lo); */

    //const int64_t t_start_sample_us = ggml_time_us();

    float * logits   = llama_get_logits(ctx);
    size_t vocabSize = llama_n_vocab(ctx);
    size_t depth     = params.depth;
    //float scale      = params.scale;

    auto lastToken = last_tokens.data()[ last_tokens.size() - 1 ];
    auto lastType  = ::types[lastToken];
/*
    // -- Normalize all tokens agains their scales before doing anything

    for (size_t i = 1; i < vocabSize; i++) {
        logits[i] *= scales[i];
    }
*/
/*
    //llama_token topToken = 0;
    //float topLogit = logits[0];
    for (size_t i = 1; i < vocabSize; i++) {
        logits[i] *= scales[i];
        //if (logits[i] > topLogit) {
        //    topToken = i;
        //    topLogit = logits[i];
        //}       
    }
    //auto topType = types[topToken];

    // -- Slightly boost the top token when the word continuation expected
    //    It should allow better coherence for languages with complex grammar

    if (
        ((lastType == LANG_RU || lastType == SPACE_RU) && (topType == LANG_EN || topType == LANG_OTHER))
        ||
        ((lastType == LANG_EN || lastType == SPACE_EN) && topType == LANG_RU) // Europeans mix ASCII and UTF-8
    ) {
        logits[topToken] *= 1.0 + (1.0 / scale - 1.0) * 0.05; 
    } 
*/
    // -- Boost <EOS> token when we are closer to the limit
    //    NB! It looks like it enough just do not penalize it at all [ allowing scale == 1.0 ] ?

    logits[EOS] *= 1.0 + log(1.0 + float(pos - promptLen) / float(max)) * 0.05;

    // -- Smart pessimization for repeated tokens
    //    For better performance we are excluding prompt tokens

    size_t diveDepth = std::min(depth, pos - promptLen);
    for (size_t i = 0; i < diveDepth; i++) {
        //fprintf(stderr, " [ i=%d | pos=%d | depth=%d | len=%d ] ", i, pos, depth, promptLen); // DEBUG
        auto id = last_tokens.data()[ last_tokens.size() - i ]; 

        // Decrease reperition penalty for word continuation tokens to help prevent wrong wordings in complex languages
        // TODO: Maybe we need to skip the last token itself [ with check of i > 0 ] ?! 
        if ((lastType == SPACE_RU || lastType == LANG_RU) && ::types[id] == LANG_RU) {
            logits[id] *= 1.0 - (1.0 - ::scales[id]) * 0.20;
            continue;
        }

        // well, let just ignore negative probabilities
        // how it was before: logits[id] /= 1.0 + (penalty - 1.0) * 0.10;
        logits[id] *= ::scales[id];
    }

    // -- Double down incompatible tokens (like word endings in some other language)

    for (size_t id = 0; id < vocabSize; id++) {
        auto curType = ::types[id];

        if (
            ((lastType == LANG_RU || lastType == SPACE_RU) && (curType == LANG_EN || curType == LANG_OTHER))
            ||
            ((lastType == LANG_EN || lastType == SPACE_EN) && curType == LANG_RU) // Europeans mix ASCII and UTF-8
        ) {
            // was: logits[id] /= 1.0 + (penalty - 1.0) * 3.00;
            logits[id] *= 0.50; 
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
    //    We'll use some general cutoff for most of tokens
    //    and pedantic cutoff for sensitive ones

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

void initJanus(struct llama_context * ctx, struct gpt_params & params) {

    const int vocabSize = llama_n_vocab(ctx);
    ::scales = new float[vocabSize] {};
    ::types = new float[vocabSize] {};

    // -- safe defaults

    if (params.scale <= 0.0 || params.scale > 1.0) {
        params.scale = 0.96;
    }

    if (params.depth <= 0 || params.depth > params.n_predict) {
        params.depth = 200;
    }

    if (params.hi <= 0.0 || params.hi > 1.0) {
        params.hi = 0.99;
    }

    if (params.lo <= 0.0 || params.lo > 1.0) {
        params.lo = 0.96;
    }

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
            continue;
        }   

        // -- special case for complex languages like Russian. Do not penalise
        //    much tokens that might work as other word parts!

        float probes[] = { 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40, 0.42, 0.44, 0.48, 0.50 };

        if (type == LANG_RU && lower) {
            // NB! Size in bytes is 2x of UTF-8 chars for RU
            ::scales[id] = 1.0 - (1.0 - scale) * probes[len/2];
            continue;
        }

        // -- similar hack for EN

        if (type == LANG_EN && lower) {
            ::scales[id] = 1.0 - (1.0 - scale) * probes[len];
            continue;
        }

        // -- full penalization for other tokens

        ::scales[id] = scale;
    }

    // -- Assign manually specific penalties for high-frequency tokens
    // TODO: Need more work with real texts and statistical probabilities

    ::scales[0]     = 1.0;   // to be safe
    ::scales[EOS]   = scale; // penalize <EOS> in the beginning and allow it to boost over 1.0 later
    
    ::scales[NL]    = 1.0 - (1.0 - scale) * 0.10; // newline

    ::scales[259]   = 1.0 - (1.0 - scale) * 0.20; //   259 => "  "
    ::scales[268]   = 1.0 - (1.0 - scale) * 0.20; //   268 => "    "

    ::scales[29871] = 1.0 - (1.0 - scale) * 0.20; // 29871 => " "
    ::scales[29892] = 1.0 - (1.0 - scale) * 0.20; // 29892 => ","
    ::scales[29889] = 1.0 - (1.0 - scale) * 0.20; // 29889 => "."

    ::scales[29899] = 1.0 - (1.0 - scale) * 0.30; // 29899 => "-" [ used as bullet point ]
    ::scales[29901] = 1.0 - (1.0 - scale) * 0.30; // 29901 => ":"
    ::scales[29936] = 1.0 - (1.0 - scale) * 0.30; // 29936 => ";"
    ::scales[813]   = 1.0 - (1.0 - scale) * 0.30; // 813   => " —"

    ::scales[313]   = 1.0 - (1.0 - scale) * 0.30; // 313   => " ("
    ::scales[467]   = 1.0 - (1.0 - scale) * 0.30; // 467   => ")."
    ::scales[1723]  = 1.0 - (1.0 - scale) * 0.40; // 1723  => " )"
    ::scales[29897] = 1.0 - (1.0 - scale) * 0.60; // 29897 => ")"
    ::scales[29898] = 1.0 - (1.0 - scale) * 0.60; // 29898 => "("
    
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

    426,   // " {"
    500,   // " }"
    518,   // " ["
    4514,  // " ]"

    29912, // "{"
    29913, // "}"
    29961, // "["
    29962, // "]"

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

    std::string in = llama_token_to_str(ctx, token); // vocab.id_to_token[token].text

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

    std::string in = llama_token_to_str(ctx, token);
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
    return llama_token_to_str(ctx, token).size();
}

// TODO: It's duplicate
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

void printDebug(struct llama_context * ctx, const int pos, const size_t shortlist, const char * text) {
    // return; // !!!

    float * logits = llama_get_logits(ctx);
    const int vocabSize = llama_n_vocab(ctx);

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
                llama_token_to_str(ctx, id).c_str()
            );
        }
    }
}
