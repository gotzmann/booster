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
float * penalties;

// -- Experimental approach Janus Sampling by gotzmann [ paper is coming ]

llama_token sample_janus_token(
        struct llama_context * ctx, 
        const struct gpt_params & params, 
        //float * logits, 
        //const int size,
        const std::vector<llama_token> & last_tokens, 
        const int pos, 
        const int max) {

    if (!isJanusInitialized) {
        initJanus(ctx, params);
    }

    /* DEBUG
    fprintf(stderr, "\n * janus = %d", params.janus);
    fprintf(stderr, "\n * depth = %d", params.depth);
    fprintf(stderr, "\n * penalty = %f", params.penalty);
    fprintf(stderr, "\n * hi_p = %f", params.hi_p);
    fprintf(stderr, "\n * lo_p = %f", params.lo_p); */

    //const int64_t t_start_sample_us = ggml_time_us();

    float * logits = llama_get_logits(ctx);
    const int vocabSize = llama_n_vocab(ctx);
    float penalty = params.penalty;

    printDebug(ctx, pos, 0, "TOP LIST"); // -- DEBUG 

    // -- Boost <EOS> token when we are nearing prediction limits

    const int EOS = 2;
    // was: float mult = 1.0 + 0.2 * log(1.0 + (float(pos) / float(max)));
    float mult = penalty + log(1.0 + float(pos) / float(max)) * 0.2;
    logits[EOS] *= mult;

    // -- Apply penalty for repeated tokens except pedantic

    // Look up last tokens for certain depth in reverese order [ context_size .. depth ]
    size_t depth = 0;
    if (params.depth != -1 && params.depth != 0 && params.depth < (int32_t) last_tokens.size()) {
        depth = last_tokens.size() - params.depth;
    }

    for (size_t i = last_tokens.size() - 1; i >= depth; i--) {

        llama_token id = last_tokens.data()[i]; 
        if (id == 0) break; // stop looping after reaching the end of previously generated tokens 

        // well, let just ignore negative probabilities
        // how it was before: logits[id] /= 1.0 + (penalty - 1.0) * 0.10;
        logits[id] *= penalties[id];
    }

    // -- DOUBLE penalty for incompatible tokens (like english ending for russian word)

    auto lastToken = last_tokens.data()[last_tokens.size() - 1];
    auto lastType = tokType(ctx, lastToken);
    
    if (lastToken != 0) {
        
        //fprintf(stderr, "\n=== LAST \"%s\" === TYPE %d === ", llama_token_to_str(ctx, lastToken).c_str(), lastType);

        for (llama_token id = 0; id < vocabSize; id++) {
            auto curType = tokType(ctx, id);

            if(
                ((lastType == LANG_RU || lastType == SPACE_RU) && (curType == LANG_EN || curType == LANG_OTHER))
                ||
                ((lastType == LANG_EN || lastType == SPACE_EN) && curType == LANG_RU) // It's OK to expect other lang, europeans mix ASCII and UTF-8
            ) {
                // was: logits[id] /= 1.0 + (penalty - 1.0) * 3.00;

                // 3x: 0.936 => 0.808
                // 3x: [ 12.061 * 0.987 ] "mi" => [ 11.598 * 0.987 ] "mi" => 3x is not enough!
                // 10x: 0.936 => 0.36
                // 10x: [ 12.445 * 0.987 ] "mi" => [ 10.852 * 0.987 ] "mi" => 10x so-so
                // logits[id] *= 1.0 - (1.0 - ::penalties[id]) * 100.00;

                logits[id] *= 0.5; 
            }
        }        
    }

    // -- finally sort all logits

    std::vector<llama_token_data> candidates;
    //candidates.reserve(llama_n_vocab(ctx));
    candidates.clear();

    for (llama_token id = 0; id < vocabSize; id++) {
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

    //printDebug(ctx, pos, 0, "AFTER JANUS"); // -- DEBUG

    // -- Final choice [ with experimental cutoff ]
    //    We'll use some general cutoff for most of tokens
    //    and pedantic cutoff for sensitive ones

    auto topToken = candidates.data()[0].id;
    auto topType = tokType(ctx, topToken);
    float topLogit = candidates.data()[0].logit;

    float cutoff = params.lo_p;
    if (isPedantic(topType) || lastType == SPACE_RU || lastType == LANG_RU) {
        cutoff = params.hi_p;
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

// -- initJanus prefills base penalties for each token depending on Janus Sampling euristics

void initJanus(struct llama_context * ctx, const struct gpt_params & params) {

    const int vocabSize = llama_n_vocab(ctx);
    ::penalties = new float[vocabSize] {};
    float penalty = params.penalty;

    // -- init tokens with some heuristic rules

    // how it was before: logits[id] /= 1.0 + (penalty - 1.0) * 0.10;
    // and then wrong: ::penalties[id] = penalty * prob;
    for (llama_token id = 0; id < vocabSize; id++) {

        auto tokenType = tokType(ctx, id);      

        // -- pedantic tokens

        if (isPedantic(id)) {
            ::penalties[id] = 1.0 - (1.0 - penalty) * 0.05;
            continue;
        }   

        // -- Special case for complex languages like Russian
        //    Do not penalise much tokens that might work as other words parts!

        if (tokenType == LANG_RU) {
            float prob = (float) tokSize(ctx, id) * 0.05; // 0.1, 0.2, 0.3 ...
            ::penalties[id] = 1.0 - (1.0 - penalty) * prob;
            continue;
        }

        // -- Similar hack for EN (slightly decrease penalty ) ?!

        tokenType = tokType(ctx, id);
        if (tokenType == LANG_EN) {
            float prob = (float) tokSize(ctx, id) * 0.1; // 0.1, 0.2, 0.3 ...
            ::penalties[id] = 1.0 - (1.0 - penalty) * prob;
            continue;
        }

        // -- Full penalization for other tokens

        ::penalties[id] = penalty;

    }

    // -- rewrite some specific penalties for high-frequency tokens
    
    ::penalties[13]    = 1.0 - (1.0 - penalty) * 0.05; // newline

    ::penalties[29871] = 1.0 - (1.0 - penalty) * 0.20; // 29871 => " "

    ::penalties[29892] = 1.0 - (1.0 - penalty) * 0.10; // 29892 => ","
    ::penalties[29889] = 1.0 - (1.0 - penalty) * 0.10; // 29889 => "."

    ::penalties[29901] = 1.0 - (1.0 - penalty) * 0.30; // 29901 => ":"
    ::penalties[29936] = 1.0 - (1.0 - penalty) * 0.30; // 29936 => ";"

    ::penalties[29898] = 1.0 - (1.0 - penalty) * 0.40; // 29898 => "("
    ::penalties[313]   = 1.0 - (1.0 - penalty) * 0.40; // 313   => " ("
    ::penalties[29897] = 1.0 - (1.0 - penalty) * 0.40; // 29897 => ")"
    ::penalties[1723]  = 1.0 - (1.0 - penalty) * 0.40; // 1723  => " )"

    // -- Popular RU parts

    ::penalties[490]   = 1.0 - (1.0 - penalty) * 0.10; // 490 => " в"
    ::penalties[531]   = 1.0 - (1.0 - penalty) * 0.10; // 531 => " с"
    ::penalties[606]   = 1.0 - (1.0 - penalty) * 0.10; // 606 => " и"
    ::penalties[614]   = 1.0 - (1.0 - penalty) * 0.10; // 614 => " о"

    ::penalties[665]   = 1.0 - (1.0 - penalty) * 0.20; // 665 => " на"
    ::penalties[733]   = 1.0 - (1.0 - penalty) * 0.20; // 733 => " по"
    ::penalties[863]   = 1.0 - (1.0 - penalty) * 0.20; // 863 => " у"

    // -- Popular EN parts

    ::penalties[263]   = 1.0 - (1.0 - penalty) * 0.10; // 263 => " a"
    ::penalties[278]   = 1.0 - (1.0 - penalty) * 0.10; // 278 => " the"
    ::penalties[297]   = 1.0 - (1.0 - penalty) * 0.10; // 297 => " in"
    ::penalties[304]   = 1.0 - (1.0 - penalty) * 0.10; // 304 => " to"
    ::penalties[310]   = 1.0 - (1.0 - penalty) * 0.10; // 310 => " of"

    ::penalties[322]   = 1.0 - (1.0 - penalty) * 0.20; // 322 => " and"
    ::penalties[372]   = 1.0 - (1.0 - penalty) * 0.20; // 372 => " it"
    ::penalties[373]   = 1.0 - (1.0 - penalty) * 0.20; // 373 => " on"
    ::penalties[385]   = 1.0 - (1.0 - penalty) * 0.20; // 385 => " an"
}

// Tokens very often used for math, coding and JSON (aka repetitive),
// so we should be care about them and not penalize
llama_token pedanticTokens[] = {

    2, // <EOS>

    // -- Code

    28956 , // "```"

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
    // 29922, // "="
    // 353, // " ="
    // 29974, // "+"
    // 718, // " +"
    // 448, // " -"

    // -- JSON

    29912, // "{"
    426, // " {"
    29913, // "}"
    500, // " }"
    29961, // "["
    518, // " ["
    29962, // "]"
    4514, // " ]"

    // 29898, // "("
    // 313, // " ("
    // 29897, // ")"
    // 1723, // " )"
    // 3319, // "({"
    // 1800, // "})"
    // 4197, // "(["
    // 29889, // "."
    // 29892, // ","
    // 29901, // ":"
    // 29936, // ";"

    // -- JSON

    // 29908, // """
    // 376, //  " ""
    // 1115, // "":"
    // 4710, // "":""
    // 613, // "","
    // 8853, // " {""
    // 29871, // " "
};

bool isPedantic(llama_token id) {
    size_t pedanticLen = *(&pedanticTokens + 1) - pedanticTokens;
    for (size_t i = 0; i < pedanticLen; i++) {
        if (id == pedanticTokens[i]) {
            return true;
        }
    }
    return false;
}

// this function receives any std::string 
// and returns a vector<byte> containing the numerical value of each byte in the string
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
    //in = "ёл";
    //in = "ё"; // 30043 => {209} {145} => {0xD1} {0x91}
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
            if ((buf[i] >= std::byte{0x90} && buf[i] <= std::byte{0xBF}))
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

int tokSize(const llama_context *ctx, const llama_token token) {
    return llama_token_to_str(ctx, token).size();
}

void printDebug(struct llama_context * ctx, const int pos, const size_t shortlist, const char * text) {

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

        auto id =candidates.data()[i].id;
        auto logit = candidates.data()[i].logit;
        std::string zero = "";

        if (logit < 10.0) {
            zero = "0";
        }

        if (shortlist > 0 && i == shortlist) {
            fprintf(stderr, "\n  ---------------------------");
        }

        if (13 == id) {
            fprintf(stderr, 
                "\n  --    13 [ %s%.3f * %.3f ] \"\\n\"",
                zero.c_str(),
                logit, 
                penalties[id]
            );
        } else if (2 == id) {
            fprintf(stderr, 
                "\n  --     2 [ %s%.3f * %.3f ] \"<EOS>\"",
                zero.c_str(),
                logit, 
                penalties[id]
            );
        } else {
            fprintf(stderr, 
                "\n  -- %5d [ %s%.3f * %.3f ] \"%s\"", 
                id,
                zero.c_str(),
                logit,
                penalties[id],
                llama_token_to_str(ctx, id).c_str()
            );
        }
    }
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
