#pragma once

#include <string>
#include <vector>

#include "llama.h"

#include "ggml-common.h"
#include "ggml-backend.h"

// -- Janus Sampling Parameters

struct janus_params {
    int32_t janus = 1;    // 0 = off or Janus Sampling version
    int32_t depth = 200;  // last n tokens to penalize [ -1 = context size ]
    float   scale = 0.96; // janus scale factor for penalty and other heuristics
    float   hi    = 0.99; // 1.0 = max pedantic [ 100% strict ]
    float   lo    = 0.96; // 0.0 = min pedantic [ 100% random ]
};

// -- Tokens

const int EOS = 2;
const int NL  = 13;

// -- Token types

const int LANG_ZERO = 0;
// const int LANG_NEUTRAL = 1;
// const int LANG_MIXED = 1;
const int LANG_EN = 2;
const int SPACE_EN = 20;
const int LANG_RU = 3;
const int SPACE_RU = 30;
const int LANG_OTHER = 4;
const int SPACE_OTHER = 40;

llama_token sample_janus_token(
    struct llama_context * ctx, 
    struct llama_sampling_params & sampling_params,
    struct janus_params & params, 
    const std::vector<llama_token> & last_tokens, 
    const size_t promptLen,
    const size_t pos,
    const size_t max);

///// std::string llama_token_to_str(const struct llama_context * ctx, llama_token token);

std::vector<std::byte> getBytes(std::string const &s);
bool isPedantic(struct llama_context * ctx, llama_token id);
bool isLower(const llama_context *ctx, const llama_token token);
int tokType(const llama_context *ctx, const llama_token token);
int tokSize(const llama_context *ctx, const llama_token token);
void initJanus(struct llama_context * ctx, struct janus_params & params, char * debug);
void printDebug(struct llama_context * ctx, const int pos, const size_t shortlist, const char * text);


// Get a string representation of the last sampled tokens
//std::string llama_sampling_prev_str(llama_sampling_context * ctx_sampling, llama_context * ctx_main, int n); 
