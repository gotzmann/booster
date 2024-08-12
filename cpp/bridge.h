#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#if !defined (_WIN32)
#include <stdio.h>
#include <termios.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include "unistd.h"

#include "llama.h"
#include "common.h"
#include "sampling.h"

#include "ggml.h"
#include "ggml-common.h"
#include "ggml-backend.h"

#ifdef _WIN32
#define NULL_DEVICE "NUL:"
#define TTY_DEVICE "COM1:"
#else
#define NULL_DEVICE "/dev/null"
#define TTY_DEVICE "/dev/tty"
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


/*****

// sampling parameters
typedef struct llama_sampling_params {

    // -- Janus Sampling

    int32_t janus = 1;    // 0 = off or Janus Sampling version
    int32_t depth = 200;  // last n tokens to penalize [ -1 = context size ]
    float   scale = 0.96; // janus scale factor for penalty and other heuristics
    float   hi    = 0.99; // 1.0 = max pedantic [ 100% strict ]
    float   lo    = 0.96; // 0.0 = min pedantic [ 100% random ]

    // -- mainstream samplings

    int32_t     n_prev                = 64;                 // number of previous tokens to remember
    int32_t     n_probs               = 0;                  // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t     min_keep              = 0;                  // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t     top_k                 = 40;                 // <= 0 to use vocab size
    float       top_p                 = 0.95f;              // 1.0 = disabled
    float       min_p                 = 0.05f;              // 0.0 = disabled
    float       tfs_z                 = 1.00f;              // 1.0 = disabled
    float       typical_p             = 1.00f;              // 1.0 = disabled
    float       temp                  = 0.80f;              // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float       dynatemp_range        = 0.00f;              // 0.0 = disabled
    float       dynatemp_exponent     = 1.00f;              // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t     penalty_last_n        = 64;                 // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float       penalty_repeat        = 1.00f;              // 1.0 = disabled
    float       penalty_freq          = 0.00f;              // 0.0 = disabled
    float       penalty_present       = 0.00f;              // 0.0 = disabled
    int32_t     mirostat              = 0;                  // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float       mirostat_tau          = 5.00f;              // target entropy
    float       mirostat_eta          = 0.10f;              // learning rate
    bool        penalize_nl           = false;              // consider newlines as a repeatable token
    uint32_t    seed                  = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampling_context

    std::vector<llama_sampler_type> samplers_sequence = {
        llama_sampler_type::TOP_K,
        llama_sampler_type::TFS_Z,
        llama_sampler_type::TYPICAL_P,
        llama_sampler_type::TOP_P,
        llama_sampler_type::MIN_P,
        llama_sampler_type::TEMPERATURE
    };

    std::string grammar;  // optional BNF-like grammar to constrain sampling

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    std::string cfg_negative_prompt; // string to help guidance
    float       cfg_scale     = 1.f; // how strong is guidance

    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens

    std::vector<llama_token> penalty_prompt_tokens;
    bool                     use_penalty_prompt_tokens = false;
} llama_sampling_params;

*****/

// ---

void hide();
void show();

struct llama_context * init_context(int idx);
int64_t do_inference(
    int idx, 
    struct llama_context * ctx, 
    const std::string & jobID, 
    const std::string & sessionID, 
    const std::string & text);

const char * statusCPP(const std::string & jobID);
int64_t promptEvalCPP(const std::string & jobID);
int64_t getPromptTokenCountCPP(const std::string & jobID);
int64_t timingCPP(const std::string & jobID);
uint32_t getSeedCPP(const std::string & jobID);

extern "C" { // -----    

void init(char * swap, char * debug);

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
    char * debug);

int64_t doInference(
    int idx, 
    void * ctx, 
    char * jobID, 
    char * sessionID, 
    char * prompt); 

void stopInference(int idx);
const char * status(char * jobID);
int64_t promptEval(char * jobID);
int64_t getPromptTokenCount(char * jobID);
int64_t timing(char * jobID);  
uint32_t getSeed(char * jobID);  

} // ------- extern "C"

std::vector<std::byte> getBytes(std::string const &s);

int tokType(const llama_context *ctx, const llama_token token);
int tokSize(const llama_context *ctx, const llama_token token);

// This works fast and allows for 100% deterministic sampling
llama_token sample_top_token(const float * logits, const int size);
