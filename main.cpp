/*
#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include "common.h"

struct llama_context * initFromParams(char * modelName) {
    gpt_params params;
    params.model = modelName;
    return llama_init_from_gpt_params(&params);
}

#ifdef __cplusplus
}
#endif
*/