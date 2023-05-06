//#ifdef __cplusplus
//extern "C" {
//#endif

void * initFromParams(char * modelName);
void * tokenize(void * ctx, char * prompt);
void loop(struct llama_context * ctx, std::vector<llama_token> & embd_inp);

//#define __has_cpp_attribute(__x) 0
//#define __has_keyword(__x) !(__is_identifier(__x))

//#include <string>
//std::string prompt;

//#define __has_keyword(__x) !(__is_identifier(__x))
//#define __has_cpp_attribute(__x) 0

//#include <string>
//#include <vector>

//#include "examples/common.h"
//#include <stdbool.h>

//
// Model utils
//

//struct llama_context * llama_init_from_gpt_params(const gpt_params & params);
/*
void * initFromParams(char * modelName) {
   gpt_params params;
   params.model = modelName;
   //return llama_init_from_gpt_params(&params);
   return NULL;
}*/

//struct llama_context * llama_init_from_gpt_params(const gpt_params & params);


/*
void *llama_allocate_state();

int llama_bootstrap(const char *model_path, void* state_pr, int n_ctx, int n_parts, int memory_type_int);

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty, int repeat_last_n);
void llama_free_params(void* params_ptr);

int llama_predict(void* params_ptr, void* state_pr);
*/

//#ifdef __cplusplus
}
//#endif