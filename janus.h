#include "llama.h"
#include <string>
#include <vector>

const int LANG_ZERO = 0;
// const int LANG_NEUTRAL = 1;
// const int LANG_MIXED = 1;
const int LANG_EN = 2;
const int SPACE_EN = 20;
const int LANG_RU = 3;
const int SPACE_RU = 30;
const int LANG_OTHER = 4;
const int SPACE_OTHER = 40;

static std::string llama_token_to_str(const struct llama_context * ctx, llama_token token);

std::vector<std::byte> getBytes(std::string const &s);
bool isPedantic(llama_token id);
int tokType(const llama_context *ctx, const llama_token token);
int tokSize(const llama_context *ctx, const llama_token token);
void initJanus(struct llama_context * ctx, const struct gpt_params & params);
void printDebug(struct llama_context * ctx, const int pos, const char * text);