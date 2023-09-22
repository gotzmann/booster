#include "llama.h"
#include "common.h"
#include "console.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <codecvt>
#include <map>
#include <vector>
#include <locale>

typedef int codepoint;

static std::string codepoint_to_utf8(codepoint cp) {
    std::string result;
    if (0x00 <= cp && cp <= 0x7f) {
        result.push_back(cp);
    } else if (0x80 <= cp && cp <= 0x7ff) {
        result.push_back(0xc0 | ((cp >> 6) & 0x1f));
        result.push_back(0x80 | (cp & 0x3f));
    } else if (0x800 <= cp && cp <= 0xffff) {
        result.push_back(0xe0 | ((cp >> 12) & 0x0f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
    } else if (0x10000 <= cp && cp <= 0x10ffff) {
        result.push_back(0xf0 | ((cp >> 18) & 0x07));
        result.push_back(0x80 | ((cp >> 12) & 0x3f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
    } else {
        throw std::invalid_argument("invalid codepoint");
    }
    return result;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vocab-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    fprintf(stderr, "%s : reading vocab from: '%s'\n", __func__, fname.c_str());

    llama_model * model;
    llama_context * ctx;

    llama_backend_init(false);

    // load the vocab
    {
        auto lparams = llama_context_default_params();

        lparams.vocab_only = true;

        model = llama_load_model_from_file(fname.c_str(), lparams);

        if (model == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            return 1;
        }

        ctx = llama_new_context_with_model(model, lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            llama_free_model(model);
            return 1;
        }
    }

    GGML_ASSERT(llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM);

#ifdef _WIN32
    // We need this for unicode console support
    console::init(false, false);
    atexit([]() { console::cleanup(); });
#endif

    const int n_vocab = llama_n_vocab(ctx);

    for (int i = 0; i < n_vocab; ++i) {
        std::string str = llama_detokenize_spm(ctx, std::vector<int>(1, i));
        std::vector<llama_token> tokens = llama_tokenize(ctx, str, false);
        std::string check = llama_detokenize_spm(ctx, tokens);
        if (check != str) {
            fprintf(stderr, "%s : error: token %d detokenizes to '%s'(%zu) but tokenization of this detokenizes to '%s'(%zu)\n",
                __func__, i, str.c_str(), str.length(), check.c_str(), check.length());
            return 2;
        }
    }

    for (codepoint cp = 0x0000; cp < 0xffff; ++cp) {
        if (cp < 0xd800 || cp > 0xdfff) {
            std::string str = codepoint_to_utf8(cp);
            std::vector<llama_token> tokens = llama_tokenize(ctx, str, false);
            std::string check = llama_detokenize_spm(ctx, tokens);
            if (cp != 9601 && str != check) {
                fprintf(stderr, "%s : error: codepoint %d detokenizes to '%s'(%zu) instead of '%s'(%zu)\n",
                    __func__, cp, check.c_str(), check.length(), str.c_str(), str.length());
                return 3;
            }
        }
    }
    for (codepoint cp = 0x10000; cp < 0x0010ffff; ++cp) {
        std::string str = codepoint_to_utf8(cp);
        std::vector<llama_token> tokens = llama_tokenize(ctx, str, false);
        std::string check = llama_detokenize_spm(ctx, tokens);
        if (str != check) {
            fprintf(stderr, "%s : error: codepoint %d detokenizes to '%s'(%zu) instead of '%s'(%zu)\n",
                __func__, cp, check.c_str(), check.length(), str.c_str(), str.length());
            return 4;
        }
    }

    llama_free_model(model);
    llama_free(ctx);

    llama_backend_free();

    return 0;
}
