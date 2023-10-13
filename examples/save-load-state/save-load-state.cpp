#include "build-info.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>
#include <chrono>

int main(int argc, char ** argv) {
    gpt_params params;
    llama_sampling_params & sparams = params.sampling_params;
    params.seed = 42;
    params.n_threads = 4;
    sparams.repeat_last_n = 64;
    params.prompt = "The quick brown fox";

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    print_build_info();

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    auto n_past = 0;
    auto last_n_tokens_data = std::vector<llama_token>(sparams.repeat_last_n, 0);

    // init
    llama_model * model;
    llama_context * ctx;

    std::tie(model, ctx) = llama_init_from_gpt_params( params );
    if (model == nullptr) {
        return 1;
    }
    if (ctx == nullptr) {
        llama_free_model(model);
        return 1;
    }
    auto tokens = llama_tokenize(ctx, params.prompt, true);
    auto n_prompt_tokens = tokens.size();
    if (n_prompt_tokens < 1) {
        fprintf(stderr, "%s : failed to tokenize prompt\n", __func__);
        llama_free(ctx);
        llama_free_model(model);
        return 1;
    }

    // evaluate prompt
    llama_decode(ctx, llama_batch_get_one(tokens.data(), n_prompt_tokens, n_past, 0));

    last_n_tokens_data.insert(last_n_tokens_data.end(), tokens.data(), tokens.data() + n_prompt_tokens);
    n_past += n_prompt_tokens;

    const size_t state_size = llama_get_state_size(ctx);
    uint8_t * state_mem = new uint8_t[state_size];

    // Save state (rng, logits, embedding and kv_cache) to file
    {
        FILE *fp_write = fopen("dump_state.bin", "wb");
        llama_copy_state_data(ctx, state_mem); // could also copy directly to memory mapped file
        fwrite(state_mem, 1, state_size, fp_write);
        fclose(fp_write);
    }

    // save state (last tokens)
    const auto last_n_tokens_data_saved = std::vector<llama_token>(last_n_tokens_data);
    const auto n_past_saved = n_past;

    // first run
    printf("\n%s", params.prompt.c_str());

    for (auto i = 0; i < params.n_predict; i++) {
        auto * logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(model);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        auto next_token = llama_sample_token(ctx, &candidates_p);
        auto next_token_str = llama_token_to_piece(ctx, next_token);
        last_n_tokens_data.push_back(next_token);

        printf("%s", next_token_str.c_str());
        if (llama_decode(ctx, llama_batch_get_one(&next_token, 1, n_past, 0))) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_free(ctx);
            llama_free_model(model);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");

    // free old context
    llama_free(ctx);

    // make new context
    auto * ctx2 = llama_new_context_with_model(model, llama_context_params_from_gpt_params(params));

    // Load state (rng, logits, embedding and kv_cache) from file
    {
        FILE *fp_read = fopen("dump_state.bin", "rb");
        if (state_size != llama_get_state_size(ctx2)) {
            fprintf(stderr, "\n%s : failed to validate state size\n", __func__);
            llama_free(ctx2);
            llama_free_model(model);
            return 1;
        }

        const size_t ret = fread(state_mem, 1, state_size, fp_read);
        if (ret != state_size) {
            fprintf(stderr, "\n%s : failed to read state\n", __func__);
            llama_free(ctx2);
            llama_free_model(model);
            return 1;
        }

        llama_set_state_data(ctx2, state_mem);  // could also read directly from memory mapped file
        fclose(fp_read);
    }

    delete[] state_mem;

    // restore state (last tokens)
    last_n_tokens_data = last_n_tokens_data_saved;
    n_past = n_past_saved;

    // second run
    for (auto i = 0; i < params.n_predict; i++) {
        auto * logits = llama_get_logits(ctx2);
        auto n_vocab = llama_n_vocab(model);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        auto next_token = llama_sample_token(ctx2, &candidates_p);
        auto next_token_str = llama_token_to_piece(ctx2, next_token);
        last_n_tokens_data.push_back(next_token);

        printf("%s", next_token_str.c_str());
        if (llama_decode(ctx, llama_batch_get_one(&next_token, 1, n_past, 0))) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            llama_free(ctx2);
            llama_free_model(model);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");

    llama_free(ctx2);
    llama_free_model(model);

    return 0;
}
