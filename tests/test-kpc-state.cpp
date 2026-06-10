// kpc4_1 whole-state save/restore on a unified multi-sequence cache: two seqs left
// mid-group, save, restore into a fresh context; next-token logits must match the
// no-restore baseline for both (open-group staging is serialized per sequence)

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include "kpc-test-utils.h"

#include <vector>
#include <cstdio>
#include <cmath>

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;
    params.kv_unified     = true;                          // all sequences packed into one stream
    params.n_parallel     = 2;                             // n_seq_max = 2
    params.n_ctx          = 256;
    common_init_result_ptr init = kpc_model_init(argc, argv, params, __func__);
    if (!init || init->model() == nullptr) {
        return 1;
    }
    if (init->context() == nullptr) {
        return 0;
    }
    llama_model * model = init->model();

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    const auto          cparams = common_context_params_to_llama(params);

    const int         P   = 40;   // leaves both sequences mid-group (open members 32..39)
    const llama_token tok = 1;

    // snapshot state and take baseline logits
    std::vector<uint8_t> state;
    std::vector<float> l0_base, l1_base;
    {
        llama_context * ctx = init->context();

        llama_batch batch = llama_batch_init(2 * P, 0, 1);
        for (int i = 0; i < P; ++i) {
            for (int s = 0; s < 2; ++s) {
                common_batch_add(batch, tok, i, { s }, false);
            }
        }
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s: prompt decode failed\n", __func__);
            return 1;
        }
        llama_batch_free(batch);

        state.resize(llama_state_get_size(ctx));
        llama_state_get_data(ctx, state.data(), state.size());

        l0_base = decode_one(ctx, tok, P, 0, n_vocab);
        l1_base = decode_one(ctx, tok, P, 1, n_vocab);
    }

    // ctx 2: a fresh context (its KPC staging starts cleared), restore the saved state, then take the
    // same next-token logits. They must match the baseline for both sequences.
    std::vector<float> l0_rest, l1_rest;
    {
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            fprintf(stderr, "%s: failed to create second context\n", __func__);
            return 1;
        }
        const size_t nset = llama_state_set_data(ctx, state.data(), state.size());
        if (nset != state.size()) {
            fprintf(stderr, "%s: state restore consumed %zu of %zu bytes\n", __func__, nset, state.size());
            llama_free(ctx);
            return 1;
        }
        l0_rest = decode_one(ctx, tok, P, 0, n_vocab);
        l1_rest = decode_one(ctx, tok, P, 1, n_vocab);
        llama_free(ctx);
    }

    // ctx 3: restoring into a context with a different n_seq_max must be refused cleanly
    // (pool ids were laid out under n_seq_max=2 bands); a clean refusal consumes 0 bytes
    {
        auto cparams_big = cparams;
        cparams_big.n_seq_max = 4;
        llama_context * ctx = llama_init_from_model(model, cparams_big);
        if (ctx == nullptr) {
            fprintf(stderr, "%s: failed to create cross-size context\n", __func__);
            return 1;
        }
        const size_t nset = llama_state_set_data(ctx, state.data(), state.size());
        llama_free(ctx);
        if (nset != 0) {
            fprintf(stderr, "%s: FAIL - cross-n_seq_max restore was not refused (consumed %zu of %zu bytes)\n",
                    __func__, nset, state.size());
            return 1;
        }
        fprintf(stderr, "%s: cross-n_seq_max (2 -> 4) restore refused cleanly\n", __func__);
    }

    const float eps = 1e-3f;
    const float d0 = max_abs_diff(l0_base, l0_rest);
    const float d1 = max_abs_diff(l1_base, l1_rest);

    fprintf(stderr, "%s: seq 0 logit max|diff| = %.6f (control)\n",     __func__, d0);
    fprintf(stderr, "%s: seq 1 logit max|diff| = %.6f (staging fix)\n", __func__, d1);

    if (d0 >= eps || d1 >= eps) {
        fprintf(stderr, "%s: FAIL - restored state diverges from baseline (per-sequence staging not preserved)\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s: PASS - unified KPC whole-state round-trip preserves per-sequence staging\n", __func__);
    return 0;
}
