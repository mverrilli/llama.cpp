// kpc eager-close: 4 seqs round-robin in 32-token chunks through L=2 slots; boundary evictions are lossless,
// so next-token logits match a default L=4 run bit-for-bit

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include "kpc-test-utils.h"

#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed   = 1234;
    params.kv_unified      = true;
    params.n_parallel      = 4;
    params.n_ctx           = 1024;
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

    const int         NS     = 4;
    const int         ROUNDS = 2;    // 32-token chunks per sequence (so each ends at pos 64)
    const llama_token tok    = 1;

    // round-robin: each sequence's chunk is its own decode, so when seq k decodes, seqs != k are idle and evictable
    auto run = [&](llama_context * ctx, std::vector<std::vector<float>> & out) -> bool {
        for (int r = 0; r < ROUNDS; ++r) {
            for (int s = 0; s < NS; ++s) {
                if (!decode_chunk(ctx, s, r*32, tok)) {
                    return false;
                }
            }
        }
        out.resize(NS);
        for (int s = 0; s < NS; ++s) {
            out[s] = decode_one(ctx, tok, ROUNDS*32, s, n_vocab);
        }
        return true;
    };

    // baseline: default staging (L == n_seq_max == 4), no eviction
    std::vector<std::vector<float>> base;
    {
        unsetenv("KPC_STAGING_SLOTS");
        unsetenv("KPC_EAGER_CLOSE");
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!run(ctx, base)) { fprintf(stderr, "%s: baseline run failed\n", __func__); llama_free(ctx); return 1; }
        llama_free(ctx);
    }

    // eager-close: only L=2 slots, evict idle sequences on slot pressure (would exhaust without KPC_EAGER_CLOSE)
    std::vector<std::vector<float>> eager;
    {
        setenv("KPC_STAGING_SLOTS", "2", 1);
        setenv("KPC_EAGER_CLOSE",   "1", 1);
        llama_context * ctx = llama_init_from_model(model, cparams);
        const bool ok = run(ctx, eager);
        llama_free(ctx);
        unsetenv("KPC_STAGING_SLOTS");
        unsetenv("KPC_EAGER_CLOSE");
        if (!ok) { fprintf(stderr, "%s: FAIL - eager-close run did not complete (aborted/exhausted)\n", __func__); return 1; }
    }

    const float eps = 1e-3f;
    float worst = 0.0f;
    for (int s = 0; s < NS; ++s) {
        const float d = max_abs_diff(base[s], eager[s]);
        fprintf(stderr, "%s: seq %d logit max|diff| = %.6f\n", __func__, s, d);
        worst = std::max(worst, d);
    }
    if (worst >= eps) {
        fprintf(stderr, "%s: FAIL - eager-close (L=2) diverges from default (L=4)\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s: PASS - %d seqs through L=2 slots with boundary evictions match the default bit-for-bit\n", __func__, NS);
    return 0;
}
