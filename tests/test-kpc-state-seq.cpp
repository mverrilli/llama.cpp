// kpc4_1 PER-SEQUENCE state save/restore: decode two sequences, save ONLY seq 1 via llama_state_seq_get_data,
// restore it into another context, and require seq-1 next-token logits to match the no-restore baseline.
// Covers, via env vars, the configs that gate server adoption:
//   KPC_SEQ_RAW=1          -> Stage 2 RAW fast path (int4+scalezp verbatim, bit-exact on aligned restore)
//   KPC_STAGING_SLOTS=1    -> virtualized staging (slot < n_seq_max)
//   KPC_TEST_NONUNIFIED=1  -> non-unified cache (each sequence its own stream)
// Two restore targets are checked: a FRESH context, and a LIVE context where seq 0 is already decoded (so seq 1
// lands in non-empty, non-32-aligned cells and must not collide with seq 0).

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "kpc-test-utils.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// restore seq 1 from `seqstate` into a new context (optionally with seq 0 already live), return seq-1 logits;
// when live, also writes seq-0 logits to *l0_out so the caller can check seq 0 was not disturbed.
static std::vector<float> restore_and_decode(llama_model * model, const llama_context_params & cparams,
                                             const std::vector<uint8_t> & seqstate, int n_vocab, int P,
                                             llama_token tok, bool live_seq0, std::vector<float> * l0_out) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        return {};
    }
    if (live_seq0) {
        llama_batch b = llama_batch_init(P, 0, 1);
        for (int i = 0; i < P; ++i) {
            common_batch_add(b, tok, i, { 0 }, false);
        }
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc) { fprintf(stderr, "  live seq0 decode failed\n"); llama_free(ctx); return {}; }
    }
    const size_t nset = llama_state_seq_set_data(ctx, seqstate.data(), seqstate.size(), 1);
    if (nset != seqstate.size()) {
        fprintf(stderr, "  restore consumed %zu of %zu bytes\n", nset, seqstate.size());
        llama_free(ctx);
        return {};
    }
    std::vector<float> l1 = decode_one(ctx, tok, P, 1, n_vocab);
    if (live_seq0 && l0_out) {
        *l0_out = decode_one(ctx, tok, P, 0, n_vocab);
    }
    llama_free(ctx);
    return l1;
}

int main(int argc, char ** argv) {
    const bool nonunified = getenv("KPC_TEST_NONUNIFIED") != nullptr;

    common_params params;
    params.sampling.seed = 1234;
    params.kv_unified     = !nonunified;   // unified (shared stream, banded pools) vs non-unified (per-seq streams)
    params.n_parallel     = 2;             // n_seq_max = 2
    params.n_ctx          = 256;
    common_init_result_ptr init = kpc_model_init(argc, argv, params, __func__);
    if (!init || init->model() == nullptr) {
        return 1;
    }
    if (init->context() == nullptr) {
        return 0;   // context unavailable (head_dim guard) -> skip cleanly
    }
    llama_model * model = init->model();

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    const auto          cparams = common_context_params_to_llama(params);

    const int         P   = 40;   // leaves seq 1 mid-group (open members 32..39) -> exercises the staging tail
    const llama_token tok = 1;

    fprintf(stderr, "%s: config raw=%s staging=%s %s\n", __func__,
            getenv("KPC_SEQ_RAW") ? getenv("KPC_SEQ_RAW") : "0",
            getenv("KPC_STAGING_SLOTS") ? getenv("KPC_STAGING_SLOTS") : "(default)",
            nonunified ? "NON-UNIFIED" : "unified");

    std::vector<uint8_t> seqstate;
    std::vector<float>   l0_base, l1_base;
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

        const size_t sz = llama_state_seq_get_size(ctx, 1);
        fprintf(stderr, "%s: per-seq state size = %zu bytes\n", __func__, sz);
        seqstate.resize(sz);
        const size_t got = llama_state_seq_get_data(ctx, seqstate.data(), seqstate.size(), 1);
        if (got != sz || sz == 0) {
            fprintf(stderr, "%s: FAIL - per-seq save returned %zu of %zu bytes\n", __func__, got, sz);
            return 1;
        }
        l0_base = decode_one(ctx, tok, P, 0, n_vocab);
        l1_base = decode_one(ctx, tok, P, 1, n_vocab);
    }

    const float eps = 1e-2f;
    int fails = 0;

    // (1) restore into a FRESH context
    {
        std::vector<float> l1 = restore_and_decode(model, cparams, seqstate, n_vocab, P, tok, false, nullptr);
        const float d = l1.empty() ? 1e9f : max_abs_diff(l1_base, l1);
        fprintf(stderr, "%s: [fresh-ctx]  seq 1 max|diff| = %.6f\n", __func__, d);
        if (d >= eps) { fprintf(stderr, "%s: FAIL fresh-ctx restore\n", __func__); ++fails; }
    }

    // (2) restore into a LIVE context (seq 0 already decoded -> seq 1 lands in non-empty, non-aligned cells)
    {
        std::vector<float> l0_live;
        std::vector<float> l1 = restore_and_decode(model, cparams, seqstate, n_vocab, P, tok, true, &l0_live);
        const float d1 = l1.empty()      ? 1e9f : max_abs_diff(l1_base, l1);
        const float d0 = l0_live.empty() ? 1e9f : max_abs_diff(l0_base, l0_live);
        fprintf(stderr, "%s: [live-ctx]   seq 1 max|diff| = %.6f, seq 0 (undisturbed) max|diff| = %.6f\n", __func__, d1, d0);
        if (d1 >= eps) { fprintf(stderr, "%s: FAIL live-ctx restore (seq 1)\n", __func__); ++fails; }
        if (d0 >= eps) { fprintf(stderr, "%s: FAIL live-ctx restore corrupted seq 0\n", __func__); ++fails; }
    }

    if (fails) {
        return 1;
    }
    fprintf(stderr, "%s: PASS - KPC per-sequence state round-trip (fresh + live context)\n", __func__);
    return 0;
}
