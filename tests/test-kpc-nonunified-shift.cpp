// kpc4_1 context-shift (RoPE K-shift) on a non-unified cache (n_stream > 1):
// the shifted stream must match the trusted unified-cache shift, and a co-resident
// unshifted stream must survive the requant pass

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "kpc-test-utils.h"

#include <vector>
#include <cstdio>
#include <cmath>

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed   = 1234;
    params.kv_unified      = false;                         // NON-UNIFIED -> n_stream == n_seq_max
    params.n_parallel      = 2;                             // n_seq_max = 2 -> n_stream = 2
    params.n_ctx           = 256;
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
    const auto          base    = common_context_params_to_llama(params);

    const int         P       = 64;
    const int         n_keep  = 8;
    const int         discard = 24;   // shift sequence 0 down by this many positions
    const llama_token tok     = 1;

    auto fill = [&](llama_context * ctx, bool with_seq1) -> bool {
        const int ns = with_seq1 ? 2 : 1;
        llama_batch batch = llama_batch_init(ns * P, 0, 1);
        for (int i = 0; i < P; ++i) {
            for (int s = 0; s < ns; ++s) {
                common_batch_add(batch, tok, i, { s }, false);
            }
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        return ok;
    };

    // apply the completion-tool context shift to sequence 0: drop [n_keep, n_keep+discard), slide the rest down
    auto shift_seq0 = [&](llama_context * ctx) {
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_seq_rm (mem, 0, n_keep,           n_keep + discard);
        llama_memory_seq_add(mem, 0, n_keep + discard, P, -discard);
    };

    // trusted reference: UNIFIED cache (n_stream==1), single sequence, same fill+shift. seq 0's first post-shift
    // prediction here is the ground truth the non-unified (n_stream==2) run must reproduce.
    int unif_tok0 = -1;
    {
        auto cp = base; cp.kv_unified = true; cp.n_seq_max = 1;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (ctx == nullptr || !fill(ctx, false)) {
            fprintf(stderr, "%s: unified reference fill failed\n", __func__);
            return 1;
        }
        shift_seq0(ctx);
        std::vector<float> lg = decode_one(ctx, tok, P - discard, 0, n_vocab);
        llama_free(ctx);
        if (!finite_vec(lg)) {
            fprintf(stderr, "%s: unified reference logits not finite\n", __func__);
            return 1;
        }
        unif_tok0 = argmax(lg);
    }

    // no-shift reference for the co-resident stream: NON-UNIFIED, fill both seqs, record seq 1's prediction at P.
    int ref1_tok = -1;
    std::vector<float> ref1;
    {
        auto cp = base; cp.kv_unified = false; cp.n_seq_max = 2;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (ctx == nullptr || !fill(ctx, true)) {
            fprintf(stderr, "%s: seq1 reference fill failed\n", __func__);
            return 1;
        }
        ref1 = decode_one(ctx, tok, P, 1, n_vocab);
        llama_free(ctx);
        if (!finite_vec(ref1)) {
            fprintf(stderr, "%s: seq1 reference logits not finite\n", __func__);
            return 1;
        }
        ref1_tok = argmax(ref1);
    }

    // the test: NON-UNIFIED (n_stream==2). Fill both seqs, shift seq 0 (multi-stream shift graph runs over both
    // streams), then probe seq 0 (the shifted stream) and seq 1 (the co-resident, requant-only stream).
    auto cp = base; cp.kv_unified = false; cp.n_seq_max = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr || !fill(ctx, true)) {
        fprintf(stderr, "%s: shift fill failed\n", __func__);
        return 1;
    }
    shift_seq0(ctx);

    // sequence 0 continues at its shifted head -> this decode forces build_rope_shift over n_stream=2.
    std::vector<float> s0 = decode_one(ctx, tok, P - discard, 0, n_vocab);
    if (!finite_vec(s0)) {
        fprintf(stderr, "%s: FAIL - seq 0 post-shift logits not finite\n", __func__);
        llama_free(ctx);
        return 1;
    }
    const int nonunif_tok0 = argmax(s0);

    bool ok = true;
    for (int step = 1; step < 6 && ok; ++step) {
        std::vector<float> lg = decode_one(ctx, tok, P - discard + step, 0, n_vocab);
        ok = finite_vec(lg);
    }

    // sequence 1 was NOT shifted, but the shift graph requantized its K (zero rope) -> prediction must still match
    // the no-shift reference (argmax identical; near-lossless requant keeps logits close).
    std::vector<float> shf1 = decode_one(ctx, tok, P, 1, n_vocab);
    llama_free(ctx);

    if (!ok) {
        fprintf(stderr, "%s: FAIL - a post-shift seq 0 decode produced non-finite logits\n", __func__);
        return 1;
    }
    if (!finite_vec(shf1)) {
        fprintf(stderr, "%s: FAIL - seq 1 post-shift logits not finite\n", __func__);
        return 1;
    }
    const int shf1_tok = argmax(shf1);

    float seq1_maxdiff = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        seq1_maxdiff = std::max(seq1_maxdiff, std::fabs(shf1[i] - ref1[i]));
    }

    fprintf(stderr, "%s: seq 0 post-shift argmax: unified=%d non-unified=%d\n", __func__, unif_tok0, nonunif_tok0);
    fprintf(stderr, "%s: seq 1 argmax ref=%d shifted=%d, logit max|diff|=%.4f\n", __func__, ref1_tok, shf1_tok, seq1_maxdiff);

    if (nonunif_tok0 != unif_tok0) {
        fprintf(stderr, "%s: FAIL - non-unified shifted-stream prediction differs from the trusted unified shift\n", __func__);
        return 1;
    }
    if (shf1_tok != ref1_tok) {
        fprintf(stderr, "%s: FAIL - co-resident seq 1 prediction changed under seq 0's shift (cross-stream corruption)\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s: PASS - non-unified KPC context-shift (n_stream=2) matches the unified shift; co-resident stream intact\n", __func__);
    return 0;
}
