// kpc4_1 context-shift contract: the K-shift pass must rebuild group_index from post-shift
// positions, so shift+continue matches an independent fresh context that encoded the
// surviving positions directly; only requant noise separates the runs

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
    params.kv_unified      = true;
    params.n_parallel      = 1;                             // n_seq_max = 1
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
    const auto          cparams = common_context_params_to_llama(params);

    const int         P       = 64;   // prompt length (positions 0..63)
    const int         n_keep  = 8;
    const int         discard = 24;   // completion-style shift: drop [8, 32), slide [32, 64) down by 24
    const int         STEPS   = 6;
    const llama_token tok     = 1;

    // run A: fill to P, shift, then continue decoding at the shifted head (positions 40..45)
    std::vector<std::vector<float>> la(STEPS);
    {
        llama_context * ctx = init->context();
        if (!write_seq(ctx, tok, P, 0)) {
            fprintf(stderr, "%s: shift-run fill failed\n", __func__);
            return 1;
        }
        llama_memory_t mem = llama_get_memory(ctx);
        if (!llama_memory_can_shift(mem)) {
            fprintf(stderr, "%s: FAIL - KPC memory reports can_shift == false\n", __func__);
            return 1;
        }
        llama_memory_seq_rm (mem, 0, n_keep,           n_keep + discard);
        llama_memory_seq_add(mem, 0, n_keep + discard, P, -discard);
        for (int i = 0; i < STEPS; ++i) {
            la[i] = decode_one(ctx, tok, P - discard + i, 0, n_vocab);
        }
    }

    // run B (independent reference): the post-shift cache is semantically 8 kept + 32 surviving tokens
    // now at positions 8..39, which a fresh context encodes directly; continue with the same 6 tokens
    std::vector<std::vector<float>> lb(STEPS);
    {
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr || !write_seq(ctx, tok, P - discard, 0)) {
            fprintf(stderr, "%s: reference fill failed\n", __func__);
            if (ctx) llama_free(ctx);
            return 1;
        }
        for (int i = 0; i < STEPS; ++i) {
            lb[i] = decode_one(ctx, tok, P - discard + i, 0, n_vocab);
        }
        llama_free(ctx);
    }

    int fails = 0;
    for (int i = 0; i < STEPS; ++i) {
        if (!finite_vec(la[i]) || !finite_vec(lb[i])) {
            fprintf(stderr, "%s: step %d FAIL - non-finite logits\n", __func__, i);
            ++fails;
            continue;
        }
        const int   aa = argmax(la[i]);
        const int   ab = argmax(lb[i]);
        const float d  = max_abs_diff(la[i], lb[i]);
        // argmax equality is the bar; tolerance fallback for borderline ties on the tiny fixture
        const bool  ok = aa == ab || d < 0.05f;
        fprintf(stderr, "%s: step %d pos %d argmax shift=%d ref=%d max|diff|=%.4f -> %s\n",
                __func__, i, P - discard + i, aa, ab, d, !ok ? "FAIL" : (aa == ab ? "ok (argmax)" : "ok (diff)"));
        if (!ok) {
            ++fails;
        }
    }

    if (fails) {
        fprintf(stderr, "%s: FAIL - shift+continue diverges from the independent reference (%d/%d steps)\n", __func__, fails, STEPS);
        return 1;
    }
    fprintf(stderr, "%s: PASS - shift+continue matches an independently encoded reference for %d steps\n", __func__, STEPS);
    return 0;
}
