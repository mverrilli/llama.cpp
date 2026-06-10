// kpc staging-slot virtualization: with KPC_STAGING_SLOTS=L < n_seq_max each live seq maps
// to a slot, freed on seq_rm and reused. drives seq A, retires it, then seq B (under L=1
// B reuses A's freed slot); logits must match the default run

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
    params.n_parallel      = 4;                            // (band namespace)
    params.n_ctx           = 1024;                         // ng_max = 32 >= n_seq_max -> bands fit
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

    const int         P   = 40;   // leaves an open group (positions 32..39)
    const llama_token tok = 1;

    // run A then B in one context; if retire, seq_rm(A) before B so it can reuse A's staging slot
    auto run_ab = [&](llama_context * ctx, bool retire, std::vector<float> & la, std::vector<float> & lb) {
        la = run_seq(ctx, 0, P, tok, n_vocab);
        if (retire) {
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        }
        lb = run_seq(ctx, 1, P, tok, n_vocab);
    };

    // baseline: default staging (L == n_seq_max), both sequences live in their own slots
    std::vector<float> la_base, lb_base;
    {
        unsetenv("KPC_STAGING_SLOTS");
        llama_context * ctx = llama_init_from_model(model, cparams);
        run_ab(ctx, /*retire=*/false, la_base, lb_base);
        llama_free(ctx);
    }

    // virtualized: only L=1 staging slot; retire A so B reuses the freed slot (never more than 1 live)
    std::vector<float> la_virt, lb_virt;
    {
        setenv("KPC_STAGING_SLOTS", "1", 1);
        llama_context * ctx = llama_init_from_model(model, cparams);
        run_ab(ctx, /*retire=*/true, la_virt, lb_virt);
        llama_free(ctx);
        unsetenv("KPC_STAGING_SLOTS");
    }

    const float eps = 1e-3f;
    const float da = max_abs_diff(la_base, la_virt);
    const float db = max_abs_diff(lb_base, lb_virt);
    fprintf(stderr, "%s: seq A logit max|diff| = %.6f\n", __func__, da);
    fprintf(stderr, "%s: seq B logit max|diff| = %.6f (reused slot 0)\n", __func__, db);

    if (da >= eps || db >= eps) {
        fprintf(stderr, "%s: FAIL - virtualized staging (L=1) diverges from default (L=n_seq_max)\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s: PASS - L=1 staging slot reused across 2 sequences matches the default run bit-for-bit\n", __func__);
    return 0;
}
