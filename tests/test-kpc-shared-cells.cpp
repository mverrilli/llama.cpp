// kpc4_1 under the ctx_other shared-cells path (the server MTP/draft mechanism: a second context
// shares the target's cell metadata). A shared context must decode without crashing and must not
// corrupt the target's KPC cache. Note: layer-TENSOR sharing only happens for GEMMA4_ASSISTANT;
// for a plain model ctx_other shares only cell metadata, which is what this exercises.

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
    params.kv_unified    = true;
    params.n_parallel    = 1;
    params.n_ctx         = 256;

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

    const int         P   = 40;   // prompt leaves an open group (members 32..39)
    const llama_token tok = 1;

    llama_context * ctx_tgt = init->context();

    // fill the target's KPC cache with a prompt, snapshot its next-token logits, restore to [0,P)
    if (!write_seq(ctx_tgt, tok, P, 0)) {
        fprintf(stderr, "%s: prompt write failed\n", __func__);
        return 1;
    }
    std::vector<float> l_ref = decode_one(ctx_tgt, tok, P, 0, n_vocab);
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, P, -1);
    if (!finite_vec(l_ref)) {
        fprintf(stderr, "%s: target reference logits not finite\n", __func__);
        return 1;
    }

    // create a second context sharing the target's cells (the ctx_other / MTP-draft path)
    llama_context_params cp_sh = cparams;
    cp_sh.ctx_other = ctx_tgt;
    llama_context * ctx_sh = llama_init_from_model(model, cp_sh);
    if (ctx_sh == nullptr) {
        // a clean refusal is an acceptable, documented outcome for kpc + shared cells
        fprintf(stderr, "%s: ctx_other + kpc4_1 context refused/failed to create (clean refusal OK)\n", __func__);
        return 0;
    }

    // decode on the shared context: reads the shared cell metadata, writes its own KPC tensors.
    // must not nan / assert / crash even though its K for the shared cells is its own.
    std::vector<float> l_sh = decode_one(ctx_sh, tok, P, 0, n_vocab);
    const bool sh_ok = finite_vec(l_sh);
    llama_free(ctx_sh);

    // remove whatever cell the shared context added to the shared metadata, restoring [0,P)
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, P, -1);

    // the target's KPC cache must be unchanged by the shared context's activity
    std::vector<float> l_ref2 = decode_one(ctx_tgt, tok, P, 0, n_vocab);
    const float corrupt = max_abs_diff(l_ref, l_ref2);

    fprintf(stderr, "%s: shared-ctx logits finite=%d ; target corruption max|diff|=%.6f\n",
            __func__, (int) sh_ok, corrupt);

    if (!sh_ok) {
        fprintf(stderr, "%s: FAIL - shared-context decode produced non-finite logits\n", __func__);
        return 1;
    }
    if (corrupt >= 1e-3f) {
        fprintf(stderr, "%s: FAIL - target KPC cache corrupted by the shared context (max|diff|=%.6f)\n", __func__, corrupt);
        return 1;
    }
    fprintf(stderr, "%s: PASS - kpc4_1 survives the ctx_other shared-cells path (no crash, target intact)\n", __func__);
    return 0;
}
