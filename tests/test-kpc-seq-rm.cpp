// kpc4_1 survivor-rescue end-to-end via the real llama_memory_seq_rm + continue path (the op-level test in
// test-kpc.cpp fills the survivor list by hand; this drives set_input_kpc's real gid==pool fill). Decode [0,N),
// seq_rm a mid-group p0 so the sealed group keeps survivors [0,p0), then re-decode so the re-seal rescues them,
// and require the rewind to reproduce the base logits. Backend via -ngl (none=CPU, 99=CUDA via ctest args).
// On this tiny model the rescue's logit delta is ~ULP, so this is a path/liveness guard -- the numerical proof
// is test_rescue_survivors (cache NMSE); KPC_TEST_VARYTOK + KPC_TEST_DUMP feed the real-model on/off check.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "kpc-test-utils.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// decode positions [lo,hi) of `seq`, logits only on the last token; return its logits. vary=true gives each
// position a distinct token (token = pos mod (n_vocab-4) + 2) so survivor K differs from the re-decoded members'
// -- needed on a real model to make the members-only slab a poor fit for the survivors (isolates the rescue).
static std::vector<float> decode_range_last(llama_context * ctx, llama_seq_id seq, int lo, int hi,
                                            llama_token tok, int n_vocab, bool vary) {
    const int n = hi - lo;
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        const llama_token t = vary ? (llama_token)(((lo + i) % (n_vocab > 8 ? n_vocab - 4 : 1)) + 2) : tok;
        common_batch_add(b, t, lo + i, { seq }, /*logits=*/ i == n - 1);
    }
    std::vector<float> out;
    if (llama_decode(ctx, b) == 0) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (lg) {
            out.assign(lg, lg + n_vocab);
        }
    }
    llama_batch_free(b);
    return out;
}

int main(int argc, char ** argv) {
    // scenario params are env-overridable so the SAME test drives the real-model numerical end-to-end: pass
    // -m <real.gguf> (real magnitudes/sharper attention make a rescue-off control a visible logit/PPL spike) and
    // tune these. Defaults match the synthetic fixture (group 0 = pos 0..31).
    auto envi = [](const char * k, int d) { const char * v = getenv(k); return v ? atoi(v) : d; };
    const int         N      = envi("KPC_TEST_N", 48);        // seals group 0 (0..31), opens group 1 (32..47)
    const int         p0     = envi("KPC_TEST_P0", 20);       // mid the SEALED group 0; survivors [0,p0)
    const int         psplit = envi("KPC_TEST_PSPLIT", 4);    // re-open group 0 before the re-seal batch
    const llama_token tok    = (llama_token) envi("KPC_TEST_TOK", 1);
    const float       eps    = getenv("KPC_TEST_EPS") ? (float) atof(getenv("KPC_TEST_EPS")) : 1e-2f;

    common_params params;
    params.sampling.seed = 1234;
    params.n_ctx         = (uint32_t) envi("KPC_TEST_NCTX", (N + 16) * 2 < 128 ? 128 : (N + 16) * 2);
    params.n_parallel    = 1;

    common_init_result_ptr init = kpc_model_init(argc, argv, params, __func__);
    if (!init || init->model() == nullptr) {
        return 1;
    }
    if (init->context() == nullptr) {
        return 0;   // context unavailable (head_dim guard) -> skip cleanly
    }

    llama_model       * model   = init->model();
    llama_context     * ctx     = init->context();
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    const bool gpu  = params.n_gpu_layers > 0;
    const bool vary = getenv("KPC_TEST_VARYTOK") != nullptr;   // distinct token per pos (real-model discrimination)
    fprintf(stderr, "%s: backend=%s n_ctx=%d N=%d p0=%d psplit=%d tok=%d vary=%d eps=%.4g KPC_GROUP=32\n",
            __func__, gpu ? "CUDA(-ngl)" : "CPU", (int) params.n_ctx, N, p0, psplit, (int) tok, (int) vary, eps);

    // (1) base: decode the whole prompt [0,N) in one batch, capture next-token logits.
    std::vector<float> L_base = decode_range_last(ctx, /*seq=*/0, 0, N, tok, n_vocab, vary);
    if (L_base.empty()) {
        fprintf(stderr, "%s: FAIL - base decode failed\n", __func__);
        return 1;
    }

    // (2) rewind: real seq_rm removes positions [p0, +inf) of seq 0 (survivors = sealed cells [0,p0)).
    llama_memory_t mem = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(mem, /*seq_id=*/0, /*p0=*/p0, /*p1=*/-1)) {
        fprintf(stderr, "%s: FAIL - llama_memory_seq_rm(0, %d, -1) returned false\n", __func__, p0);
        return 1;
    }

    // (3) re-decode [p0,N) in TWO batches: the first re-opens group 0's pool, the second re-seals it while the
    // survivors [0,p0) are still committed -- so set_input_kpc's gid==pool fill hands the survivor list to the
    // kernel. A single batch would re-open and re-seal in one ubatch, leaving set_input_kpc nothing to rescue.
    {
        std::vector<float> warm = decode_range_last(ctx, /*seq=*/0, p0, p0 + psplit, tok, n_vocab, vary);
        if (warm.empty()) {
            fprintf(stderr, "%s: FAIL - rewind re-open batch failed\n", __func__);
            return 1;
        }
    }
    std::vector<float> L_rewind = decode_range_last(ctx, /*seq=*/0, p0 + psplit, N, tok, n_vocab, vary);
    if (L_rewind.empty()) {
        fprintf(stderr, "%s: FAIL - rewind re-seal batch failed\n", __func__);
        return 1;
    }
    if (const char * dump = getenv("KPC_TEST_DUMP")) {   // raw L_rewind -> compare rescue-ON vs rescue-OFF externally
        if (FILE * f = fopen(dump, "wb")) { fwrite(L_rewind.data(), sizeof(float), L_rewind.size(), f); fclose(f); }
    }

    // (4) compare.
    const bool  finite = finite_vec(L_base) && finite_vec(L_rewind);
    const float diff   = max_abs_diff(L_base, L_rewind);
    fprintf(stderr, "%s: finite=%d max|L_base - L_rewind| = %.6f (eps=%.4f)\n",
            __func__, (int) finite, diff, eps);

    if (!finite || diff >= eps) {
        fprintf(stderr, "%s: FAIL - rewind not reproducible (%s)\n",
                __func__, !finite ? "non-finite logits" : "diff >= eps");
        return 1;
    }
    fprintf(stderr, "%s: PASS - KPC seq_rm rewind reproducible (survivors rescued), diff=%.6f\n", __func__, diff);
    return 0;
}
