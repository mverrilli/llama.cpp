// kpc4_1 speculative-decode access-pattern guard. Reproduces the spec-decode corruption found reproducing
// discussion #24518 on GPU: repeated multi-token-batch write + partial seq_rm rollback of the open group, with
// the rolled-back (rejected draft) tokens differing from the kept (accepted) token -- the pattern llama-speculative
// drives but test-kpc-seq-rm.cpp does not (it re-decodes identical tokens, one rollback cycle).
//
// Reference = clean one-token-per-step autoregressive decode of the SAME accepted sequence. The cache content for
// positions [0,M) must be byte-identical regardless of how it was built, so the probe logits at M must match.
//
// NOTE on sensitivity: like test-kpc-seq-rm, on the tiny head_dim=128 fixture the logit delta is ~0 (its K ranges
// are small enough that quantization is near-lossless), so the ctest run is a PATH/LIVENESS guard. The numerical
// proof is a real model: on qwen2.5-1.5b q8_0, this exact harness (M=40 D=15) read max|dL|=16.9 (argmax flip,
// collapse) before the set_input_kpc open-group fix and 1.7 (argmax preserved, ~= the f16 FP-reordering floor of
// 0.27) after -- run `test-kpc-specdecode -m <real.gguf> -ngl 99` with KPC_SPEC_* to reproduce. KPC_FORCE_F16=1
// gives the no-quantization control (isolates the bug from harness/FP-ordering noise).
//
// Modes (KPC_SPEC_MODE):
//   spec   (default): each step decode [accepted@p, D junk drafts@p+1..p+D], then seq_rm(p+1,-1). Advance p.
//   writeA          : open a group mid-way (clean [0,base)), then one D-token batch [base,base+D) -- multi-token
//                     write into a partially-filled group, NO rollback. Isolates write-path from rollback.
// Run: test-kpc-specdecode -m <model.gguf> [-ngl 99]   (CPU vs CUDA via -ngl)

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "kpc-test-utils.h"

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// distinct token per position so rejected-draft K differs from accepted K (else stale staging matches by value).
static llama_token acc_tok(int pos, int n_vocab) { return (llama_token)((pos * 7 + 3) % (n_vocab > 8 ? n_vocab - 4 : 1) + 2); }
static llama_token drf_tok(int pos, int n_vocab) { return (llama_token)((pos * 13 + 101) % (n_vocab > 8 ? n_vocab - 4 : 1) + 2); }

static std::vector<float> probe(llama_context * ctx, llama_seq_id seq, int pos, llama_token tok, int n_vocab) {
    return decode_one(ctx, tok, pos, seq, n_vocab);
}

int main(int argc, char ** argv) {
    auto envi = [](const char * k, int d) { const char * v = getenv(k); return v ? atoi(v) : d; };
    const int   M    = envi("KPC_SPEC_M", 40);     // accepted-sequence length
    const int   D    = envi("KPC_SPEC_D", 15);     // draft tokens proposed per step (rolled back in spec mode)
    const int   base = envi("KPC_SPEC_BASE", 8);   // writeA: clean prefix length (opens a mid-group)
    const std::string mode = getenv("KPC_SPEC_MODE") ? getenv("KPC_SPEC_MODE") : "spec";
    const float eps  = getenv("KPC_SPEC_EPS") ? (float) atof(getenv("KPC_SPEC_EPS")) : 1e-2f;

    common_params params;
    params.sampling.seed = 1234;
    params.n_ctx         = (uint32_t) std::max(256, (M + D + 8) * 2);
    params.n_parallel    = 1;

    common_init_result_ptr init = kpc_model_init(argc, argv, params, __func__);
    if (!init || init->model() == nullptr) return 1;
    if (init->context() == nullptr) return 0;     // head_dim guard -> skip

    llama_context     * ctx     = init->context();
    const llama_vocab * vocab   = llama_model_get_vocab(init->model());
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t      mem     = llama_get_memory(ctx);
    const bool          gpu     = params.n_gpu_layers > 0;
    fprintf(stderr, "%s: backend=%s mode=%s M=%d D=%d base=%d eps=%.4g\n",
            __func__, gpu ? "CUDA(-ngl)" : "CPU", mode.c_str(), M, D, base, eps);

    // (1) REFERENCE: clean one-token-per-step decode of accepted tokens [0,M), then probe at M.
    for (int p = 0; p < M; ++p) {
        llama_batch b = llama_batch_init(1, 0, 1);
        common_batch_add(b, acc_tok(p, n_vocab), p, { 0 }, false);
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) { fprintf(stderr, "%s: FAIL ref decode @%d\n", __func__, p); return 1; }
    }
    std::vector<float> L_ref = probe(ctx, 0, M, acc_tok(M, n_vocab), n_vocab);
    if (L_ref.empty()) { fprintf(stderr, "%s: FAIL ref probe\n", __func__); return 1; }
    if (const char * d = getenv("KPC_SPEC_DUMP_REF")) {   // dump clean-decode logits to verify backend (CPU vs CUDA L_ref must differ)
        if (FILE * f = fopen(d, "wb")) { fwrite(L_ref.data(), sizeof(float), L_ref.size(), f); fclose(f); }
    }

    // reset seq 0
    llama_memory_seq_rm(mem, 0, 0, -1);

    // (2) TEST: rebuild [0,M) via the chosen access pattern.
    if (mode == "writeA") {
        // clean prefix [0,base) one-per-step (opens the group containing `base`)
        for (int p = 0; p < base; ++p) {
            llama_batch b = llama_batch_init(1, 0, 1);
            common_batch_add(b, acc_tok(p, n_vocab), p, { 0 }, false);
            if (llama_decode(ctx, b)) { llama_batch_free(b); fprintf(stderr, "%s: FAIL writeA prefix\n", __func__); return 1; }
            llama_batch_free(b);
        }
        // one multi-token batch [base,M), NO rollback
        const int n = M - base;
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) common_batch_add(b, acc_tok(base + i, n_vocab), base + i, { 0 }, false);
        if (llama_decode(ctx, b)) { llama_batch_free(b); fprintf(stderr, "%s: FAIL writeA batch\n", __func__); return 1; }
        llama_batch_free(b);
    } else {
        // spec: per step, decode [accepted@p, D junk@p+1..p+D], then seq_rm(p+1,-1) to reject drafts.
        for (int p = 0; p < M; ++p) {
            const int n = 1 + D;
            llama_batch b = llama_batch_init(n, 0, 1);
            common_batch_add(b, acc_tok(p, n_vocab), p, { 0 }, false);            // accepted (kept)
            for (int j = 1; j <= D; ++j) common_batch_add(b, drf_tok(p + j, n_vocab), p + j, { 0 }, false); // drafts (rejected)
            const int rc = llama_decode(ctx, b);
            llama_batch_free(b);
            if (rc != 0) { fprintf(stderr, "%s: FAIL spec decode @%d\n", __func__, p); return 1; }
            llama_memory_seq_rm(mem, 0, p + 1, -1);   // reject the D drafts, keep [0,p]
        }
    }

    std::vector<float> L_test = probe(ctx, 0, M, acc_tok(M, n_vocab), n_vocab);
    if (L_test.empty()) { fprintf(stderr, "%s: FAIL test probe\n", __func__); return 1; }

    const bool  finite = finite_vec(L_ref) && finite_vec(L_test);
    const float diff   = max_abs_diff(L_ref, L_test);
    const int   amx_r  = argmax(L_ref), amx_t = argmax(L_test);
    fprintf(stderr, "%s: finite=%d max|L_ref - L_test| = %.6f (eps=%.4f)  argmax ref=%d test=%d\n",
            __func__, (int) finite, diff, eps, amx_r, amx_t);

    if (!finite || diff >= eps) {
        fprintf(stderr, "%s: FAIL - %s access pattern does not reproduce clean decode (%s)\n",
                __func__, mode.c_str(), !finite ? "non-finite" : "diff >= eps");
        return 1;
    }
    fprintf(stderr, "%s: PASS - %s reproduces clean decode, diff=%.6f\n", __func__, mode.c_str(), diff);
    return 0;
}
