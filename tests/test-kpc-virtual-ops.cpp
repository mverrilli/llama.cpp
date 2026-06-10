// kpc4_1 virtualized-staging (KPC_STAGING_SLOTS < n_seq_max) tests: seq_cp, whole-cache save/restore, slot spill
// each sets KPC_STAGING_SLOTS before creating the context and validates against a reference
// plus: default-staging fork-then-reuse rescue, -fa off multi-stream, and pool-band validation

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "kpc-test-utils.h"

#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>

static llama_context_params mk(const common_params & params, uint32_t n_seq_max, const char * staging_slots) {
    auto cp = common_context_params_to_llama(params);
    cp.kv_unified  = true;
    cp.n_seq_max   = n_seq_max;
    setenv("KPC_STAGING_SLOTS", staging_slots, 1);
    return cp;
}

// 1. seq_cp under virtualized staging (L=3 < n_seq_max=4): copy seq 0 -> seq 1, predictions must agree
static int test_seqcp(llama_model * model, const common_params & params, int n_vocab) {
    const auto cp = mk(params, 4, "3");
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "seqcp: ctx create failed\n"); return 1; }

    const int P = 32; const llama_token tok = 1;   // 32 = full group -> clean boundary, copied prefix finalized
    llama_batch b = llama_batch_init(P, 0, 1);
    for (int i = 0; i < P; ++i) common_batch_add(b, tok, i, { 0 }, false);
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    if (!ok) { fprintf(stderr, "seqcp: prompt decode failed\n"); llama_free(ctx); return 1; }

    llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, -1, -1);   // copy seq 0 -> seq 1

    const auto l0 = decode_one(ctx, tok, P, 0, n_vocab);
    const auto l1 = decode_one(ctx, tok, P, 1, n_vocab);
    llama_free(ctx);

    if (!finite_vec(l0) || !finite_vec(l1)) { fprintf(stderr, "seqcp: FAIL non-finite logits\n"); return 1; }
    const float d = max_abs_diff(l0, l1);
    fprintf(stderr, "seqcp: argmax src=%d dst=%d, logit max|diff|=%.4f\n", argmax(l0), argmax(l1), d);
    if (argmax(l0) != argmax(l1) || d > 1e-2f) {
        fprintf(stderr, "seqcp: FAIL - copied sequence diverges from source under virtualized staging\n");
        return 1;
    }
    fprintf(stderr, "seqcp: PASS\n");
    return 0;
}

// 2. save/restore under virtualized staging (L=2 < n_seq_max=3): snapshot two open-group seqs, restore, logits match
static int test_state(llama_model * model, const common_params & params, int n_vocab) {
    const auto cp = mk(params, 3, "2");
    const int P = 40; const llama_token tok = 1;   // 40 -> open group (members 32..39 present)

    auto fill = [&](llama_context * ctx) -> bool {
        for (int s = 0; s < 2; ++s) {
            llama_batch b = llama_batch_init(P, 0, 1);
            for (int i = 0; i < P; ++i) common_batch_add(b, tok, i, { s }, false);
            const bool ok = llama_decode(ctx, b) == 0;
            llama_batch_free(b);
            if (!ok) return false;
        }
        return true;
    };

    std::vector<uint8_t> state;
    std::vector<float> base0, base1;
    {
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx || !fill(ctx)) { fprintf(stderr, "state: baseline fill failed\n"); if (ctx) llama_free(ctx); return 1; }
        state.resize(llama_state_get_size(ctx));
        llama_state_get_data(ctx, state.data(), state.size());
        base0 = decode_one(ctx, tok, P, 0, n_vocab);
        base1 = decode_one(ctx, tok, P, 1, n_vocab);
        llama_free(ctx);
    }

    std::vector<float> rest0, rest1;
    {
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "state: restore ctx failed\n"); return 1; }
        const size_t nset = llama_state_set_data(ctx, state.data(), state.size());
        if (nset != state.size()) {
            fprintf(stderr, "state: FAIL - restore consumed %zu of %zu bytes\n", nset, state.size());
            llama_free(ctx); return 1;
        }
        rest0 = decode_one(ctx, tok, P, 0, n_vocab);
        rest1 = decode_one(ctx, tok, P, 1, n_vocab);
        llama_free(ctx);
    }

    const float d0 = max_abs_diff(base0, rest0), d1 = max_abs_diff(base1, rest1);
    fprintf(stderr, "state: seq0 max|diff|=%.6f  seq1 max|diff|=%.6f\n", d0, d1);
    if (d0 >= 1e-3f || d1 >= 1e-3f) {
        fprintf(stderr, "state: FAIL - virtualized-staging round-trip diverges from baseline\n");
        return 1;
    }
    fprintf(stderr, "state: PASS\n");
    return 0;
}

// 3. slot spill: two live seqs in one ubatch, L=1 (none evictable) -> overflow seq spills gracefully. complete
//    groups are staging-independent, so the spilled cache matches an L=2 reference bit-for-bit
static int test_spill(llama_model * model, const common_params & params, int n_vocab) {
    const int K = 32; const llama_token tok = 1;  // one COMPLETE group per seq, both written in one ubatch

    auto run = [&](const char * slots, std::vector<float> & out1) -> bool {
        const auto cp = mk(params, 2, slots);
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) return false;
        llama_batch b = llama_batch_init(2 * K, 0, 1);
        for (int i = 0; i < K; ++i) {
            for (int s = 0; s < 2; ++s) common_batch_add(b, tok, i, { s }, false);
        }
        const bool ok = llama_decode(ctx, b) == 0;
        llama_batch_free(b);
        if (ok) out1 = decode_one(ctx, tok, K, 1, n_vocab);
        llama_free(ctx);
        return ok;
    };

    std::vector<float> ref1, spl1;
    if (!run("2", ref1)) { fprintf(stderr, "spill: reference (L=2) decode failed\n"); return 1; }
    if (!run("1", spl1)) { fprintf(stderr, "spill: FAIL - L=1 decode aborted (spill not handled)\n"); return 1; }

    if (!finite_vec(ref1) || !finite_vec(spl1)) { fprintf(stderr, "spill: FAIL non-finite logits\n"); return 1; }
    const float d = max_abs_diff(ref1, spl1);
    fprintf(stderr, "spill: argmax ref=%d spilled=%d, logit max|diff|=%.6f\n", argmax(ref1), argmax(spl1), d);
    if (argmax(ref1) != argmax(spl1) || d >= 1e-3f) {
        fprintf(stderr, "spill: FAIL - spilled cache differs from staged reference\n");
        return 1;
    }
    fprintf(stderr, "spill: PASS\n");
    return 0;
}

// 4. fork-then-reuse rescue (default staging): after seq_cp(0->1) and seq_rm(0), reusing seq 0
//    re-encodes pools the copy still references; kpc_write must requantize those live cells
static int test_fork_reuse(llama_model * model, const common_params & params, int n_vocab) {
    unsetenv("KPC_STAGING_SLOTS");   // default layout: slot == seq
    auto cp = common_context_params_to_llama(params);
    cp.kv_unified = true;
    cp.n_seq_max  = 2;
    cp.n_ctx      = 512;

    const llama_token tok = 1, tok2 = 2, tok3 = 3;
    const int P = 64;   // two complete groups on seq 0

    auto fill_and_fork = [&](llama_context * ctx) -> bool {
        llama_batch b = llama_batch_init(P, 0, 1);
        for (int i = 0; i < P; ++i) common_batch_add(b, tok, i, { 0 }, false);
        const bool ok = llama_decode(ctx, b) == 0;
        llama_batch_free(b);
        if (!ok) return false;
        llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, -1, -1);
        return true;
    };

    // control: the ref capture writes one probe token into seq 1; verify probe -> seq_rm -> re-probe is
    // itself stable before relying on it for the corruption check
    bool pattern_stable = false;
    {
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx || !fill_and_fork(ctx)) { fprintf(stderr, "fork: control setup failed\n"); if (ctx) llama_free(ctx); return 1; }
        const auto ref = decode_one(ctx, tok2, P, 1, n_vocab);
        llama_memory_seq_rm(llama_get_memory(ctx), 1, P, P + 1);
        const auto re  = decode_one(ctx, tok2, P, 1, n_vocab);
        llama_free(ctx);
        if (!finite_vec(ref) || !finite_vec(re)) { fprintf(stderr, "fork: control probe non-finite\n"); return 1; }
        const float dc = max_abs_diff(ref, re);
        pattern_stable = dc < 1e-6f;
        fprintf(stderr, "fork: control probe/rm/reprobe max|diff|=%.6f -> %s\n",
                dc, pattern_stable ? "in-context ref" : "separate-context ref");
    }

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx || !fill_and_fork(ctx)) { fprintf(stderr, "fork: setup failed\n"); if (ctx) llama_free(ctx); return 1; }

    std::vector<float> ref;
    if (pattern_stable) {
        ref = decode_one(ctx, tok2, P, 1, n_vocab);
        llama_memory_seq_rm(llama_get_memory(ctx), 1, P, P + 1);   // remove the probe token again
    } else {
        // the probe pattern perturbs logits; take the reference from a separate identical context
        llama_context * rctx = llama_init_from_model(model, cp);
        if (!rctx || !fill_and_fork(rctx)) { fprintf(stderr, "fork: ref ctx setup failed\n"); if (rctx) llama_free(rctx); llama_free(ctx); return 1; }
        ref = decode_one(rctx, tok2, P, 1, n_vocab);
        llama_free(rctx);
    }
    if (!finite_vec(ref)) { fprintf(stderr, "fork: ref probe non-finite\n"); llama_free(ctx); return 1; }

    // retire the source and reuse its seq id: 33 tokens re-encode band-0 groups 0 and 1
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    {
        llama_batch b = llama_batch_init(33, 0, 1);
        for (int i = 0; i < 33; ++i) common_batch_add(b, tok3, i, { 0 }, false);
        const bool ok = llama_decode(ctx, b) == 0;
        llama_batch_free(b);
        if (!ok) { fprintf(stderr, "fork: reused-seq decode failed\n"); llama_free(ctx); return 1; }
    }

    const auto post = decode_one(ctx, tok2, P, 1, n_vocab);
    llama_free(ctx);

    // fixture caveat: prefix sensitivity at this probe is ~3e-8, so this catches catastrophic
    // corruption (non-finite, argmax flips), not subtle requant drift
    if (!finite_vec(post)) { fprintf(stderr, "fork: FAIL non-finite post-reuse logits\n"); return 1; }
    const float d = max_abs_diff(ref, post);
    fprintf(stderr, "fork: argmax ref=%d post=%d, logit max|diff|=%.3e\n", argmax(ref), argmax(post), d);
    if (argmax(ref) != argmax(post) || d >= 1e-3f) {
        fprintf(stderr, "fork: FAIL - reusing the source seq corrupted the forked sequence's shared prefix\n");
        return 1;
    }
    fprintf(stderr, "fork: PASS\n");
    return 0;
}

// 5. -fa off with a multi-stream KPC cache: must create and decode (V stays f16; quantized V needs FA)
static int test_fa_off(llama_model * model, const common_params & params, int n_vocab) {
    unsetenv("KPC_STAGING_SLOTS");
    auto cp = common_context_params_to_llama(params);
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.kv_unified      = false;
    cp.n_seq_max       = 2;
    cp.type_v          = GGML_TYPE_F16;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "fa_off: FAIL - multi-stream KPC context with FA disabled was refused\n"); return 1; }

    bool ok = write_seq(ctx, 1, 8, 0) && write_seq(ctx, 1, 8, 1);
    std::vector<float> l0, l1;
    if (ok) {
        l0 = decode_one(ctx, 1, 8, 0, n_vocab);
        l1 = decode_one(ctx, 1, 8, 1, n_vocab);
    }
    llama_free(ctx);

    if (!ok || !finite_vec(l0) || !finite_vec(l1)) {
        fprintf(stderr, "fa_off: FAIL - decode failed or non-finite logits on the non-FA multi-stream path\n");
        return 1;
    }
    fprintf(stderr, "fa_off: PASS\n");
    return 0;
}

// 6. pool-band validation: kv_size/32 < n_seq_max cannot host one pool band per sequence -> clean refusal.
//    n_ctx pads to a multiple of 256 (llama-context), so kv_size=256 -> 8 pools: 8 seqs fit, 16 must refuse
static int test_band_validation(llama_model * model, const common_params & params) {
    unsetenv("KPC_STAGING_SLOTS");
    auto cp = common_context_params_to_llama(params);
    cp.kv_unified = true;
    cp.n_ctx      = 128;   // padded to 256 -> 8 pool bands

    cp.n_seq_max = 8;      // boundary: exactly one pool per seq -> must succeed
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "band: FAIL - boundary context (8 pools, 8 seqs) was refused\n");
        return 1;
    }
    llama_free(ctx);

    cp.n_seq_max = 16;     // 8 pools < 16 seqs -> clean refusal
    ctx = llama_init_from_model(model, cp);
    if (ctx) {
        fprintf(stderr, "band: FAIL - context created despite kv_size/32 < n_seq_max\n");
        llama_free(ctx);
        return 1;
    }
    fprintf(stderr, "band: PASS - undersized pool-band context refused cleanly (boundary 8/8 ok)\n");
    return 0;
}

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed   = 1234;
    params.n_ctx           = 256;
    common_init_result_ptr init = kpc_model_init(argc, argv, params, __func__);
    if (!init || !init->model()) return 1;
    if (init->context() == nullptr) return 0;
    llama_model * model = init->model();

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    int fails = 0;
    fails += test_seqcp(model, params, n_vocab);
    fails += test_state(model, params, n_vocab);
    fails += test_spill(model, params, n_vocab);
    fails += test_fork_reuse(model, params, n_vocab);
    fails += test_fa_off(model, params, n_vocab);
    fails += test_band_validation(model, params);

    if (fails) { fprintf(stderr, "test-kpc-virtual-ops: %d FAIL(s)\n", fails); return 1; }
    fprintf(stderr, "test-kpc-virtual-ops: all passed\n");
    return 0;
}
