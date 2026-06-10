// kpc4_1 default-staging (slot == seq): seq_rm/seq_keep/rewind/clear must retire open-group
// staging, else a reused seq id folds stale residuals into the new group's quantization.
// the corruption is latent on the tiny fixture, so assert via kpc_staged_mask(), not logits

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "kpc-test-utils.h"

#include "../src/llama-kv-cache.h"

#include <vector>
#include <cstdio>
#include <cstdlib>

// max staged_mask across all layers for a staging slot (0 == no open group staged anywhere)
static int32_t any_staged(const llama_kv_cache * kv, int n_layer, int32_t slot) {
    int32_t acc = 0;
    for (int il = 0; il < n_layer; ++il) {
        acc |= kv->kpc_staged_mask((uint32_t) il, slot);
    }
    return acc;
}

static int test_reuse(llama_model * model, const common_params & params) {
    unsetenv("KPC_STAGING_SLOTS");   // default layout: slot == seq (non-virtualized staging)

    auto cp = common_context_params_to_llama(params);
    cp.kv_unified = true;
    cp.n_seq_max  = 2;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "reuse: ctx create failed\n"); return 1; }

    llama_kv_cache * kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx));
    if (!kv) { fprintf(stderr, "reuse: memory is not a llama_kv_cache\n"); llama_free(ctx); return 1; }

    const int n_layer = llama_model_n_layer(model);

    // seq 0 ends mid-group: members 0..7 of group 0 staged (open group)
    if (!write_seq(ctx, /*tok*/ 1, /*n*/ 8, /*seq*/ 0)) {
        fprintf(stderr, "reuse: partial write failed\n"); llama_free(ctx); return 1;
    }

    // precondition: the partial write must actually leave an open group staged, else the test
    // exercises nothing (e.g. model not KPC, or group size != what we assume)
    const int32_t staged_before = any_staged(kv, n_layer, /*slot*/ 0);
    if (staged_before == 0) {
        fprintf(stderr, "reuse: SKIP - no open group staged after partial write (mask=0); not a KPC open-group setup\n");
        llama_free(ctx);
        return 0;
    }

    // full removal must retire the staging slot
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    const int32_t staged_after = any_staged(kv, n_layer, /*slot*/ 0);
    fprintf(stderr, "reuse: staged_mask slot0  before_rm=0x%x  after_rm=0x%x\n",
            (unsigned) staged_before, (unsigned) staged_after);

    if (staged_after != 0) {
        fprintf(stderr, "reuse: FAIL - seq_rm left stale staged residuals (mask=0x%x); reused seq id would fold them\n",
                (unsigned) staged_after);
        llama_free(ctx);
        return 1;
    }

    // sanity: the reused seq must still decode without crashing after the cleared staging
    if (!write_seq(ctx, /*tok*/ 2, /*n*/ 4, /*seq*/ 0)) {
        fprintf(stderr, "reuse: FAIL - reused seq decode failed after staging cleared\n");
        llama_free(ctx);
        return 1;
    }

    llama_free(ctx);
    fprintf(stderr, "reuse: PASS\n");
    return 0;
}

// fresh default-staging unified context (slot == seq)
static llama_context * mk_ctx(llama_model * model, const common_params & params, llama_kv_cache ** kv) {
    unsetenv("KPC_STAGING_SLOTS");
    auto cp = common_context_params_to_llama(params);
    cp.kv_unified = true;
    cp.n_seq_max  = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    *kv = ctx ? dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx)) : nullptr;
    return ctx;
}

// a) match-any removal: seq_rm(-1) must retire all staging slots
static int test_rm_any(llama_model * model, const common_params & params) {
    llama_kv_cache * kv = nullptr;
    llama_context * ctx = mk_ctx(model, params, &kv);
    if (!ctx || !kv) { fprintf(stderr, "rm_any: ctx create failed\n"); if (ctx) llama_free(ctx); return 1; }

    const int n_layer = llama_model_n_layer(model);

    if (!write_seq(ctx, 1, 8, 0)) { fprintf(stderr, "rm_any: write failed\n"); llama_free(ctx); return 1; }

    const int32_t before = any_staged(kv, n_layer, 0);
    if (before == 0) { fprintf(stderr, "rm_any: SKIP - no open group staged\n"); llama_free(ctx); return 0; }

    llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);

    const int32_t after = any_staged(kv, n_layer, 0);
    fprintf(stderr, "rm_any: staged_mask slot0  before=0x%x  after=0x%x\n", (unsigned) before, (unsigned) after);
    if (after != 0) {
        fprintf(stderr, "rm_any: FAIL - seq_rm(-1) left stale staged residuals\n");
        llama_free(ctx);
        return 1;
    }

    if (!write_seq(ctx, 2, 4, 0)) {
        fprintf(stderr, "rm_any: FAIL - decode after match-any removal failed\n");
        llama_free(ctx);
        return 1;
    }

    llama_free(ctx);
    fprintf(stderr, "rm_any: PASS\n");
    return 0;
}

// b) seq_keep(0): seq 1's slot must retire, seq 0's must stay staged unchanged
static int test_keep(llama_model * model, const common_params & params) {
    llama_kv_cache * kv = nullptr;
    llama_context * ctx = mk_ctx(model, params, &kv);
    if (!ctx || !kv) { fprintf(stderr, "keep: ctx create failed\n"); if (ctx) llama_free(ctx); return 1; }

    const int n_layer = llama_model_n_layer(model);

    if (!write_seq(ctx, 1, 8, 0) || !write_seq(ctx, 1, 8, 1)) {
        fprintf(stderr, "keep: write failed\n");
        llama_free(ctx);
        return 1;
    }

    std::vector<int32_t> slot0_before(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        slot0_before[il] = kv->kpc_staged_mask((uint32_t) il, 0);
    }
    if (any_staged(kv, n_layer, 0) == 0 || any_staged(kv, n_layer, 1) == 0) {
        fprintf(stderr, "keep: SKIP - no open group staged\n");
        llama_free(ctx);
        return 0;
    }

    llama_memory_seq_keep(llama_get_memory(ctx), 0);

    const int32_t s1_after = any_staged(kv, n_layer, 1);
    bool s0_intact = true;
    for (int il = 0; il < n_layer; ++il) {
        s0_intact = s0_intact && kv->kpc_staged_mask((uint32_t) il, 0) == slot0_before[il];
    }
    fprintf(stderr, "keep: after seq_keep(0)  slot1=0x%x  slot0 %s\n",
            (unsigned) s1_after, s0_intact ? "unchanged" : "CHANGED");
    if (s1_after != 0) {
        fprintf(stderr, "keep: FAIL - seq_keep(0) left seq 1's staging live\n");
        llama_free(ctx);
        return 1;
    }
    if (!s0_intact) {
        fprintf(stderr, "keep: FAIL - seq_keep(0) disturbed the kept sequence's staging\n");
        llama_free(ctx);
        return 1;
    }

    llama_free(ctx);
    fprintf(stderr, "keep: PASS\n");
    return 0;
}

// c) partial tail rewind: seq_rm(0, 36, -1) on a 40-token seq trims only members w4..w7 of the open group
static int test_rewind(llama_model * model, const common_params & params) {
    llama_kv_cache * kv = nullptr;
    llama_context * ctx = mk_ctx(model, params, &kv);
    if (!ctx || !kv) { fprintf(stderr, "rewind: ctx create failed\n"); if (ctx) llama_free(ctx); return 1; }

    const int n_layer = llama_model_n_layer(model);

    // positions 0..39: group 0 complete, group 1 open with members w0..w7 -> mask 0xff
    if (!write_seq(ctx, 1, 40, 0)) { fprintf(stderr, "rewind: write failed\n"); llama_free(ctx); return 1; }

    std::vector<int32_t> before(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        before[il] = kv->kpc_staged_mask((uint32_t) il, 0);
    }
    if (any_staged(kv, n_layer, 0) != 0xff) {
        fprintf(stderr, "rewind: SKIP - unexpected staged shape (mask=0x%x, want 0xff)\n",
                (unsigned) any_staged(kv, n_layer, 0));
        llama_free(ctx);
        return 0;
    }

    // remove positions 36..39 -> w4..w7 cleared, w0..w3 (positions 32..35) keep their bits
    llama_memory_seq_rm(llama_get_memory(ctx), 0, 36, -1);

    int bad = 0;
    for (int il = 0; il < n_layer; ++il) {
        const int32_t after = kv->kpc_staged_mask((uint32_t) il, 0);
        if (after != (before[il] & 0x0f)) {
            fprintf(stderr, "rewind: FAIL - layer %d mask before=0x%x after=0x%x want=0x%x\n",
                    il, (unsigned) before[il], (unsigned) after, (unsigned) (before[il] & 0x0f));
            ++bad;
        }
    }
    fprintf(stderr, "rewind: staged_mask slot0  before=0xff  after=0x%x (want 0x0f)\n",
            (unsigned) any_staged(kv, n_layer, 0));
    if (bad) { llama_free(ctx); return 1; }

    llama_batch b = llama_batch_init(1, 0, 1);
    common_batch_add(b, 1, 36, { 0 }, false);
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    if (!ok) {
        fprintf(stderr, "rewind: FAIL - decode at rewound position 36 failed\n");
        llama_free(ctx);
        return 1;
    }

    llama_free(ctx);
    fprintf(stderr, "rewind: PASS\n");
    return 0;
}

// d) clear(false): metadata clear must retire staging too
static int test_clear(llama_model * model, const common_params & params) {
    llama_kv_cache * kv = nullptr;
    llama_context * ctx = mk_ctx(model, params, &kv);
    if (!ctx || !kv) { fprintf(stderr, "clear: ctx create failed\n"); if (ctx) llama_free(ctx); return 1; }

    const int n_layer = llama_model_n_layer(model);

    if (!write_seq(ctx, 1, 8, 0)) { fprintf(stderr, "clear: write failed\n"); llama_free(ctx); return 1; }

    const int32_t before = any_staged(kv, n_layer, 0);
    if (before == 0) { fprintf(stderr, "clear: SKIP - no open group staged\n"); llama_free(ctx); return 0; }

    llama_memory_clear(llama_get_memory(ctx), false);

    const int32_t after = any_staged(kv, n_layer, 0);
    fprintf(stderr, "clear: staged_mask slot0  before=0x%x  after=0x%x\n", (unsigned) before, (unsigned) after);
    if (after != 0) {
        fprintf(stderr, "clear: FAIL - clear(false) left stale staged residuals\n");
        llama_free(ctx);
        return 1;
    }

    llama_free(ctx);
    fprintf(stderr, "clear: PASS\n");
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

    int fails = 0;
    fails += test_reuse (model, params);
    fails += test_rm_any(model, params);
    fails += test_keep  (model, params);
    fails += test_rewind(model, params);
    fails += test_clear (model, params);
    if (fails) { fprintf(stderr, "test-kpc-seq-reuse: %d FAIL(s)\n", fails); return 1; }
    fprintf(stderr, "test-kpc-seq-reuse: all passed\n");
    return 0;
}
