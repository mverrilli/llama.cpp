
#pragma once

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
static inline int setenv(const char * name, const char * value, int /*overwrite*/) { return _putenv_s(name, value); }
static inline int unsetenv(const char * name) { return _putenv_s(name, ""); }
#endif

// init model+context with the kpc4_1/q4_1 cache from argv (caller pre-fills params). returns null on
// parse fail; caller checks model() (bail) and context() (skip: incompatible head_dim)
static inline common_init_result_ptr kpc_model_init(int argc, char ** argv, common_params & params, const char * who) {
    params.cache_type_k    = GGML_TYPE_KPC4_1;
    params.cache_type_v    = GGML_TYPE_Q4_1;                 // KPC pairs kpc4_1 K with q4_1 V
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // KPC fused attention is a flash-attn op

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return nullptr;
    }

    ggml_backend_load_all();

    common_init_result_ptr init = common_init_from_params(params);
    if (init->model() == nullptr) {
        fprintf(stderr, "%s: failed to load model\n", who);
    } else if (init->context() == nullptr) {
        // any quantized KV cache (q4_0/q8_0/kpc4_1) needs n_embd_head_k % block_size == 0
        fprintf(stderr, "%s: skipping -- model incompatible with kpc4_1 (n_embd_head_k must be a multiple of 32)\n", who);
    }
    return init;
}

static inline std::vector<float> decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq, int n_vocab) {
    llama_batch b = llama_batch_init(1, 0, 1);
    common_batch_add(b, tok, pos, { seq }, true);
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

// decode P tokens on `seq` at positions [0,P), then return its next-token logits at position P
static inline std::vector<float> run_seq(llama_context * ctx, llama_seq_id seq, int P, llama_token tok, int n_vocab) {
    llama_batch batch = llama_batch_init(P, 0, 1);
    for (int i = 0; i < P; ++i) {
        common_batch_add(batch, tok, i, { seq }, false);
    }
    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        return {};
    }
    llama_batch_free(batch);
    return decode_one(ctx, tok, P, seq, n_vocab);
}

// decode n tokens at positions [0,n) for `seq`, no logits
static inline bool write_seq(llama_context * ctx, llama_token tok, int n, llama_seq_id seq) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) common_batch_add(b, tok, i, { seq }, false);
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

// decode a 32-token chunk for `seq` at [pos0, pos0+32) (a complete group)
static inline bool decode_chunk(llama_context * ctx, llama_seq_id seq, int pos0, llama_token tok) {
    llama_batch b = llama_batch_init(32, 0, 1);
    for (int i = 0; i < 32; ++i) {
        common_batch_add(b, tok, pos0 + i, { seq }, false);
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static inline float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) {
        return 1e30f;
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static inline int argmax(const std::vector<float> & v) {
    int best = -1;
    float bv = -INFINITY;
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] > bv) { bv = v[i]; best = (int) i; }
    }
    return best;
}

static inline bool finite_vec(const std::vector<float> & v) {
    if (v.empty()) return false;
    for (float x : v) { if (!std::isfinite(x)) return false; }
    return true;
}
