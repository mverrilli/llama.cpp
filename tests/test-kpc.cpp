// unit tests for the KPC per-channel int4 K-cache ops (ggml-cpu/kpc.cpp)

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define KPC_GROUP             GGML_KPC_GROUP   // shared via ggml.h

#define KPC_SZ_GROUP_BYTES(C) GGML_KPC_SZ_GROUP_BYTES(C)

// decode one channel of a group's int8 scalezp slab (mirror of kpc_sz_decode1)
static inline void kpc_sz_dec1(const uint8_t * slab, int64_t C, int64_t c, float * scale, float * zp) {
    ggml_fp16_t h; float ss, zmin, sz;
    memcpy(&h, slab + 0, 2); ss   = ggml_fp16_to_fp32(h);
    memcpy(&h, slab + 2, 2); zmin = ggml_fp16_to_fp32(h);
    memcpy(&h, slab + 4, 2); sz   = ggml_fp16_to_fp32(h);
    *scale = slab[6 + c] * ss;
    *zp    = zmin + slab[6 + C + c] * sz;
}

// deterministic source matrix value for (channel c, token t); stable across runs and across one-shot/chunked
static float src_val(int64_t c, int64_t t) {
    return 0.7f*sinf(0.10f*(float)c + 0.37f*(float)t) + 0.3f*cosf(0.013f*(float)(c*t) + 1.3f);
}

// write tokens [0,N) into a fresh cache in `chunks` (contiguous ubatches at the running head)
// so residual staging is exercised as in decode; returns packed bytes + scalezp
static void build_cache(int64_t C, int64_t kv, int64_t N, const std::vector<int64_t> & chunks,
                        int n_threads, std::vector<uint8_t> & k_out, std::vector<uint8_t> & sz_out) {
    struct ggml_init_params ip = { (size_t) 256*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t ng = kv / KPC_GROUP;
    struct ggml_tensor * k  = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, kv);
    struct ggml_tensor * sz = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
    struct ggml_tensor * rs = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, C, KPC_GROUP);
    struct ggml_tensor * gi = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kv, 1);
    struct ggml_tensor * rsl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPC_GROUP, 1);
    struct ggml_tensor * sgp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * smk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    memset(k->data,   0, ggml_nbytes(k));
    memset(sz->data,  0, ggml_nbytes(sz));
    memset(rs->data,  0, ggml_nbytes(rs));
    memset(gi->data, 0xFF, ggml_nbytes(gi));
    memset(rsl->data, 0, ggml_nbytes(rsl));
    memset(sgp->data, 0, ggml_nbytes(sgp));
    memset(smk->data, 0, ggml_nbytes(smk));   // staged_mask==0 -> no open group (fresh)

    int64_t head = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const int64_t cs = chunks[ci];
        struct ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, cs);
        struct ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, cs);
        struct ggml_tensor * ks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cs);   // kpc_seq
        struct ggml_tensor * kp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cs);   // kpc_pos
        float   * kcd = (float   *) kc->data;
        int64_t * idd = (int64_t *) id->data;
        int32_t * ksd = (int32_t *) ks->data;
        int32_t * kpd = (int32_t *) kp->data;
        for (int64_t i = 0; i < cs; ++i) {
            for (int64_t c = 0; c < C; ++c) {
                kcd[i*C + c] = src_val(c, head + i);
            }
            idd[i] = head + i;                 // single stream: global slot == stream-local head == position
            ksd[i] = 0;
            kpd[i] = (int32_t)(head + i);      // true position
        }
        struct ggml_tensor * out = ggml_kpc_write(ctx, k, sz, rs, gi, rsl, sgp, smk, kc, id, ks, kp, 1);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        head += cs;
    }
    GGML_ASSERT(head == N);

    k_out.resize(ggml_nbytes(k));
    memcpy(k_out.data(), k->data, ggml_nbytes(k));
    sz_out.resize(ggml_nelements(sz));
    memcpy(sz_out.data(), sz->data, ggml_nbytes(sz));

    ggml_free(ctx);
}

// like build_cache but logical token i lands at physical slot perm[i] (SWA ring-wrap / defrag
// scatter); groups still form by logical pos/32, only placement is scattered
static void build_cache_scatter(int64_t C, int64_t kv, int64_t N, const std::vector<int64_t> & chunks,
                                const std::vector<int64_t> & perm, int n_threads,
                                std::vector<uint8_t> & k_out, std::vector<uint8_t> & sz_out,
                                std::vector<int32_t> & gi_out) {
    struct ggml_init_params ip = { (size_t) 256*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t ng = kv / KPC_GROUP;
    struct ggml_tensor * k  = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, kv);
    struct ggml_tensor * sz = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
    struct ggml_tensor * rs = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, C, KPC_GROUP);
    struct ggml_tensor * gi = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kv, 1);
    struct ggml_tensor * rsl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPC_GROUP, 1);
    struct ggml_tensor * sgp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * smk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    memset(k->data,   0, ggml_nbytes(k));
    memset(sz->data,  0, ggml_nbytes(sz));
    memset(rs->data,  0, ggml_nbytes(rs));
    memset(gi->data, 0xFF, ggml_nbytes(gi));
    memset(rsl->data, 0, ggml_nbytes(rsl));
    memset(sgp->data, 0, ggml_nbytes(sgp));
    memset(smk->data, 0, ggml_nbytes(smk));

    int64_t head = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const int64_t cs = chunks[ci];
        struct ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, cs);
        struct ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, cs);
        struct ggml_tensor * ks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cs);   // kpc_seq
        struct ggml_tensor * kp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cs);   // kpc_pos
        float   * kcd = (float   *) kc->data;
        int64_t * idd = (int64_t *) id->data;
        int32_t * ksd = (int32_t *) ks->data;
        int32_t * kpd = (int32_t *) kp->data;
        for (int64_t i = 0; i < cs; ++i) {
            for (int64_t c = 0; c < C; ++c) {
                kcd[i*C + c] = src_val(c, head + i);   // value indexed by LOGICAL position (head+i)
            }
            idd[i] = perm[head + i];                   // physical slot is scattered via the permutation
            ksd[i] = 0;
            kpd[i] = (int32_t)(head + i);              // LOGICAL position
        }
        struct ggml_tensor * out = ggml_kpc_write(ctx, k, sz, rs, gi, rsl, sgp, smk, kc, id, ks, kp, 1);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        head += cs;
    }
    GGML_ASSERT(head == N);

    k_out.resize(ggml_nbytes(k));
    memcpy(k_out.data(), k->data, ggml_nbytes(k));
    sz_out.resize(ggml_nelements(sz));
    memcpy(sz_out.data(), sz->data, ggml_nbytes(sz));
    gi_out.resize(kv);
    memcpy(gi_out.data(), gi->data, kv * sizeof(int32_t));

    ggml_free(ctx);
}

// Dequantize the first N tokens of a packed cache.
static void dequant_cache(int64_t C, int64_t N, const std::vector<uint8_t> & k_bytes,
                          const std::vector<uint8_t> & sz, int n_threads, std::vector<float> & deq_out) {
    struct ggml_init_params ip = { (size_t) 64*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t ng = (N + KPC_GROUP - 1) / KPC_GROUP;
    struct ggml_tensor * pk = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, N);
    struct ggml_tensor * s  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
    memcpy(pk->data, k_bytes.data(), ggml_nbytes(pk));        // packed rows are contiguous from slot 0
    memcpy(s->data,  sz.data(),      ggml_nbytes(s));

    struct ggml_tensor * d = ggml_kpc_dequant(ctx, pk, s, NULL);   // contiguous: slot/32 grouping
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    deq_out.resize(C*N);
    const ggml_fp16_t * dd = (const ggml_fp16_t *) d->data;
    for (int64_t i = 0; i < C*N; ++i) {
        deq_out[i] = ggml_fp16_to_fp32(dd[i]);
    }
    ggml_free(ctx);
}

// NMSE of a dequantized [C,N] tile against the original source matrix, plus the max abs error.
static double tile_nmse(int64_t C, int64_t N, const std::vector<float> & deq, double & maxerr) {
    double se = 0.0, sref = 0.0;
    maxerr = 0.0;
    for (int64_t t = 0; t < N; ++t) {
        for (int64_t c = 0; c < C; ++c) {
            const float orig = src_val(c, t);
            const float e    = fabsf(orig - deq[t*C + c]);
            se   += (double) e * e;
            sref += (double) orig * orig;
            if (e > maxerr) {
                maxerr = e;
            }
        }
    }
    return se / (sref > 0.0 ? sref : 1.0);
}

// fused attention (ggml_kpc_attn) vs a full-precision reference; the kernel quantizes q to
// int8 per group so an exact match is not expected, small NMSE is the bar
static int test_attn(int n_threads) {
    const int64_t DK = 64, DV = 64, NKV = 40;            // partial group
    const int64_t ng = (NKV + KPC_GROUP - 1) / KPC_GROUP;

    std::vector<uint8_t>     k_os;
    std::vector<uint8_t> sz_os;
    build_cache(DK, 64, NKV, { NKV }, n_threads, k_os, sz_os);
    std::vector<float> kdeq;                              // dequantized K [NKV][DK]
    dequant_cache(DK, NKV, k_os, sz_os, n_threads, kdeq);

    std::vector<float> q(DK), vf(DV * NKV);
    for (int64_t c = 0; c < DK; ++c) {
        q[c] = 0.5f * sinf(0.21f*(float) c + 0.6f);
    }
    for (int64_t ic = 0; ic < NKV; ++ic) {
        for (int64_t d = 0; d < DV; ++d) {
            vf[ic*DV + d] = 0.4f * cosf(0.07f*(float) d + 0.11f*(float) ic);
        }
    }

    // reference attention
    std::vector<double> sc(NKV);
    double smax = -1e30;
    for (int64_t ic = 0; ic < NKV; ++ic) {
        double s = 0.0;
        for (int64_t c = 0; c < DK; ++c) {
            s += (double) q[c] * kdeq[ic*DK + c];
        }
        sc[ic] = s;
        if (s > smax) {
            smax = s;
        }
    }
    double ssum = 0.0;
    for (int64_t ic = 0; ic < NKV; ++ic) {
        sc[ic] = exp(sc[ic] - smax);
        ssum += sc[ic];
    }
    std::vector<float> out_ref(DV, 0.0f);
    for (int64_t ic = 0; ic < NKV; ++ic) {
        const float w = (float) (sc[ic] / ssum);
        for (int64_t d = 0; d < DV; ++d) {
            out_ref[d] += w * vf[ic*DV + d];
        }
    }

    // fused op
    struct ggml_init_params ip = { (size_t) 64*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * pk = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, DK, NKV);
    struct ggml_tensor * sz = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(DK), ng);
    struct ggml_tensor * qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, 1, 1, 1);
    struct ggml_tensor * vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, NKV, 1, 1);
    struct ggml_tensor * mk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, NKV, 1);
    memcpy(pk->data, k_os.data(),  ggml_nbytes(pk));
    memcpy(sz->data, sz_os.data(), ggml_nbytes(sz));
    memcpy(qt->data, q.data(),     DK * sizeof(float));
    ggml_fp16_t * vd = (ggml_fp16_t *) vt->data;
    for (int64_t i = 0; i < DV*NKV; ++i) {
        vd[i] = ggml_fp32_to_fp16(vf[i]);
    }
    ggml_fp16_t * mdat = (ggml_fp16_t *) mk->data;
    for (int64_t i = 0; i < NKV; ++i) {
        mdat[i] = ggml_fp32_to_fp16(0.0f);
    }
    struct ggml_tensor * gx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NKV, 1);
    int32_t * gxd = (int32_t *) gx->data;
    for (int64_t i = 0; i < NKV; ++i) {
        gxd[i] = (int32_t) (i / KPC_GROUP);   // contiguous: slot -> group = slot/32
    }

    struct ggml_tensor * o = ggml_kpc_attn(ctx, qt, pk, sz, vt, mk, gx, NULL, 1.0f, 0.0f, 0.0f, 1);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    const float * od = (const float *) o->data;   // [DV, 1, 1, 1]
    double se = 0.0, sref = 0.0, maxe = 0.0;
    int nan = 0;
    for (int64_t d = 0; d < DV; ++d) {
        if (od[d] != od[d]) {
            nan++;
        }
        const float e = fabsf(od[d] - out_ref[d]);
        se   += (double) e * e;
        sref += (double) out_ref[d] * out_ref[d];
        if (e > maxe) {
            maxe = e;
        }
    }
    const double nmse = se / (sref > 0.0 ? sref : 1.0);
    ggml_free(ctx);

    if (nan == 0 && nmse < 0.02 && maxe < 0.05) {
        printf("PASS fused-attn: NMSE=%.5f maxerr=%.4f\n", nmse, maxe);
        return 0;
    }
    printf("FAIL fused-attn: NMSE=%.5f maxerr=%.4f nan=%d\n", nmse, maxe, nan);
    return 1;
}

// fused attention with logit-softcap, attention-sinks and ALiBi vs a full-precision reference;
// multiple heads so the per-head ALiBi slope and sink logit are distinct, small NMSE is the bar
static int test_attn_extras(int n_threads, float softcap, bool use_sinks, float max_bias, const char * label) {
    const int64_t DK = 64, DV = 64, NKV = 40, NH = 4;    // NKV not group-aligned -> partial group; multi-head
    const int64_t ng = (NKV + KPC_GROUP - 1) / KPC_GROUP;

    // one shared K-cache (single kv head -> GQA broadcast over the 4 q heads)
    std::vector<uint8_t>     k_os;
    std::vector<uint8_t> sz_os;
    build_cache(DK, 64, NKV, { NKV }, n_threads, k_os, sz_os);
    std::vector<float> kdeq;                              // dequantized K [NKV][DK] (what the kernel reads back)
    dequant_cache(DK, NKV, k_os, sz_os, n_threads, kdeq);

    // per-head q, shared V, per-head sink logit
    std::vector<float> q(DK*NH), vf(DV*NKV), sinks(NH);
    for (int64_t hh = 0; hh < NH; ++hh) {
        for (int64_t c = 0; c < DK; ++c) {
            q[hh*DK + c] = 0.5f * sinf(0.21f*(float) c + 0.6f + 0.3f*(float) hh);
        }
        sinks[hh] = 0.2f + 0.15f*(float) hh;
    }
    for (int64_t ic = 0; ic < NKV; ++ic) {
        for (int64_t d = 0; d < DV; ++d) {
            vf[ic*DV + d] = 0.4f * cosf(0.07f*(float) d + 0.11f*(float) ic);
        }
    }

    // ALiBi: mask carries the position term (pos - last), slope applied per head in-kernel. We feed a
    // causal-style mask of (ic - (NKV-1)) so the reference slope*mask matches the kernel exactly.
    std::vector<float> maskf(NKV);
    for (int64_t ic = 0; ic < NKV; ++ic) {
        maskf[ic] = (max_bias > 0.0f) ? (float)(ic - (NKV - 1)) : 0.0f;
    }

    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) NH));
    const float    m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float    m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // full-precision reference per head
    std::vector<float> out_ref(DV*NH, 0.0f);
    for (int64_t hh = 0; hh < NH; ++hh) {
        const float slope = (max_bias > 0.0f) ? ((uint64_t) hh < n_head_log2 ? powf(m0, hh + 1) : powf(m1, 2*(hh - n_head_log2) + 1)) : 1.0f;
        std::vector<double> sc(NKV);
        double smax = -1e30;
        for (int64_t ic = 0; ic < NKV; ++ic) {
            double s = 0.0;
            for (int64_t c = 0; c < DK; ++c) {
                s += (double) q[hh*DK + c] * kdeq[ic*DK + c];
            }
            if (softcap != 0.0f) {
                s = (double) softcap * tanh(s / (double) softcap);
            }
            s += (double) slope * maskf[ic];
            sc[ic] = s;
            if (s > smax) smax = s;
        }
        double ssum = 0.0;
        if (use_sinks && sinks[hh] > smax) smax = sinks[hh];   // sink participates in the max
        for (int64_t ic = 0; ic < NKV; ++ic) {
            sc[ic] = exp(sc[ic] - smax);
            ssum += sc[ic];
        }
        if (use_sinks) ssum += exp((double) sinks[hh] - smax);  // sink in denominator, no V contribution
        for (int64_t ic = 0; ic < NKV; ++ic) {
            const float w = (float)(sc[ic] / ssum);
            for (int64_t d = 0; d < DV; ++d) {
                out_ref[hh*DV + d] += w * vf[ic*DV + d];
            }
        }
    }

    // fused op
    struct ggml_init_params ip = { (size_t) 64*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * pk = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, DK, NKV);
    struct ggml_tensor * sz = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(DK), ng);
    struct ggml_tensor * qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, 1, NH, 1);   // [DK, n_q=1, n_head, n_stream]
    struct ggml_tensor * vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, NKV, 1, 1);  // single kv head -> GQA bcast
    struct ggml_tensor * mk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, NKV, 1);
    struct ggml_tensor * skt = use_sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NH) : NULL;
    memcpy(pk->data, k_os.data(),  ggml_nbytes(pk));
    memcpy(sz->data, sz_os.data(), ggml_nbytes(sz));
    memcpy(qt->data, q.data(),     DK*NH * sizeof(float));
    ggml_fp16_t * vd = (ggml_fp16_t *) vt->data;
    for (int64_t i = 0; i < DV*NKV; ++i) {
        vd[i] = ggml_fp32_to_fp16(vf[i]);
    }
    ggml_fp16_t * mdat = (ggml_fp16_t *) mk->data;
    for (int64_t i = 0; i < NKV; ++i) {
        mdat[i] = ggml_fp32_to_fp16(maskf[i]);
    }
    if (skt) {
        memcpy(skt->data, sinks.data(), NH * sizeof(float));
    }
    struct ggml_tensor * gx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NKV, 1);
    int32_t * gxd = (int32_t *) gx->data;
    for (int64_t i = 0; i < NKV; ++i) {
        gxd[i] = (int32_t)(i / KPC_GROUP);   // group_index = slot/KPC_GROUP
    }

    struct ggml_tensor * o = ggml_kpc_attn(ctx, qt, pk, sz, vt, mk, gx, skt, 1.0f, max_bias, softcap, 1);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    const float * od = (const float *) o->data;   // head stride = DV
    double se = 0.0, sref = 0.0, maxe = 0.0;
    int nan = 0;
    for (int64_t hh = 0; hh < NH; ++hh) {
        for (int64_t d = 0; d < DV; ++d) {
            const float got = od[hh*DV + d];
            if (got != got) nan++;
            const float e = fabsf(got - out_ref[hh*DV + d]);
            se   += (double) e * e;
            sref += (double) out_ref[hh*DV + d] * out_ref[hh*DV + d];
            if (e > maxe) maxe = e;
        }
    }
    const double nmse = se / (sref > 0.0 ? sref : 1.0);
    ggml_free(ctx);

    if (nan == 0 && nmse < 0.02 && maxe < 0.05) {
        printf("PASS fused-attn-%s: NMSE=%.5f maxerr=%.4f vs full-precision reference\n", label, nmse, maxe);
        return 0;
    }
    printf("FAIL fused-attn-%s: NMSE=%.5f maxerr=%.4f nan=%d\n", label, nmse, maxe, nan);
    return 1;
}

// Dequantize a single logical token: read the packed nibble row at physical `slot`, with per-channel
// scale/zp from scalezp[group_index[slot]]. Mirrors the kernel's read path. Fills out[C].
static void deq_slot(int64_t C, int64_t slot, const std::vector<uint8_t> & kb, const std::vector<uint8_t> & sz,
                     const std::vector<int32_t> & gi, float * out) {
    const int64_t krow = C / 2;
    const int64_t pool = gi[slot];
    const uint8_t * row  = kb.data() + slot*krow;
    const uint8_t * slab = sz.data() + pool*KPC_SZ_GROUP_BYTES(C);
    for (int64_t c = 0; c < C; ++c) {
        const int q = (c & 1) ? (row[c/2] >> 4) : (row[c/2] & 0x0F);
        float s, z; kpc_sz_dec1(slab, C, c, &s, &z);
        out[c] = q * s + z;
    }
}

// Run the real 3-arg ggml_kpc_dequant over `slots` physical slots, reading scalezp[group_index[slot]].
// Returns the dequantized [C, slots] tile (token-major: out[slot*C + c]).
static void dequant_cache_gi(int64_t C, int64_t slots, const std::vector<uint8_t> & kb,
                             const std::vector<uint8_t> & sz, int64_t ng,
                             const std::vector<int32_t> & gi, int n_threads, std::vector<float> & deq_out) {
    struct ggml_init_params ip = { (size_t) 64*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * pk = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, slots);
    struct ggml_tensor * s  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
    struct ggml_tensor * gx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, slots);
    memcpy(pk->data, kb.data(), ggml_nbytes(pk));
    memcpy(s->data,  sz.data(), ggml_nbytes(s));
    memcpy(gx->data, gi.data(), slots * sizeof(int32_t));

    struct ggml_tensor * d = ggml_kpc_dequant(ctx, pk, s, gx);   // group_index-aware path
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    deq_out.resize(C*slots);
    const ggml_fp16_t * dd = (const ggml_fp16_t *) d->data;
    for (int64_t i = 0; i < C*slots; ++i) {
        deq_out[i] = ggml_fp16_to_fp32(dd[i]);
    }
    ggml_free(ctx);
}

// 3-arg ggml_kpc_dequant over a scattered cache: each logical token must dequant bit-identical
// to the same token in a contiguous cache via the NULL (slot/32) path
static int test_scatter_dequant_op(int n_threads) {
    const int64_t C = 128, kv = 128, N = 70;             // 2 full groups + a partial of 6
    const int64_t ng = kv / KPC_GROUP;
    std::vector<int64_t> identity(N), perm(N);
    for (int64_t i = 0; i < N; ++i) identity[i] = i;
    for (int64_t i = 0; i < N; ++i) perm[i] = (i + 100) % kv;   // ring-wrap scatter

    const std::vector<int64_t> chunks = { 7, 25, 13, 19, 6 };

    std::vector<uint8_t>     k_id,  k_sc;
    std::vector<uint8_t> sz_id, sz_sc;
    std::vector<int32_t>     gi_id, gi_sc;
    build_cache_scatter(C, kv, N, chunks, identity, n_threads, k_id, sz_id, gi_id);
    build_cache_scatter(C, kv, N, chunks, perm,     n_threads, k_sc, sz_sc, gi_sc);

    // contiguous reference: NULL group_index (slot/32) over the first N slots
    std::vector<float> deq_ref;
    dequant_cache(C, N, k_id, sz_id, n_threads, deq_ref);
    // scattered: full cache through the op with the scattered group_index
    std::vector<float> deq_sc;
    dequant_cache_gi(C, kv, k_sc, sz_sc, ng, gi_sc, n_threads, deq_sc);

    int mismatches = 0;
    for (int64_t i = 0; i < N; ++i) {
        const float * a = deq_ref.data() + identity[i]*C;   // logical token i, contiguous
        const float * b = deq_sc.data()  + perm[i]*C;       // same token at its scattered physical slot
        for (int64_t c = 0; c < C; ++c) {
            if (a[c] != b[c]) { mismatches++; break; }
        }
    }

    if (mismatches == 0) {
        printf("PASS scatter-dequant-op: %lld logical tokens dequant bit-identical via 3-arg op (group_index)\n",
               (long long) N);
        return 0;
    }
    printf("FAIL scatter-dequant-op: %d logical token(s) differ (group_index dequant wrong)\n", mismatches);
    return 1;
}

// scatter-write invariant: logical positions written to scattered physical slots must decode
// bit-identical per token to a contiguous write (grouping is by logical position, not slot)
static int test_scatter(int n_threads) {
    const int64_t C = 128, kv = 128, N = 70;             // 2 full groups + a partial of 6
    std::vector<int64_t> identity(N), perm(N);
    for (int64_t i = 0; i < N; ++i) identity[i] = i;
    // a rotation by 100 within [0,kv): logical i -> physical (i+100)%kv. Genuinely wraps the ring (i+100 > kv
    // for most i), so physical slots are scattered/non-monotonic and groups land on disjoint slot runs.
    for (int64_t i = 0; i < N; ++i) perm[i] = (i + 100) % kv;

    // both written in the same arbitrary chunking so staging is exercised identically
    const std::vector<int64_t> chunks = { 7, 25, 13, 19, 6 };

    std::vector<uint8_t>     k_id,  k_sc;
    std::vector<uint8_t> sz_id, sz_sc;
    std::vector<int32_t>     gi_id, gi_sc;
    build_cache_scatter(C, kv, N, chunks, identity, n_threads, k_id, sz_id, gi_id);
    build_cache_scatter(C, kv, N, chunks, perm,     n_threads, k_sc, sz_sc, gi_sc);

    int mismatches = 0;
    std::vector<float> a(C), b(C);
    for (int64_t i = 0; i < N; ++i) {
        deq_slot(C, identity[i], k_id, sz_id, gi_id, a.data());
        deq_slot(C, perm[i],     k_sc, sz_sc, gi_sc, b.data());
        for (int64_t c = 0; c < C; ++c) {
            if (a[c] != b[c]) { mismatches++; break; }   // bit-identical dequant per logical token
        }
        // group_index points at logical group
        if ((int64_t) gi_sc[perm[i]] != (i / KPC_GROUP) % (kv / KPC_GROUP)) mismatches++;
    }

    if (mismatches == 0) {
        printf("PASS scatter-write: %lld logical tokens decode bit-identical contiguous vs scattered (rot+100)\n",
               (long long) N);
        return 0;
    }
    printf("FAIL scatter-write: %d logical token(s) differ between contiguous and scattered placement\n", mismatches);
    return 1;
}

// per-sequence source value (distinct per seq so cross-seq pool collisions are detectable)
static float src_val_seq(int64_t seq, int64_t c, int64_t t) {
    return src_val(c, t) + 0.21f*(float) seq * cosf(0.05f*(float) c - 0.02f*(float) t);
}

// multi-sequence interleaved write into a unified cache (n_stream=1) in one op call; each seq
// writes positions [0,npos) into its own disjoint slot range, tokens interleaved in the ubatch
static void build_cache_multiseq(int64_t C, int64_t kv, int64_t NS, int64_t npos, int n_threads,
                                 std::vector<uint8_t> & k_out, std::vector<uint8_t> & sz_out,
                                 std::vector<int32_t> & gi_out) {
    struct ggml_init_params ip = { (size_t) 256*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t ng    = kv / KPC_GROUP;        // ng_max
    const int64_t rsize = kv / NS;               // per-seq slot region
    const int64_t nt    = NS * npos;
    GGML_ASSERT(npos <= rsize);

    struct ggml_tensor * k   = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, kv);
    struct ggml_tensor * sz  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng, 1);   // n_stream=1
    struct ggml_tensor * rs  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, C, KPC_GROUP, NS);
    struct ggml_tensor * gi  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kv, 1);
    struct ggml_tensor * rsl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPC_GROUP, NS);
    struct ggml_tensor * sgp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NS);
    struct ggml_tensor * smk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NS);
    memset(k->data, 0, ggml_nbytes(k));   memset(sz->data, 0, ggml_nbytes(sz));
    memset(rs->data, 0, ggml_nbytes(rs)); memset(gi->data, 0xFF, ggml_nbytes(gi));
    memset(rsl->data, 0, ggml_nbytes(rsl));
    memset(sgp->data, 0, ggml_nbytes(sgp));
    memset(smk->data, 0, ggml_nbytes(smk));

    struct ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, nt);
    struct ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);
    struct ggml_tensor * ks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
    struct ggml_tensor * kp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
    float   * kcd = (float   *) kc->data;
    int64_t * idd = (int64_t *) id->data;
    int32_t * ksd = (int32_t *) ks->data;
    int32_t * kpd = (int32_t *) kp->data;
    int64_t ti = 0;
    for (int64_t p = 0; p < npos; ++p) {
        for (int64_t sb = 0; sb < NS; ++sb) {
            for (int64_t c = 0; c < C; ++c) {
                kcd[ti*C + c] = src_val_seq(sb, c, p);
            }
            idd[ti] = sb*rsize + p;     // disjoint physical slot range per seq (global slot, n_stream=1)
            ksd[ti] = (int32_t)(sb | (sb << GGML_KPC_SLOT_SHIFT));   // pack slot==seq
            kpd[ti] = (int32_t) p;
            ti++;
        }
    }
    GGML_ASSERT(ti == nt);

    struct ggml_tensor * out = ggml_kpc_write(ctx, k, sz, rs, gi, rsl, sgp, smk, kc, id, ks, kp, (int32_t) NS);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    k_out.resize(ggml_nbytes(k));
    memcpy(k_out.data(), k->data, ggml_nbytes(k));
    sz_out.resize(ggml_nelements(sz));
    memcpy(sz_out.data(), sz->data, ggml_nbytes(sz));
    gi_out.resize(kv);
    memcpy(gi_out.data(), gi->data, kv * sizeof(int32_t));

    ggml_free(ctx);
}

// interleaved seqs must land in disjoint pool bands (seq sb -> [sb*band_size,(sb+1)*band_size))
// and each seq must dequant bit-identical to a standalone single-seq build
static int test_multiseq(int n_threads) {
    const int64_t C = 128, kv = 128, NS = 2, npos = 64;   // ng_max=4, band_size=2; 2 full groups per seq
    const int64_t rsize = kv / NS;
    const int64_t band_size = (kv / KPC_GROUP) / NS;

    std::vector<uint8_t>     k_ms;
    std::vector<uint8_t> sz_ms;
    std::vector<int32_t>     gi_ms;
    build_cache_multiseq(C, kv, NS, npos, n_threads, k_ms, sz_ms, gi_ms);

    int mismatches = 0, pool_errs = 0;
    std::vector<float> got(C);
    for (int64_t sb = 0; sb < NS; ++sb) {
        // independent single-seq reference cache for this seq's values, one-shot
        std::vector<uint8_t>     k_ref;
        std::vector<uint8_t> sz_ref;
        {
            struct ggml_init_params ip = { (size_t) 64*1024*1024, NULL, false };
            struct ggml_context * ctx = ggml_init(ip);
            const int64_t ng = kv / KPC_GROUP;
            struct ggml_tensor * k  = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, kv);
            struct ggml_tensor * sz = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
            struct ggml_tensor * rs = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, C, KPC_GROUP);
            struct ggml_tensor * gi = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kv, 1);
            struct ggml_tensor * rsl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPC_GROUP, 1);
            struct ggml_tensor * sgp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            struct ggml_tensor * smk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            memset(k->data, 0, ggml_nbytes(k));   memset(sz->data, 0, ggml_nbytes(sz));
            memset(rs->data, 0, ggml_nbytes(rs)); memset(gi->data, 0xFF, ggml_nbytes(gi));
            memset(rsl->data, 0, ggml_nbytes(rsl));
            memset(sgp->data, 0, ggml_nbytes(sgp)); memset(smk->data, 0, ggml_nbytes(smk));
            struct ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, npos);
            struct ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, npos);
            struct ggml_tensor * ks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, npos);
            struct ggml_tensor * kp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, npos);
            float   * kcd = (float   *) kc->data;
            int64_t * idd = (int64_t *) id->data;
            int32_t * ksd = (int32_t *) ks->data;
            int32_t * kpd = (int32_t *) kp->data;
            for (int64_t p = 0; p < npos; ++p) {
                for (int64_t c = 0; c < C; ++c) kcd[p*C + c] = src_val_seq(sb, c, p);
                idd[p] = p; ksd[p] = 0; kpd[p] = (int32_t) p;
            }
            struct ggml_tensor * out = ggml_kpc_write(ctx, k, sz, rs, gi, rsl, sgp, smk, kc, id, ks, kp, 1);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_graph_compute_with_ctx(ctx, gf, n_threads);
            k_ref.resize(ggml_nbytes(k));
            memcpy(k_ref.data(), k->data, ggml_nbytes(k));
            sz_ref.resize(ggml_nelements(sz));
            memcpy(sz_ref.data(), sz->data, ggml_nbytes(sz));
            ggml_free(ctx);
        }

        std::vector<float> ref(C);
        std::vector<int32_t> gi_ref(kv, 0);
        for (int64_t p = 0; p < npos; ++p) gi_ref[p] = (int32_t)(p / KPC_GROUP);   // single-seq pool = lg
        for (int64_t p = 0; p < npos; ++p) {
            const int64_t slot = sb*rsize + p;
            // pool in seq's band
            const int64_t pool = gi_ms[slot];
            if (pool < sb*band_size || pool >= (sb+1)*band_size) pool_errs++;
            deq_slot(C, slot, k_ms, sz_ms, gi_ms, got.data());
            deq_slot(C, p,    k_ref, sz_ref, gi_ref, ref.data());
            for (int64_t c = 0; c < C; ++c) {
                if (got[c] != ref[c]) { mismatches++; break; }
            }
        }
    }

    if (mismatches == 0 && pool_errs == 0) {
        printf("PASS multiseq: %d seqs, each reconstructs bit-identical\n", (int) NS);
        return 0;
    }
    printf("FAIL multiseq: %d cell mismatch(es), %d pool-band error(s) (cross-seq collision)\n", mismatches, pool_errs);
    return 1;
}

// continuous-batch decode steps: one token per seq per combined write; each seq must dequant
// bit-identical to a standalone single-seq write (live groups keep their own scales)
static int test_contig_batch_steps(int n_threads) {
    const int64_t C = 128, kv = 128, NS = 2;
    const int64_t ng = kv / KPC_GROUP, rsize = kv / NS;
    const int64_t band_size = ng / NS;
    const int64_t nsteps = 40;   // crosses a group boundary (pos 0..39 -> groups 0 and 1)

    struct ggml_init_params ip = { (size_t) 256*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * k   = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, kv);
    struct ggml_tensor * sz  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng, 1);
    struct ggml_tensor * rs  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, C, KPC_GROUP, NS);
    struct ggml_tensor * gi  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kv, 1);
    struct ggml_tensor * rsl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPC_GROUP, NS);
    struct ggml_tensor * sgp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NS);
    struct ggml_tensor * smk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NS);
    memset(k->data, 0, ggml_nbytes(k));   memset(sz->data, 0, ggml_nbytes(sz));
    memset(rs->data, 0, ggml_nbytes(rs)); memset(gi->data, 0xFF, ggml_nbytes(gi));
    memset(rsl->data, 0, ggml_nbytes(rsl));
    memset(sgp->data, 0, ggml_nbytes(sgp)); memset(smk->data, 0, ggml_nbytes(smk));

    for (int64_t p = 0; p < nsteps; ++p) {
        const int64_t nt = NS;   // one token per seq this step
        struct ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, nt);
        struct ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nt);
        struct ggml_tensor * ks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
        struct ggml_tensor * kp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
        float   * kcd = (float   *) kc->data;
        int64_t * idd = (int64_t *) id->data;
        int32_t * ksd = (int32_t *) ks->data;
        int32_t * kpd = (int32_t *) kp->data;
        for (int64_t sb = 0; sb < NS; ++sb) {
            for (int64_t c = 0; c < C; ++c) kcd[sb*C + c] = src_val_seq(sb, c, p);
            idd[sb] = sb*rsize + p;
            ksd[sb] = (int32_t)(sb | (sb << GGML_KPC_SLOT_SHIFT));   // pack slot==seq
            kpd[sb] = (int32_t) p;
        }
        struct ggml_tensor * out = ggml_kpc_write(ctx, k, sz, rs, gi, rsl, sgp, smk, kc, id, ks, kp, (int32_t) NS);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    }

    std::vector<uint8_t>     k_ms(ggml_nbytes(k));
    std::vector<uint8_t> sz_ms(ggml_nelements(sz));
    std::vector<int32_t>     gi_ms(kv);
    memcpy(k_ms.data(), k->data, ggml_nbytes(k));
    memcpy(sz_ms.data(), sz->data, ggml_nbytes(sz));
    memcpy(gi_ms.data(), gi->data, kv*sizeof(int32_t));
    ggml_free(ctx);

    // per-seq reference: same values written one-shot in a single-seq cache, dequant per slot
    int mismatches = 0, pool_errs = 0;
    std::vector<float> got(C), ref(C);
    for (int64_t sb = 0; sb < NS; ++sb) {
        std::vector<uint8_t>     k_ref;
        std::vector<uint8_t> sz_ref;
        std::vector<int32_t>     gi_ref;
        // single-seq reference: write this seq's values one token per step (same staging as the batch path)
        {
            struct ggml_init_params ip2 = { (size_t) 64*1024*1024, NULL, false };
            struct ggml_context * c2 = ggml_init(ip2);
            struct ggml_tensor * k2  = ggml_new_tensor_2d(c2, GGML_TYPE_KPC4_1, C, kv);
            struct ggml_tensor * sz2 = ggml_new_tensor_2d(c2, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
            struct ggml_tensor * rs2 = ggml_new_tensor_2d(c2, GGML_TYPE_F16, C, KPC_GROUP);
            struct ggml_tensor * gi2 = ggml_new_tensor_2d(c2, GGML_TYPE_I32, kv, 1);
            struct ggml_tensor * rsl2 = ggml_new_tensor_2d(c2, GGML_TYPE_I32, KPC_GROUP, 1);
            struct ggml_tensor * sgp2 = ggml_new_tensor_1d(c2, GGML_TYPE_I32, 1);
            struct ggml_tensor * smk2 = ggml_new_tensor_1d(c2, GGML_TYPE_I32, 1);
            memset(k2->data, 0, ggml_nbytes(k2));   memset(sz2->data, 0, ggml_nbytes(sz2));
            memset(rs2->data, 0, ggml_nbytes(rs2)); memset(gi2->data, 0xFF, ggml_nbytes(gi2));
            memset(rsl2->data, 0, ggml_nbytes(rsl2));
            memset(sgp2->data, 0, ggml_nbytes(sgp2)); memset(smk2->data, 0, ggml_nbytes(smk2));
            for (int64_t p = 0; p < nsteps; ++p) {
                struct ggml_tensor * kc = ggml_new_tensor_2d(c2, GGML_TYPE_F32, C, 1);
                struct ggml_tensor * id = ggml_new_tensor_1d(c2, GGML_TYPE_I64, 1);
                struct ggml_tensor * ks = ggml_new_tensor_1d(c2, GGML_TYPE_I32, 1);
                struct ggml_tensor * kp = ggml_new_tensor_1d(c2, GGML_TYPE_I32, 1);
                float * kcd = (float *) kc->data;
                for (int64_t c = 0; c < C; ++c) kcd[c] = src_val_seq(sb, c, p);
                ((int64_t *) id->data)[0] = p;
                ((int32_t *) ks->data)[0] = 0;
                ((int32_t *) kp->data)[0] = (int32_t) p;
                struct ggml_tensor * out = ggml_kpc_write(c2, k2, sz2, rs2, gi2, rsl2, sgp2, smk2, kc, id, ks, kp, 1);
                struct ggml_cgraph * gf = ggml_new_graph(c2);
                ggml_build_forward_expand(gf, out);
                ggml_graph_compute_with_ctx(c2, gf, n_threads);
            }
            k_ref.resize(ggml_nbytes(k2));   memcpy(k_ref.data(), k2->data, ggml_nbytes(k2));
            sz_ref.resize(ggml_nelements(sz2)); memcpy(sz_ref.data(), sz2->data, ggml_nbytes(sz2));
            gi_ref.resize(kv);               memcpy(gi_ref.data(), gi2->data, kv*sizeof(int32_t));
            ggml_free(c2);
        }
        for (int64_t p = 0; p < nsteps; ++p) {
            const int64_t slot = sb*rsize + p;
            const int64_t pool = gi_ms[slot];
            if (pool < sb*band_size || pool >= (sb+1)*band_size) pool_errs++;
            deq_slot(C, slot, k_ms, sz_ms, gi_ms, got.data());
            deq_slot(C, p,    k_ref, sz_ref, gi_ref, ref.data());
            for (int64_t c = 0; c < C; ++c) {
                if (got[c] != ref[c]) { mismatches++; break; }
            }
        }
    }

    if (mismatches == 0 && pool_errs == 0) {
        printf("PASS contig-batch: %lld per-step writes x %d seqs, each bit-identical to single-seq\n",
               (long long) nsteps, (int) NS);
        return 0;
    }
    printf("FAIL contig-batch: %d cell mismatch(es), %d pool-band error(s)\n", mismatches, pool_errs);
    return 1;
}

// RoPE K-shift chain (build_rope_shift's KPC branch): dequant -> rope -> kpc_requant must
// dequant back to the roped values within quant tolerance
static int test_rope_shift(int n_threads) {
    const int64_t C = 128, kv = 64, head_dim = 64, n_head = 2;
    std::vector<uint8_t> k1; std::vector<uint8_t> sz1;
    build_cache(C, kv, kv, { kv }, n_threads, k1, sz1);          // contiguous: pool = slot/32

    struct ggml_init_params ip = { (size_t) 128*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    const int64_t ng = kv / KPC_GROUP;
    struct ggml_tensor * pk  = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, kv);
    struct ggml_tensor * sz  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
    struct ggml_tensor * gi  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv);
    memcpy(pk->data, k1.data(),  ggml_nbytes(pk));
    memcpy(sz->data, sz1.data(), ggml_nbytes(sz));
    for (int64_t i = 0; i < kv; ++i) {
        ((int32_t *) gi->data)[i]  = (int32_t)(i / KPC_GROUP);
        ((int32_t *) pos->data)[i] = -8;                        // uniform shift delta
    }

    ggml_tensor * kf    = ggml_cast(ctx, ggml_kpc_dequant(ctx, pk, sz, gi), GGML_TYPE_F32);
    ggml_tensor * k3    = ggml_reshape_3d(ctx, kf, head_dim, n_head, kv);
    ggml_tensor * roped = ggml_rope_ext(ctx, k3, pos, NULL, (int) head_dim, GGML_ROPE_TYPE_NEOX,
                                        0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * roped2 = ggml_reshape_2d(ctx, roped, C, kv);
    ggml_tensor * out    = ggml_kpc_requant(ctx, pk, sz, gi, roped2);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    std::vector<float> ref(C*kv);
    memcpy(ref.data(), roped2->data, C*kv*sizeof(float));       // the rope'd reference (requant didn't touch it)
    std::vector<uint8_t> k2(ggml_nbytes(pk));            memcpy(k2.data(),  pk->data, ggml_nbytes(pk));
    std::vector<uint8_t> sz2(ggml_nelements(sz));    memcpy(sz2.data(), sz->data, ggml_nbytes(sz));
    ggml_free(ctx);

    std::vector<float> deq;
    dequant_cache(C, kv, k2, sz2, n_threads, deq);              // pool = slot/32

    double se = 0.0, sref = 0.0, maxe = 0.0;
    for (int64_t i = 0; i < C*kv; ++i) {
        const double e = (double) deq[i] - (double) ref[i];
        se += e*e; sref += (double) ref[i]*(double) ref[i];
        maxe = std::max(maxe, std::fabs(e));
    }
    const double nmse = se / (sref > 0.0 ? sref : 1.0);
    if (nmse < 0.01) {
        printf("PASS rope-shift: requant(rope(dequant)) NMSE=%.5f maxerr=%.4f vs rope'd reference\n", nmse, maxe);
        return 0;
    }
    printf("FAIL rope-shift: NMSE=%.5f maxerr=%.4f\n", nmse, maxe);
    return 1;
}

// multi-stream RoPE K-shift: each stream has its own data slab and shift delta and must be
// bit-identical to an independent single-stream shift (catches cross-stream bleed)
static int test_rope_shift_multistream(int n_threads) {
    const int64_t C = 128, kv = 64, head_dim = 64, n_head = 2, NS = 2;
    const int64_t ng = kv / KPC_GROUP;
    const size_t  krow  = ggml_row_size(GGML_TYPE_KPC4_1, C);   // packed bytes per token row
    const size_t  szrow = (size_t) ng * KPC_SZ_GROUP_BYTES(C);  // scalezp int8 bytes per stream

    std::vector<uint8_t> k1; std::vector<uint8_t> sz1;
    build_cache(C, kv, kv, { kv }, n_threads, k1, sz1);         // pool = slot/32
    GGML_ASSERT(k1.size() == krow*kv && sz1.size() == szrow);

    // independent single-stream reference: dequant -> rope(shiftval) -> kpc_requant -> dequant, returns [C*kv]
    auto run_single = [&](int32_t shiftval, std::vector<float> & deq_out) {
        struct ggml_init_params ip = { (size_t) 128*1024*1024, NULL, false };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * pk  = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, kv);
        struct ggml_tensor * sz  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
        struct ggml_tensor * gi  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv);
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv);
        memcpy(pk->data, k1.data(),  ggml_nbytes(pk));
        memcpy(sz->data, sz1.data(), ggml_nbytes(sz));
        for (int64_t i = 0; i < kv; ++i) {
            ((int32_t *) gi->data)[i]  = (int32_t)(i / KPC_GROUP);
            ((int32_t *) pos->data)[i] = shiftval;
        }
        ggml_tensor * kf  = ggml_cast(ctx, ggml_kpc_dequant(ctx, pk, sz, gi), GGML_TYPE_F32);
        ggml_tensor * k3  = ggml_reshape_3d(ctx, kf, head_dim, n_head, kv);
        ggml_tensor * rp  = ggml_rope_ext(ctx, k3, pos, NULL, (int) head_dim, GGML_ROPE_TYPE_NEOX,
                                          0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * rp2 = ggml_reshape_2d(ctx, rp, C, kv);
        ggml_tensor * out = ggml_kpc_requant(ctx, pk, sz, gi, rp2);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        std::vector<uint8_t> kb(ggml_nbytes(pk));         memcpy(kb.data(),  pk->data, ggml_nbytes(pk));
        std::vector<uint8_t> szb(ggml_nelements(sz)); memcpy(szb.data(), sz->data, ggml_nbytes(sz));
        ggml_free(ctx);
        dequant_cache(C, kv, kb, szb, n_threads, deq_out);
    };

    const int32_t sh[NS] = { -8, -4 };                          // distinct per-stream shift deltas
    std::vector<float> ref[NS];
    for (int64_t s = 0; s < NS; ++s) run_single(sh[s], ref[s]);

    // combined non-unified run: 3D pk/sz/gi with NS streams, ONE rope over the flattened (stream,cell) token dim
    struct ggml_init_params ip = { (size_t) 256*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * pk  = ggml_new_tensor_3d(ctx, GGML_TYPE_KPC4_1, C, kv, NS);
    struct ggml_tensor * sz  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng, NS);
    struct ggml_tensor * gi  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kv, NS);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv*NS);
    for (int64_t s = 0; s < NS; ++s) {
        memcpy((char *) pk->data + s*pk->nb[2], k1.data(),  krow*kv);             // each stream = same base cache
        memcpy((char *) sz->data + s*sz->nb[2], sz1.data(), szrow);
        for (int64_t i = 0; i < kv; ++i) {
            ((int32_t *) gi->data)[s*kv + i]  = (int32_t)(i / KPC_GROUP);
            ((int32_t *) pos->data)[s*kv + i] = sh[s];

        }
    }
    ggml_tensor * kf  = ggml_cast(ctx, ggml_kpc_dequant(ctx, pk, sz, gi), GGML_TYPE_F32);
    ggml_tensor * k3  = ggml_reshape_3d(ctx, kf, head_dim, n_head, kv*NS);
    ggml_tensor * rp  = ggml_rope_ext(ctx, k3, pos, NULL, (int) head_dim, GGML_ROPE_TYPE_NEOX,
                                      0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * rp3 = ggml_reshape_3d(ctx, rp, C, kv, NS);
    ggml_tensor * out = ggml_kpc_requant(ctx, pk, sz, gi, rp3);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    std::vector<uint8_t> kb(ggml_nbytes(pk));         memcpy(kb.data(),  pk->data, ggml_nbytes(pk));
    std::vector<uint8_t> szb(ggml_nelements(sz)); memcpy(szb.data(), sz->data, ggml_nbytes(sz));
    ggml_free(ctx);

    double worst = 0.0;
    for (int64_t s = 0; s < NS; ++s) {
        std::vector<uint8_t> ks(kb.begin() + s*krow*kv, kb.begin() + (s+1)*krow*kv);
        std::vector<uint8_t> szs(szb.begin() + s*szrow, szb.begin() + (s+1)*szrow);
        std::vector<float> deq;
        dequant_cache(C, kv, ks, szs, n_threads, deq);
        for (int64_t i = 0; i < C*kv; ++i) worst = std::max(worst, (double) std::fabs(deq[i] - ref[s][i]));
    }
    double sep = 0.0;                                           // streams must actually differ (distinct shifts)
    for (int64_t i = 0; i < C*kv; ++i) sep = std::max(sep, (double) std::fabs(ref[0][i] - ref[1][i]));

    if (worst < 1e-6 && sep > 1e-3) {
        printf("PASS rope-shift-multistream: %lld streams bit-match independent single-stream shifts (maxdiff=%.1e, stream-sep=%.3f)\n",
               (long long) NS, worst, sep);
        return 0;
    }
    printf("FAIL rope-shift-multistream: maxdiff=%.1e (want <1e-6), stream-sep=%.3f (want >1e-3)\n", worst, sep);
    return 1;
}

// staging-slot virtualization: write NS seqs sequentially, L == NS slots (slot == seq) vs L == 1
// (each seq reuses slot 0 after the prior retires). cache content is slot-independent -> must be bit-identical
static int test_virtual_staging(int n_threads) {
    const int64_t C = 128, kv = 256, NS = 3, P = 40;   // P=40 -> last group (pos 32..39) stays OPEN
    const int64_t rsize = kv / NS;                     // disjoint physical region per seq

    auto build = [&](int64_t L, std::vector<uint8_t> & kout, std::vector<uint8_t> & szout, std::vector<int32_t> & giout) {
        struct ggml_init_params ip = { (size_t) 64*1024*1024, NULL, false };
        struct ggml_context * ctx = ggml_init(ip);
        const int64_t ng = kv / KPC_GROUP;
        struct ggml_tensor * k   = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, kv);
        struct ggml_tensor * sz  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
        struct ggml_tensor * gi  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kv, 1);
        struct ggml_tensor * rs  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, C, KPC_GROUP, L);
        struct ggml_tensor * rsl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPC_GROUP, L);
        struct ggml_tensor * sgp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
        struct ggml_tensor * smk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
        memset(k->data, 0, ggml_nbytes(k));     memset(sz->data, 0, ggml_nbytes(sz));   memset(gi->data, 0xFF, ggml_nbytes(gi));
        memset(rs->data, 0, ggml_nbytes(rs));   memset(rsl->data, 0, ggml_nbytes(rsl));
        memset(sgp->data, 0, ggml_nbytes(sgp)); memset(smk->data, 0, ggml_nbytes(smk));

        for (int64_t sb = 0; sb < NS; ++sb) {
            const int64_t slot = (L == NS) ? sb : 0;   // identity slots vs reuse-one-slot
            struct ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, P);
            struct ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, P);
            struct ggml_tensor * ks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
            struct ggml_tensor * kp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
            float   * kcd = (float   *) kc->data;
            int64_t * idd = (int64_t *) id->data;
            int32_t * ksd = (int32_t *) ks->data;
            int32_t * kpd = (int32_t *) kp->data;
            for (int64_t i = 0; i < P; ++i) {
                for (int64_t c = 0; c < C; ++c) kcd[i*C + c] = src_val_seq(sb, c, i);
                idd[i] = sb*rsize + i;                          // disjoint physical slots per seq
                ksd[i] = (int32_t)(sb | (slot << GGML_KPC_SLOT_SHIFT));          // pack staging slot in high bits
                kpd[i] = (int32_t) i;
            }
            struct ggml_tensor * out = ggml_kpc_write(ctx, k, sz, rs, gi, rsl, sgp, smk, kc, id, ks, kp, (int32_t) NS);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_graph_compute_with_ctx(ctx, gf, n_threads);
            if (L == 1) {
                ((int32_t *) smk->data)[0] = 0;
                ((int32_t *) sgp->data)[0] = 0;
            }
        }
        kout.resize(ggml_nbytes(k));        memcpy(kout.data(),  k->data,  ggml_nbytes(k));
        szout.resize(ggml_nelements(sz));   memcpy(szout.data(), sz->data, ggml_nbytes(sz));
        giout.resize(ggml_nelements(gi));   memcpy(giout.data(), gi->data, ggml_nbytes(gi));
        ggml_free(ctx);
    };

    std::vector<uint8_t> k2, k1;  std::vector<uint8_t> s2, s1;  std::vector<int32_t> g2, g1;
    build(NS, k2, s2, g2);   // reference: L == NS, slot == seq
    build(1,  k1, s1, g1);

    bool ok = (k1 == k2) && (g1 == g2) && (s1.size() == s2.size());
    if (ok) { for (size_t i = 0; i < s1.size(); ++i) if (s1[i] != s2[i]) { ok = false; break; } }
    printf("%s virtual-staging: %lld seqs in L=1 staging slot reuse == L=%lld bit-identical cache\n",
           ok ? "PASS" : "FAIL", (long long) NS, (long long) NS);
    return ok ? 0 : 1;
}

// pool re-encode rescue: continuing a group after its staging was dropped must requant prior
// members from int4 against the new joint scale, not leave them on the overwritten scale
static int test_rescue(int n_threads) {
    const int64_t C = 128, kv = 128;
    struct ggml_init_params ip = { (size_t) 64*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    const int64_t ng = kv / KPC_GROUP;
    struct ggml_tensor * k   = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, C, kv);
    struct ggml_tensor * sz  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(C), ng);
    struct ggml_tensor * rs  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, C, KPC_GROUP);
    struct ggml_tensor * gi  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kv, 1);
    struct ggml_tensor * rsl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPC_GROUP, 1);
    struct ggml_tensor * sgp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * smk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    memset(k->data, 0, ggml_nbytes(k));     memset(sz->data, 0, ggml_nbytes(sz));
    memset(rs->data, 0, ggml_nbytes(rs));   memset(gi->data, 0xFF, ggml_nbytes(gi));
    memset(rsl->data, 0, ggml_nbytes(rsl)); memset(sgp->data, 0, ggml_nbytes(sgp));
    memset(smk->data, 0, ggml_nbytes(smk));

    auto write_range = [&](int64_t p0, int64_t p1) {
        const int64_t cs = p1 - p0;
        struct ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, cs);
        struct ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, cs);
        struct ggml_tensor * ks = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cs);
        struct ggml_tensor * kp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cs);
        float * kcd = (float *) kc->data;
        for (int64_t i = 0; i < cs; ++i) {
            for (int64_t c = 0; c < C; ++c) kcd[i*C + c] = src_val(c, p0 + i);
            ((int64_t *) id->data)[i] = p0 + i;
            ((int32_t *) ks->data)[i] = 0;
            ((int32_t *) kp->data)[i] = (int32_t)(p0 + i);
        }
        struct ggml_tensor * out = ggml_kpc_write(ctx, k, sz, rs, gi, rsl, sgp, smk, kc, id, ks, kp, 1);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    };

    write_range(0, 70);                       // groups 0,1 complete, group 2 open (w0..w5 staged)
    ((int32_t *) smk->data)[0] = 0;           // drop the staging (retired seq / cleared open group)
    ((int32_t *) sgp->data)[0] = 0;
    write_range(70, 76);                      // continue group 2 without staging -> rescue must kick in

    std::vector<uint8_t> kb(ggml_nbytes(k));      memcpy(kb.data(),  k->data,  ggml_nbytes(k));
    std::vector<uint8_t> szb(ggml_nelements(sz)); memcpy(szb.data(), sz->data, ggml_nbytes(sz));
    ggml_free(ctx);

    std::vector<float> deq;
    dequant_cache(C, 76, kb, szb, n_threads, deq);

    // group 2's early members (slots 64..69) must reconstruct near their originals; one extra
    // requant step is allowed, garbage (old-scale nibbles under a new-scale slab) is not
    double se = 0.0, sref = 0.0, maxe = 0.0;
    for (int64_t t = 64; t < 70; ++t) {
        for (int64_t c = 0; c < C; ++c) {
            const float orig = src_val(c, t);
            const float e = fabsf(orig - deq[t*C + c]);
            se += (double) e*e; sref += (double) orig*orig;
            if (e > maxe) maxe = e;
        }
    }
    const double nmse = se / (sref > 0.0 ? sref : 1.0);
    if (nmse < 0.02 && maxe < 0.5) {
        printf("PASS rescue: pool re-encode without staging keeps prior members (NMSE=%.5f maxerr=%.4f)\n", nmse, maxe);
        return 0;
    }
    printf("FAIL rescue: prior members corrupted by unstaged pool re-encode (NMSE=%.5f maxerr=%.4f)\n", nmse, maxe);
    return 1;
}

// GQA grouping invariance: grouped (nth=1) and ungrouped (nth=4) sibling-head walks must be
// bit-identical (grouping only shares the K/V/mask walk, per-head math order is unchanged)
static int test_attn_gqa_grouping(void) {
    const int64_t DK = 64, DV = 64, NKV = 40, NH = 4;    // 1 kv head -> rk2 = 4
    const int64_t ng = (NKV + KPC_GROUP - 1) / KPC_GROUP;

    std::vector<uint8_t> k_os, sz_os;
    build_cache(DK, 64, NKV, { NKV }, 1, k_os, sz_os);

    auto run = [&](int n_threads, std::vector<float> & out) {
        struct ggml_init_params ip = { (size_t) 64*1024*1024, NULL, false };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * pk  = ggml_new_tensor_2d(ctx, GGML_TYPE_KPC4_1, DK, NKV);
        struct ggml_tensor * szt = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(DK), ng);
        struct ggml_tensor * qt  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, 1, NH, 1);
        struct ggml_tensor * vt  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, NKV, 1, 1);
        struct ggml_tensor * mk  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, NKV, 1);
        struct ggml_tensor * gx  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NKV, 1);
        memcpy(pk->data,  k_os.data(),  ggml_nbytes(pk));
        memcpy(szt->data, sz_os.data(), ggml_nbytes(szt));
        float * qd = (float *) qt->data;
        for (int64_t hh = 0; hh < NH; ++hh) {
            for (int64_t c = 0; c < DK; ++c) qd[hh*DK + c] = 0.5f * sinf(0.21f*(float) c + 0.6f + 0.3f*(float) hh);
        }
        ggml_fp16_t * vd = (ggml_fp16_t *) vt->data;
        for (int64_t i = 0; i < DV*NKV; ++i) vd[i] = ggml_fp32_to_fp16(0.4f * cosf(0.07f*(float)(i % DV) + 0.11f*(float)(i / DV)));
        ggml_fp16_t * mdat = (ggml_fp16_t *) mk->data;
        for (int64_t i = 0; i < NKV; ++i) mdat[i] = ggml_fp32_to_fp16(0.0f);
        int32_t * gxd = (int32_t *) gx->data;
        for (int64_t i = 0; i < NKV; ++i) gxd[i] = (int32_t)(i / KPC_GROUP);

        struct ggml_tensor * o = ggml_kpc_attn(ctx, qt, pk, szt, vt, mk, gx, NULL, 1.0f, 0.0f, 0.0f, 1);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        out.resize((size_t) ggml_nelements(o));
        memcpy(out.data(), o->data, ggml_nbytes(o));
        ggml_free(ctx);
    };

    std::vector<float> o_grouped, o_single;
    run(1, o_grouped);   // nth=1: enough items per thread -> heads grouped (gsz=rk2)
    run(4, o_single);    // nth=4: 1 item at gsz>1 < nth -> falls back to gsz=1 (ungrouped)

    if (o_grouped.size() == o_single.size() &&
        memcmp(o_grouped.data(), o_single.data(), o_single.size()*sizeof(float)) == 0) {
        printf("PASS gqa-grouping: grouped (nth=1) and ungrouped (nth=4) attention outputs bit-identical\n");
        return 0;
    }
    printf("FAIL gqa-grouping: grouped vs ungrouped attention outputs differ\n");
    return 1;
}

int main() {
    const int64_t C  = 128;
    const int64_t kv = 128;
    const int64_t N  = 70;                       // 2 full groups + a partial of 6
    const int     nt = 4;
    int failures = 0;

    const int64_t krow       = C / 2;
    const int64_t k_written  = N * krow;                                   // packed bytes for slots [0,N)
    const int64_t sz_written = ((N + KPC_GROUP - 1) / KPC_GROUP) * KPC_SZ_GROUP_BYTES(C);

    // one-shot reference cache
    std::vector<uint8_t>     k_os;
    std::vector<uint8_t> sz_os;
    build_cache(C, kv, N, { N }, nt, k_os, sz_os);
    std::vector<float> deq_os;
    dequant_cache(C, N, k_os, sz_os, nt, deq_os);
    double maxerr_os = 0.0;
    const double nmse_os = tile_nmse(C, N, deq_os, maxerr_os);

    // 1. group-aligned chunks must be bit-identical to one-shot
    std::vector<uint8_t>     k_ga;
    std::vector<uint8_t> sz_ga;
    build_cache(C, kv, N, { 32, 32, 6 }, nt, k_ga, sz_ga);
    int sz_diff = 0;
    for (int64_t i = 0; i < sz_written; ++i) {
        if (sz_os[i] != sz_ga[i]) {
            sz_diff++;
        }
    }
    if (memcmp(k_os.data(), k_ga.data(), k_written) != 0 || sz_diff != 0) {
        printf("FAIL group-aligned: chunked != one-shot (scalezp_mismatches=%d)\n", sz_diff);
        failures++;
    } else {
        printf("PASS group-aligned: bit-identical to one-shot over %lld slots (%d threads)\n",
               (long long) N, nt);
    }

    // 2. arbitrary mid-group chunks: differs from one-shot by f16 staging only, NMSE must not compound
    std::vector<uint8_t>     k_ar;
    std::vector<uint8_t> sz_ar;
    build_cache(C, kv, N, { 7, 25, 13, 19, 6 }, nt, k_ar, sz_ar);
    std::vector<float> deq_ar;
    dequant_cache(C, N, k_ar, sz_ar, nt, deq_ar);
    double maxerr_ar = 0.0;
    const double nmse_ar = tile_nmse(C, N, deq_ar, maxerr_ar);
    if (nmse_ar < 0.01 && nmse_ar <= 1.5*nmse_os + 1e-4) {
        printf("PASS no-compounding: arbitrary-chunk NMSE=%.5f vs one-shot %.5f (f16 staging only)\n",
               nmse_ar, nmse_os);
    } else {
        printf("FAIL no-compounding: arbitrary-chunk NMSE=%.5f vs one-shot %.5f (error accumulated)\n",
               nmse_ar, nmse_os);
        failures++;
    }

    // 3. one-shot round-trip reconstructs within int4-per-channel bounds
    if (nmse_os < 0.1 && maxerr_os < 0.3) {
        printf("PASS round-trip: one-shot NMSE=%.5f maxerr=%.4f (int4 per-channel)\n", nmse_os, maxerr_os);
    } else {
        printf("FAIL round-trip: one-shot NMSE=%.5f maxerr=%.4f out of bounds\n", nmse_os, maxerr_os);
        failures++;
    }

    // 4. fused per-channel attention (ggml_kpc_attn) vs a full-precision reference
    failures += test_attn(nt);

    // 5. fused attention with logit-softcap, attention-sinks and ALiBi vs a full-precision reference
    failures += test_attn_extras(nt, 30.0f, false, 0.0f, "softcap");
    failures += test_attn_extras(nt,  0.0f, true,  0.0f, "sinks");
    failures += test_attn_extras(nt,  8.0f, true,  0.0f, "softcap+sinks");
    failures += test_attn_extras(nt,  0.0f, false, 8.0f, "alibi");

    // 6. scatter-write: scattered physical slots decode == contiguous
    failures += test_scatter(nt);

    // 7. scatter-dequant via 3-arg op: group_index dequant == contiguous reference
    failures += test_scatter_dequant_op(nt);

    // 8. multi-seq interleaved write: disjoint pool bands, each seq bit-identical
    failures += test_multiseq(nt);

    // 9. continuous-batch per-step writes: each live group keeps its own scales
    failures += test_contig_batch_steps(nt);

    // 10. staging-slot virtualization: slot reuse across retired seqs -> bit-identical cache
    failures += test_virtual_staging(nt);

    // 11. RoPE K-shift chain: dequant -> rope -> kpc_requant matches the rope'd reference
    failures += test_rope_shift(nt);

    // 12. multi-stream RoPE K-shift: each stream bit-matches a single-stream shift
    failures += test_rope_shift_multistream(nt);

    // 13. pool re-encode rescue: group continued after staging drop keeps prior members
    failures += test_rescue(nt);

    // 14. GQA grouping: grouped sibling-head walk is bit-identical to per-head execution
    failures += test_attn_gqa_grouping();

    if (failures) {
        printf("test-kpc: %d failure(s)\n", failures);
        return 1;
    }
    printf("test-kpc: all passed\n");
    return 0;
}
