// CPU kernels for the KPC4_1 per-channel int4 K-cache ops: 4-bit per-channel nibbles (row = C/2 bytes)
// + a companion scale/zp tensor, one (scale, zp) per channel per 32-token group.
#include "ops.h"
#include "ggml-cpu-impl.h"
#include "vec.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// wire-format constants are shared via ggml.h (GGML_KPC_*); local aliases keep the kernel code short
#define KPC_GROUP      GGML_KPC_GROUP        // token group size
#define KPC_SLOT_SHIFT GGML_KPC_SLOT_SHIFT   // kpc_seq packs staging slot in high bits, seq_id in low bits
#define KPC_SEQ_MASK   GGML_KPC_SEQ_MASK
static_assert(KPC_GROUP <= 32, "open-group members bitmask is uint32_t");

#define KPC_F16_NEG_INF ((ggml_fp16_t) 0xFC00)   // the unique fp16 encoding of -INFINITY (mask sentinel)

// dequant one packed nibble (c even -> low half, c odd -> high) with per-channel scale s, zero-point z
static inline float ggml_kpc_deq(uint8_t nib_byte, int c, float s, float z) {
    const int q = (c & 1) ? (nib_byte >> 4) : (nib_byte & 0x0F);
    return q * s + z;
}

// inverse of ggml_kpc_deq: quantize value v at per-channel scale s, zero-point z into channel c's nibble.
// round-half-up (int)(x+0.5f) and the [0,15] clamp must match every K write/requant site (s != 0 enforced upstream).
static inline void kpc_pack_nibble(uint8_t * row, int64_t c, float v, float s, float z) {
    int qv = (int)((v - z)/s + 0.5f);
    if (qv < 0)  qv = 0;
    if (qv > 15) qv = 15;
    uint8_t * b = &row[c/2];
    *b = (c & 1) ? ((*b & 0x0F) | (uint8_t)(qv << 4)) : ((*b & 0xF0) | (uint8_t) qv);
}

// scalezp slab per group: [fp16 super_scale_s][fp16 zp_min][fp16 super_scale_z][C x u8 q_s][C x u8 q_z];
// scale[c]=q_s[c]*super_scale_s, zp[c]=zp_min+q_z[c]*super_scale_z.
#define KPC_SZ_GROUP_BYTES(C) GGML_KPC_SZ_GROUP_BYTES(C)
#define KPC_SZ_QMAX 255

static inline void kpc_sz_super_read(const uint8_t * slab, float * ss, float * zmin, float * sz) {
    ggml_fp16_t h;
    memcpy(&h, slab + 0, 2); *ss   = GGML_CPU_FP16_TO_FP32(h);
    memcpy(&h, slab + 2, 2); *zmin = GGML_CPU_FP16_TO_FP32(h);
    memcpy(&h, slab + 4, 2); *sz   = GGML_CPU_FP16_TO_FP32(h);
}

// encode per-channel float scale[C] (>=0) and zp[C] (signed) into one group's int8 slab
static inline void kpc_sz_encode(const float * scale, const float * zp, int64_t C, uint8_t * slab) {
    float smax = 0.0f;
    for (int64_t c = 0; c < C; ++c) { if (scale[c] > smax) smax = scale[c]; }
    float ss = smax / (float) KPC_SZ_QMAX; if (ss == 0.0f) ss = 1.0f;
    float zmn = INFINITY, zmx = -INFINITY;
    for (int64_t c = 0; c < C; ++c) { if (zp[c] < zmn) zmn = zp[c]; if (zp[c] > zmx) zmx = zp[c]; }
    float sz = (zmx - zmn) / (float) KPC_SZ_QMAX; if (sz == 0.0f) sz = 1.0f;
    ggml_fp16_t h;
    h = GGML_CPU_FP32_TO_FP16(ss);  memcpy(slab + 0, &h, 2);
    h = GGML_CPU_FP32_TO_FP16(zmn); memcpy(slab + 2, &h, 2);
    h = GGML_CPU_FP32_TO_FP16(sz);  memcpy(slab + 4, &h, 2);
    uint8_t * qs = slab + 6, * qz = slab + 6 + C;
    for (int64_t c = 0; c < C; ++c) {
        int s = (int)(scale[c] / ss + 0.5f);          if (s < 0) s = 0; if (s > KPC_SZ_QMAX) s = KPC_SZ_QMAX;
        int z = (int)((zp[c] - zmn) / sz + 0.5f);      if (z < 0) z = 0; if (z > KPC_SZ_QMAX) z = KPC_SZ_QMAX;
        qs[c] = (uint8_t) s; qz[c] = (uint8_t) z;
    }
}

// decode a single channel's (scale, zp) from a group slab, given the group's super-params
static inline void kpc_sz_decode1(const uint8_t * slab, int64_t C, int64_t c, float ss, float zmin, float sz,
                                  float * scale, float * zp) {
    *scale = slab[6 + c] * ss;
    *zp    = zmin + slab[6 + C + c] * sz;
}

// dequant packed nibbles + scalezp -> F16 [C, T, NS]. parallelize over flattened (stream, token).
// NS=1 unified / n_stream non-unified; all addressing via nb strides.
void ggml_compute_forward_kpc_dequant(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const int ith = params->ith;
    const int nth = params->nth;
    const struct ggml_tensor * pk = dst->src[0];
    const struct ggml_tensor * sz = dst->src[1];
    const struct ggml_tensor * gi = dst->src[2];   // optional I32 [T, NS]: per-slot scalezp pool index
    const int64_t C  = dst->ne[0], T = dst->ne[1], NS = dst->ne[2];
    const int64_t TOT = T * NS;
    const int64_t i0 = (TOT * ith) / nth, i1 = (TOT * (ith + 1)) / nth;
    std::vector<float> dsc((size_t) C), dzp((size_t) C);   // cached across same-group tokens
    int64_t cur_s = -1, cur_g = -1;
    for (int64_t idx = i0; idx < i1; ++idx) {
        const int64_t s = idx / T, t = idx % T;
        const uint8_t   * szb = (const uint8_t *)((const char *) sz->data + s * sz->nb[2]);   // stream slab base
        const int32_t   * gid = gi ? (const int32_t *)((const char *) gi->data + s * gi->nb[1]) : NULL;
        int64_t g = gid ? (int64_t) gid[t] : t / KPC_GROUP;
        if (g < 0 || g >= sz->ne[1]) g = 0;   // free/unwritten cells (gid -1): clamp for indexing only
        if (s != cur_s || g != cur_g) {       // consecutive tokens usually share a group; decode its table once
            const uint8_t * slab = szb + g * sz->nb[1];
            float ss, zmin, szc; kpc_sz_super_read(slab, &ss, &zmin, &szc);
            for (int64_t c = 0; c < C; ++c) {
                kpc_sz_decode1(slab, C, c, ss, zmin, szc, &dsc[c], &dzp[c]);
            }
            cur_s = s; cur_g = g;
        }
        const uint8_t * row  = (const uint8_t *)((const char *)pk->data + s * pk->nb[2] + t * pk->nb[1]);
        ggml_fp16_t   * drow = (ggml_fp16_t *)((char *)dst->data + s * dst->nb[2] + t * dst->nb[1]);
        for (int64_t c = 0; c < C; ++c) {
            drow[c] = GGML_CPU_FP32_TO_FP16(ggml_kpc_deq(row[c / 2], (int)c, dsc[c], dzp[c]));
        }
    }
}

// requant roped f32 K [C, n_kv, NS] into packed int4 + scalezp, keeping the existing group_index.
// used by the RoPE K-shift; (stream,pool) work items, streams disjoint so they never race.
void ggml_compute_forward_kpc_requant(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const int ith = params->ith;
    const int nth = params->nth;
    const struct ggml_tensor * roped = dst->src[0];   // f32 [C, n_kv, NS] (rotated K)
    const struct ggml_tensor * gi    = dst->src[1];   // i32 [n_kv, NS]    (cell -> pool)
    struct ggml_tensor       * sz    = dst->src[2];   // f16 [2C, ng_max, NS] (in-place)

    int32_t pr[16];
    memcpy(pr, dst->op_params, sizeof(pr));
    const int64_t ng_max = pr[0];
    const int64_t C    = roped->ne[0];
    const int64_t NKV  = roped->ne[1];
    const int64_t NS   = roped->ne[2];               // stream dim (1 unified, n_stream non-unified)

    std::vector<float> mn((size_t) C), mx((size_t) C), gsc((size_t) C), gzp((size_t) C), nsc((size_t) C), nzp((size_t) C);
    // bucket cells by pool in one scan; (s, p) round-robin keeps pool ownership race-free
    std::vector<std::vector<int64_t>> bucket((size_t) ng_max);
    for (int64_t s = 0; s < NS; ++s) {
        const int32_t * gid = (const int32_t *)((const char *) gi->data  + s * gi->nb[1]);
        uint8_t       * szd = (uint8_t *)      ((char *)       sz->data  + s * sz->nb[2]);
        uint8_t       * kd  = (uint8_t *)      ((char *)       dst->data + s * dst->nb[2]);
        auto rv = [&](int64_t c, int64_t i) -> float {
            return *(const float *)((const char *)roped->data + s*roped->nb[2] + i*roped->nb[1] + c*roped->nb[0]);
        };

        for (auto & b : bucket) {
            b.clear();
        }
        for (int64_t i = 0; i < NKV; ++i) {
            const int64_t p = gid[i];
            if (p < 0 || p >= ng_max) continue;             // free (-1) or out-of-range
            if ((int) ((s*ng_max + p) % nth) != ith) continue;
            bucket[(size_t) p].push_back(i);
        }

        for (int64_t p = 0; p < ng_max; ++p) {
            const auto & cells = bucket[(size_t) p];
            if (cells.empty()) continue;                    // pool unused

            for (int64_t c = 0; c < C; ++c) { mn[c] = INFINITY; mx[c] = -INFINITY; }
            for (const int64_t i : cells) {
                for (int64_t c = 0; c < C; ++c) {
                    const float v = rv(c, i);
                    if (v < mn[c]) mn[c] = v;
                    if (v > mx[c]) mx[c] = v;
                }
            }

            uint8_t * slab = szd + p*sz->nb[1];
            for (int64_t c = 0; c < C; ++c) {
                float scale = (mx[c] - mn[c]) / 15.0f;
                if (scale == 0.0f) scale = 1.0f;
                gsc[c] = scale; gzp[c] = mn[c];
            }
            kpc_sz_encode(gsc.data(), gzp.data(), C, slab);
            {                                               // decode the int8-rounded effective table once
                float ss, zmin, szc; kpc_sz_super_read(slab, &ss, &zmin, &szc);
                for (int64_t c = 0; c < C; ++c) {
                    kpc_sz_decode1(slab, C, c, ss, zmin, szc, &nsc[c], &nzp[c]);
                    if (nsc[c] == 0.0f) nsc[c] = 1.0f;      // tiny scale can underflow to 0
                }
            }
            for (const int64_t i : cells) {                 // repack each cell against the decoded scale
                uint8_t * row = kd + i*dst->nb[1];
                for (int64_t c = 0; c < C; ++c) {
                    kpc_pack_nibble(row, c, rv(c, i), nsc[c], nzp[c]);
                }
            }
        }
    }
}

// ---- SIMD helpers + fused attention ----
// x86 (GCC/Clang/MSVC): AVX2/FMA/F16C kernels, runtime-selected. aarch64: NEON (baseline). Others: scalar.
#if (defined(__GNUC__) || defined(_MSC_VER)) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#define KPC_X86 1
#if defined(__GNUC__)
#define KPC_TGT_AVX2 __attribute__((target("avx2,fma,f16c")))   // per-fn AVX2 in a baseline TU (gcc/clang)
#else
#define KPC_TGT_AVX2                                            // MSVC emits AVX2 intrinsics without a per-fn attr
#endif

#if defined(__GNUC__)
static inline bool kpc_have_avx2() {
    static int v = -1;
    if (v < 0) v = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) ? 1 : 0;
    return v != 0;
}
#else // MSVC: cpuid + xgetbv (no __builtin_cpu_supports)
#include <intrin.h>
static inline bool kpc_have_avx2() {
    static int v = -1;
    if (v < 0) {
        int info[4]; __cpuid(info, 0); const int nids = info[0];
        bool fma = false, avx = false, f16c = false, osxsave = false, avx2 = false;
        if (nids >= 1) { __cpuidex(info, 1, 0); fma = (info[2]&(1<<12))!=0; avx = (info[2]&(1<<28))!=0; f16c = (info[2]&(1<<29))!=0; osxsave = (info[2]&(1<<27))!=0; }
        if (nids >= 7) { __cpuidex(info, 7, 0); avx2 = (info[1]&(1<<5))!=0; }
        const bool os_ymm = osxsave && avx && ((_xgetbv(0) & 0x6) == 0x6);   // XMM+YMM saved by OS
        v = (avx2 && fma && f16c && os_ymm) ? 1 : 0;   // kernels are compiled with avx2,fma,f16c
    }
    return v != 0;
}
#endif

KPC_TGT_AVX2
static int32_t kpc_dot_i8_avx2(const int8_t * qs8, const uint8_t * krow, int64_t DK) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    const __m128i mlo  = _mm_set1_epi8(0x0F);
    for (int64_t d = 0; d < DK; d += 32) {
        const __m128i bytes = _mm_loadu_si128((const __m128i *)(krow + d/2));   // 16 bytes = 32 nibbles
        const __m128i lo = _mm_and_si128(bytes, mlo);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mlo);
        const __m256i nib = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)); // [lo0,hi0,...]
        const __m256i q   = _mm256_loadu_si256((const __m256i *)(qs8 + d));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(nib, q), ones)); // u8(0-15)*s8
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

KPC_TGT_AVX2
static void kpc_vmad_f16_avx2(float * VKQ, const ggml_fp16_t * vh, float vs, int64_t DV) {
    const __m256 vvs = _mm256_set1_ps(vs);
    for (int64_t d = 0; d < DV; d += 8)
        _mm256_storeu_ps(VKQ + d, _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh + d))), vvs, _mm256_loadu_ps(VKQ + d)));
}

// fused q4_1-V dequant+mad: VKQ += dequant(q4_1 block)*vs (no V32 buffer / to_float).
// block = {f16 d, f16 m, u8 qs[16]} (20 B); lo nibbles -> [0,16), hi -> [16,32).
KPC_TGT_AVX2
static void kpc_vmad_q4_1_avx2(float * VKQ, const uint8_t * vb, float vs, int64_t DV) {
    const __m128i mlo = _mm_set1_epi8(0x0F);
    for (int64_t b = 0; b < DV/32; ++b) {
        const uint8_t * blk = vb + b*20;
        const __m256 vd  = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 0)));
        const __m256 vm  = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 2)));
        const __m256 vvs = _mm256_set1_ps(vs);
        const __m128i bytes = _mm_loadu_si128((const __m128i *)(blk + 4));
        const __m128i nib[2] = { _mm_and_si128(bytes, mlo), _mm_and_si128(_mm_srli_epi16(bytes, 4), mlo) };
        float * out = VKQ + b*32;
        for (int h = 0; h < 2; ++h) {                       // lo/hi -> slot ranges
            const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(nib[h]));
            const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(nib[h], 8)));
            float * o = out + h*16;
            // deq = q*d + m (exact, matches to_float), then VKQ += deq*vs
            _mm256_storeu_ps(o + 0, _mm256_fmadd_ps(_mm256_fmadd_ps(f0, vd, vm), vvs, _mm256_loadu_ps(o + 0)));
            _mm256_storeu_ps(o + 8, _mm256_fmadd_ps(_mm256_fmadd_ps(f1, vd, vm), vvs, _mm256_loadu_ps(o + 8)));
        }
    }
}

// fused q3v_1-V dequant+mad: VKQ += dequant(q3v_1 block)*vs (no V32 buffer / to_float).
// block = {f16 d, f16 m, u8 qs[12]} (16 B), 32 vals at 3 bits; 8 vals = 3 bytes, val k = (window >> 3k) & 7.
KPC_TGT_AVX2
static void kpc_vmad_q3v_1_avx2(float * VKQ, const uint8_t * vb, float vs, int64_t DV) {
    const __m256i shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256i m7     = _mm256_set1_epi32(7);
    for (int64_t b = 0; b < DV/32; ++b) {
        const uint8_t * blk = vb + b*16;
        const __m256 vd  = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 0)));
        const __m256 vm  = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 2)));
        const __m256 vvs = _mm256_set1_ps(vs);
        const uint8_t * qs = blk + 4;
        float * out = VKQ + b*32;
        for (int g = 0; g < 4; ++g) {                       // 4 groups of 8
            const uint32_t w = (uint32_t)qs[g*3] | ((uint32_t)qs[g*3+1] << 8) | ((uint32_t)qs[g*3+2] << 16);
            const __m256 f = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)w), shifts), m7));
            float * o = out + g*8;
            _mm256_storeu_ps(o, _mm256_fmadd_ps(_mm256_fmadd_ps(f, vd, vm), vvs, _mm256_loadu_ps(o)));
        }
    }
}

// fused q3v_2-V dequant+mad. q3v_2 = 128-elem superblock (62 B): [f16 super_d, m_min, super_m]
// [u8 qd[4]][u8 qm[4]][u8 qs[48]]; sub-block d=qd*super_d, m=m_min+qm*super_m, same 3-byte unpack.
KPC_TGT_AVX2
static void kpc_vmad_q3v_2_avx2(float * VKQ, const uint8_t * vb, float vs, int64_t DV) {
    const __m256i shifts = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256i m7     = _mm256_set1_epi32(7);
    const __m256  vvs    = _mm256_set1_ps(vs);
    for (int64_t b = 0; b < DV/128; ++b) {
        const uint8_t * blk = vb + b*62;
        const float sd = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 0));
        const float mm = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 2));
        const float sm = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 4));
        const uint8_t * qd = blk + 6;
        const uint8_t * qm = blk + 10;
        const uint8_t * qs = blk + 14;
        float * out = VKQ + b*128;
        for (int s = 0; s < 4; ++s) {                       // 4 sub-blocks
            const __m256 vd = _mm256_set1_ps(qd[s]*sd);
            const __m256 vm = _mm256_set1_ps(mm + qm[s]*sm);
            const uint8_t * qsb = qs + s*12;
            float * o = out + s*32;
            for (int g = 0; g < 4; ++g) {
                const uint32_t w = (uint32_t)qsb[g*3] | ((uint32_t)qsb[g*3+1] << 8) | ((uint32_t)qsb[g*3+2] << 16);
                const __m256 f = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)w), shifts), m7));
                float * oo = o + g*8;
                _mm256_storeu_ps(oo, _mm256_fmadd_ps(_mm256_fmadd_ps(f, vd, vm), vvs, _mm256_loadu_ps(oo)));
            }
        }
    }
}

// per-group Q prescale (DK multiple of 32): fold f16 per-channel scale into Q -> qsf, accumulate zp
// correction, requant qsf -> int8. cvtps_epi32 round-to-nearest matches lrintf (bit-identical to scalar).
KPC_TGT_AVX2
static void kpc_prescale_avx2(const float * pq, const float * sc, const float * zp,
                                   int64_t DK, float * qsf, int8_t * qs8, float * out_dqg, float * out_corr) {
    __m256 vmax = _mm256_setzero_ps();
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (int64_t d = 0; d < DK; d += 8) {
        const __m256 vq  = _mm256_loadu_ps(pq + d);
        const __m256 vsc = _mm256_loadu_ps(sc + d);
        const __m256 vv  = _mm256_mul_ps(vq, vsc);
        _mm256_storeu_ps(qsf + d, vv);
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(vv, absmask));
    }
    __m128 mxv = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    mxv = _mm_max_ps(mxv, _mm_movehl_ps(mxv, mxv));
    mxv = _mm_max_ss(mxv, _mm_shuffle_ps(mxv, mxv, 0x1));
    const float amax = _mm_cvtss_f32(mxv);
    // corr summed sequentially (scalar) to stay bit-identical to the reference -- SIMD reorder/FMA shifts it.
    float corr = 0.0f;
    for (int64_t d = 0; d < DK; ++d) corr += pq[d] * zp[d];
    *out_corr = corr;
    const float dqg = amax > 0.0f ? amax / 127.0f : 1.0f;
    *out_dqg = dqg;
    const __m256  vidq = _mm256_set1_ps(1.0f / dqg);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t d = 0; d < DK; d += 32) {
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(qsf + d +  0), vidq));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(qsf + d +  8), vidq));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(qsf + d + 16), vidq));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(qsf + d + 24), vidq));
        __m256i s  = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        _mm256_storeu_si256((__m256i *)(qs8 + d), _mm256_permutevar8x32_epi32(s, perm));
    }
}
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define KPC_ARM 1
// NEON (aarch64) counterparts of the AVX2 kernels. NEON is baseline on aarch64, so no runtime check.

static int32_t kpc_dot_i8_neon(const int8_t * qs8, const uint8_t * krow, int64_t DK) {
    // dual accumulators: break the loop-carried dep chain so the M-series NEON pipes overlap.
    int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
    for (int64_t d = 0; d < DK; d += 32) {
        const uint8x16_t bytes = vld1q_u8(krow + d/2);                 // 16 bytes = 32 nibbles
        const uint8x16_t lo = vandq_u8(bytes, vdupq_n_u8(0x0F));
        const uint8x16_t hi = vshrq_n_u8(bytes, 4);
        const int8x16_t  n0 = vreinterpretq_s8_u8(vzip1q_u8(lo, hi));   // channels d..d+15  (lo0,hi0,lo1,hi1,...)
        const int8x16_t  n1 = vreinterpretq_s8_u8(vzip2q_u8(lo, hi));   // channels d+16..d+31
        const int8x16_t  q0 = vld1q_s8(qs8 + d);
        const int8x16_t  q1 = vld1q_s8(qs8 + d + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        // nibbles are 0..15 -> identical under signed/unsigned i8, so SDOT (i8.i8, 4-way reduce per
        // lane, single instruction) is a bit-exact drop-in for the vmull/vmlal/vpadal sequence below.
        acc0 = vdotq_s32(acc0, q0, n0);
        acc1 = vdotq_s32(acc1, q1, n1);
#else
        // pre-ARMv8.2 fallback: widening multiply + pairwise-add accumulate.
        int16x8_t p0 = vmull_s8(vget_low_s8(q0), vget_low_s8(n0));
        p0 = vmlal_s8(p0, vget_high_s8(q0), vget_high_s8(n0));
        int16x8_t p1 = vmull_s8(vget_low_s8(q1), vget_low_s8(n1));
        p1 = vmlal_s8(p1, vget_high_s8(q1), vget_high_s8(n1));
        acc0 = vpadalq_s16(acc0, p0);
        acc1 = vpadalq_s16(acc1, p1);
#endif
    }
    return vaddvq_s32(vaddq_s32(acc0, acc1));
}

static void kpc_vmad_f16_neon(float * VKQ, const ggml_fp16_t * vh, float vs, int64_t DV) {
    const float32x4_t vvs = vdupq_n_f32(vs);
    for (int64_t d = 0; d < DV; d += 4) {
        const float32x4_t f = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t *)(vh + d))));
        vst1q_f32(VKQ + d, vmlaq_f32(vld1q_f32(VKQ + d), f, vvs));
    }
}

// VKQ += dequant(q4_1 block)*vs; block = {f16 d, f16 m, u8 qs[16]} (20 B); lo -> [0,16), hi -> [16,32).
static void kpc_vmad_q4_1_neon(float * VKQ, const uint8_t * vb, float vs, int64_t DV) {
    const float32x4_t vvs = vdupq_n_f32(vs);
    for (int64_t b = 0; b < DV/32; ++b) {
        const uint8_t * blk = vb + b*20;
        const float32x4_t vd = vdupq_n_f32(GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 0)));
        const float32x4_t vm = vdupq_n_f32(GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 2)));
        const uint8x16_t bytes = vld1q_u8(blk + 4);
        const uint8x16_t nib[2] = { vandq_u8(bytes, vdupq_n_u8(0x0F)), vshrq_n_u8(bytes, 4) };
        float * out = VKQ + b*32;
        for (int h = 0; h < 2; ++h) {                       // h=0 lo -> [0,16), h=1 hi -> [16,32)
            const uint16x8_t w0 = vmovl_u8(vget_low_u8(nib[h]));
            const uint16x8_t w1 = vmovl_u8(vget_high_u8(nib[h]));
            const float32x4_t f[4] = {
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(w0))),  vcvtq_f32_u32(vmovl_u16(vget_high_u16(w0))),
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(w1))),  vcvtq_f32_u32(vmovl_u16(vget_high_u16(w1))),
            };
            float * o = out + h*16;
            for (int k = 0; k < 4; ++k)                      // deq = q*d + m, VKQ += deq*vs
                vst1q_f32(o + k*4, vmlaq_f32(vld1q_f32(o + k*4), vmlaq_f32(vm, f[k], vd), vvs));
        }
    }
}

// VKQ += dequant(q3v_1 block)*vs; 32 vals at 3 bits, val k = (window >> 3k) & 7 (8 vals = 3 bytes).
static void kpc_vmad_q3v_1_neon(float * VKQ, const uint8_t * vb, float vs, int64_t DV) {
    const float32x4_t vvs = vdupq_n_f32(vs);
    static const int32_t sh0a[4] = {0,-3,-6,-9}, sh1a[4] = {-12,-15,-18,-21};   // neg shift -> right shift
    const int32x4_t sh0 = vld1q_s32(sh0a), sh1 = vld1q_s32(sh1a);
    const uint32x4_t m7 = vdupq_n_u32(7);
    for (int64_t b = 0; b < DV/32; ++b) {
        const uint8_t * blk = vb + b*16;
        const float32x4_t vd = vdupq_n_f32(GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 0)));
        const float32x4_t vm = vdupq_n_f32(GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 2)));
        const uint8_t * qs = blk + 4;
        float * out = VKQ + b*32;
        for (int g = 0; g < 4; ++g) {
            const uint32_t w = (uint32_t)qs[g*3] | ((uint32_t)qs[g*3+1] << 8) | ((uint32_t)qs[g*3+2] << 16);
            const uint32x4_t wv = vdupq_n_u32(w);
            const float32x4_t f0 = vcvtq_f32_u32(vandq_u32(vshlq_u32(wv, sh0), m7));
            const float32x4_t f1 = vcvtq_f32_u32(vandq_u32(vshlq_u32(wv, sh1), m7));
            float * o = out + g*8;
            vst1q_f32(o + 0, vmlaq_f32(vld1q_f32(o + 0), vmlaq_f32(vm, f0, vd), vvs));
            vst1q_f32(o + 4, vmlaq_f32(vld1q_f32(o + 4), vmlaq_f32(vm, f1, vd), vvs));
        }
    }
}

// VKQ += dequant(q3v_2 superblock)*vs; 128 elems, 4 sub-blocks, sub d=qd*super_d, m=m_min+qm*super_m.
static void kpc_vmad_q3v_2_neon(float * VKQ, const uint8_t * vb, float vs, int64_t DV) {
    const float32x4_t vvs = vdupq_n_f32(vs);
    static const int32_t sh0a[4] = {0,-3,-6,-9}, sh1a[4] = {-12,-15,-18,-21};
    const int32x4_t sh0 = vld1q_s32(sh0a), sh1 = vld1q_s32(sh1a);
    const uint32x4_t m7 = vdupq_n_u32(7);
    for (int64_t b = 0; b < DV/128; ++b) {
        const uint8_t * blk = vb + b*62;
        const float sd = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 0));
        const float mm = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 2));
        const float sm = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 4));
        const uint8_t * qd = blk + 6, * qm = blk + 10, * qs = blk + 14;
        float * out = VKQ + b*128;
        for (int s = 0; s < 4; ++s) {
            const float32x4_t vd = vdupq_n_f32(qd[s]*sd);
            const float32x4_t vm = vdupq_n_f32(mm + qm[s]*sm);
            const uint8_t * qsb = qs + s*12;
            float * o = out + s*32;
            for (int g = 0; g < 4; ++g) {
                const uint32_t w = (uint32_t)qsb[g*3] | ((uint32_t)qsb[g*3+1] << 8) | ((uint32_t)qsb[g*3+2] << 16);
                const uint32x4_t wv = vdupq_n_u32(w);
                const float32x4_t f0 = vcvtq_f32_u32(vandq_u32(vshlq_u32(wv, sh0), m7));
                const float32x4_t f1 = vcvtq_f32_u32(vandq_u32(vshlq_u32(wv, sh1), m7));
                float * oo = o + g*8;
                vst1q_f32(oo + 0, vmlaq_f32(vld1q_f32(oo + 0), vmlaq_f32(vm, f0, vd), vvs));
                vst1q_f32(oo + 4, vmlaq_f32(vld1q_f32(oo + 4), vmlaq_f32(vm, f1, vd), vvs));
            }
        }
    }
}

// per-group Q prescale (DK multiple of 32): qsf = pq*scale; amax; corr (sequential, bit-identical); requant qsf -> int8.
static void kpc_prescale_neon(const float * pq, const float * sc, const float * zp,
                              int64_t DK, float * qsf, int8_t * qs8, float * out_dqg, float * out_corr) {
    float32x4_t vmax = vdupq_n_f32(0.0f);
    for (int64_t d = 0; d < DK; d += 4) {
        const float32x4_t vq  = vld1q_f32(pq + d);
        const float32x4_t vsc = vld1q_f32(sc + d);
        const float32x4_t vv  = vmulq_f32(vq, vsc);
        vst1q_f32(qsf + d, vv);
        vmax = vmaxq_f32(vmax, vabsq_f32(vv));
    }
    const float amax = vmaxvq_f32(vmax);
    float corr = 0.0f;
    for (int64_t d = 0; d < DK; ++d) corr += pq[d] * zp[d];   // sequential -> bit-identical to scalar
    *out_corr = corr;
    const float dqg = amax > 0.0f ? amax / 127.0f : 1.0f;
    *out_dqg = dqg;
    const float32x4_t vidq = vdupq_n_f32(1.0f / dqg);
    for (int64_t d = 0; d < DK; d += 8) {
        const int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(qsf + d + 0), vidq));   // round-to-nearest-even
        const int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(qsf + d + 4), vidq));
        const int16x8_t  s16 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        vst1_s8(qs8 + d, vqmovn_s16(s16));
    }
}
#endif // KPC_X86 / KPC_ARM

// int8(qs8) . unpacked-int4(krow) dot
static inline int32_t kpc_dot_i8(const int8_t * qs8, const uint8_t * krow, int64_t DK) {
#ifdef KPC_X86
    if ((DK & 31) == 0 && kpc_have_avx2()) return kpc_dot_i8_avx2(qs8, krow, DK);
#elif defined(KPC_ARM)
    if ((DK & 31) == 0) return kpc_dot_i8_neon(qs8, krow, DK);
#endif
    int32_t acc = 0;
    for (int64_t d = 0; d < DK; ++d) {
        const uint8_t b = krow[d >> 1];
        acc += (int32_t) qs8[d] * ((d & 1) ? (b >> 4) : (b & 0x0F));
    }
    return acc;
}

static inline void kpc_vscale(float * VKQ, float ms, int64_t DV) {
    ggml_vec_scale_f32((int) DV, VKQ, ms);   // shared SIMD helper (all ISAs)
}

// VKQ += f16(vh)*vs, V converted with F16C
static inline void kpc_vmad_f16(float * VKQ, const ggml_fp16_t * vh, float vs, int64_t DV) {
#ifdef KPC_X86
    if ((DV & 7) == 0 && kpc_have_avx2()) { kpc_vmad_f16_avx2(VKQ, vh, vs, DV); return; }
#elif defined(KPC_ARM)
    if ((DV & 7) == 0) { kpc_vmad_f16_neon(VKQ, vh, vs, DV); return; }
#endif
    for (int64_t d = 0; d < DV; ++d) VKQ[d] += GGML_CPU_FP16_TO_FP32(vh[d]) * vs;
}

static inline void kpc_vmad_f32(float * VKQ, const float * vf, float vs, int64_t DV) {
    ggml_vec_mad_f32((int) DV, VKQ, vf, vs);   // shared SIMD helper (all ISAs)
}

// VKQ += dequant(q4_1 V row)*vs, fused; DV multiple of 32
static inline void kpc_vmad_q4_1(float * VKQ, const uint8_t * vb, float vs, int64_t DV) {
#ifdef KPC_X86
    if ((DV & 31) == 0 && kpc_have_avx2()) { kpc_vmad_q4_1_avx2(VKQ, vb, vs, DV); return; }
#elif defined(KPC_ARM)
    if ((DV & 31) == 0) { kpc_vmad_q4_1_neon(VKQ, vb, vs, DV); return; }
#endif
    for (int64_t b = 0; b < DV/32; ++b) {
        const uint8_t * blk = vb + b*20;
        const float d = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 0));
        const float m = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 2));
        const uint8_t * qs = blk + 4;
        float * o = VKQ + b*32;
        for (int j = 0; j < 16; ++j) { o[j] += ((qs[j] & 0x0F)*d + m)*vs; o[j+16] += ((qs[j] >> 4)*d + m)*vs; }
    }
}

// VKQ += dequant(q3v_1 V row)*vs, fused; DV multiple of 32
static inline void kpc_vmad_q3v_1(float * VKQ, const uint8_t * vb, float vs, int64_t DV) {
#ifdef KPC_X86
    if ((DV & 31) == 0 && kpc_have_avx2()) { kpc_vmad_q3v_1_avx2(VKQ, vb, vs, DV); return; }
#elif defined(KPC_ARM)
    if ((DV & 31) == 0) { kpc_vmad_q3v_1_neon(VKQ, vb, vs, DV); return; }
#endif
    for (int64_t b = 0; b < DV/32; ++b) {
        const uint8_t * blk = vb + b*16;
        const float d = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 0));
        const float m = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 2));
        const uint8_t * qs = blk + 4;
        float * o = VKQ + b*32;
        for (int j = 0; j < 32; ++j) {
            const int bit = 3*j, byte = bit >> 3, off = bit & 7;
            int q = qs[byte] >> off;
            if (off > 5) q |= qs[byte + 1] << (8 - off);
            q &= 7;
            o[j] += (q*d + m)*vs;
        }
    }
}

// VKQ += dequant(q3v_2 V row)*vs, fused; DV multiple of 128 (int8-meta superblock)
static inline void kpc_vmad_q3v_2(float * VKQ, const uint8_t * vb, float vs, int64_t DV) {
#ifdef KPC_X86
    if ((DV & 127) == 0 && kpc_have_avx2()) { kpc_vmad_q3v_2_avx2(VKQ, vb, vs, DV); return; }
#elif defined(KPC_ARM)
    if ((DV & 127) == 0) { kpc_vmad_q3v_2_neon(VKQ, vb, vs, DV); return; }
#endif
    for (int64_t b = 0; b < DV/128; ++b) {
        const uint8_t * blk = vb + b*62;
        const float sd = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 0));
        const float mm = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 2));
        const float sm = GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *)(blk + 4));
        const uint8_t * qd = blk + 6;
        const uint8_t * qm = blk + 10;
        const uint8_t * qs = blk + 14;
        float * out = VKQ + b*128;
        for (int s = 0; s < 4; ++s) {
            const float d = qd[s]*sd;
            const float m = mm + qm[s]*sm;
            const uint8_t * qsb = qs + s*12;
            float * o = out + s*32;
            for (int j = 0; j < 32; ++j) {
                const int bit = 3*j, byte = bit >> 3, off = bit & 7;
                int q = qsb[byte] >> off;
                if (off > 5) q |= qsb[byte + 1] << (8 - off);
                q &= 7;
                o[j] += (q*d + m)*vs;
            }
        }
    }
}

// per-group Q prescale: qsf=pq*scale; corr=sum pq*zp; dqg=max|qsf|/127; qs8=round(qsf/dqg).
// scale/zp are the decoded f32 per-channel tables for this group (kq_scale already folded in).
static inline void kpc_prescale(const float * pq, const float * sc, const float * zp,
                                     int64_t DK, float * qsf, int8_t * qs8, float * out_dqg, float * out_corr) {
#ifdef KPC_X86
    if ((DK & 31) == 0 && kpc_have_avx2()) { kpc_prescale_avx2(pq, sc, zp, DK, qsf, qs8, out_dqg, out_corr); return; }
#elif defined(KPC_ARM)
    if ((DK & 31) == 0) { kpc_prescale_neon(pq, sc, zp, DK, qsf, qs8, out_dqg, out_corr); return; }
#endif
    float amax = 0.0f, corr = 0.0f;
    for (int64_t d = 0; d < DK; ++d) {
        const float vv = pq[d] * sc[d];
        qsf[d] = vv;
        const float a = fabsf(vv);
        if (a > amax) amax = a;
        corr += pq[d] * zp[d];
    }
    const float dqg = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float idqg = 1.0f / dqg;
    for (int64_t d = 0; d < DK; ++d) qs8[d] = (int8_t) lrintf(qsf[d] * idqg);
    *out_dqg  = dqg;
    *out_corr = corr;
}

// fused per-channel int4-K attention. kq_scale is folded into the per-run scale/zp decode (no separate
// q-scaling pass). supports ALiBi, logit softcap and attention sinks, mirroring ggml_flash_attn_ext.
void ggml_compute_forward_kpc_flash_attn(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const int ith = params->ith;
    const int nth = params->nth;
    const struct ggml_tensor * q  = dst->src[0];
    const struct ggml_tensor * k  = dst->src[1];
    const struct ggml_tensor * sz = dst->src[2];
    const struct ggml_tensor * v  = dst->src[3];
    const struct ggml_tensor * m  = dst->src[4];
    const struct ggml_tensor * gi = dst->src[5];   // group_index I32 [n_kv, ns]: per-slot scalezp pool index
    const struct ggml_tensor * sk = dst->src[6];   // sinks f32 [n_head] (optional)

    float kq_scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&kq_scale,      (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int64_t DK   = q->ne[0];
    const int64_t N    = q->ne[1];   // n_q
    const int64_t NH   = q->ne[2];   // n_head
    const int64_t NS   = q->ne[3];   // n_stream
    const int64_t NKV  = k->ne[1];
    const int64_t DV   = v->ne[0];
    const int64_t C    = (sz->ne[0] - 6) / 2;   // int8 scalezp slab = [3 fp16 super] + 2C uint8 (see KPC_SZ_GROUP_BYTES)
    const int64_t rk2  = NH / k->ne[2];   // GQA group
    const int64_t rk3  = NS / k->ne[3];
    const int64_t rv2  = NH / v->ne[2];
    const int64_t rv3  = NS / v->ne[3];

    // ALiBi slope per head (matches ggml_flash_attn_ext). max_bias==0 -> slope==1 (no bias).
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) NH));
    const float    m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float    m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const bool    vf16 = v->type == GGML_TYPE_F16;
    const bool    vf32 = v->type == GGML_TYPE_F32;
    const bool    vq41 = v->type == GGML_TYPE_Q4_1;
    const bool    vq3v = v->type == GGML_TYPE_Q3V_1;
    const bool    vq3v2 = v->type == GGML_TYPE_Q3V_2;
    // other quantized V (q8_0, ...): dequant the row via the type's to_float, then SIMD-mad.
    const ggml_to_float_t v_to_float = (vf16 || vf32 || vq41 || vq3v || vq3v2) ? NULL : ggml_get_type_traits(v->type)->to_float;
    GGML_ASSERT(DV <= 512);
    GGML_ASSERT(DK <= 512);
    GGML_ASSERT(vf16 || vf32 || vq41 || vq3v || vq3v2 || v_to_float);
    GGML_ASSERT(q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float) && "KPC fused attn: q must be contiguous f32");

    // GQA grouping: sibling q-heads of a kv head share the K/V/mask/gidx walk, each keeping its own
    // qs8/dqg/corr + VKQ/S/M -> bit-identical to ungrouped; gsz shrinks for load balance.
    #define KPC_GQA_MAX 8
    int64_t gsz = rk2 < KPC_GQA_MAX ? rk2 : KPC_GQA_MAX;
    if (m && m->ne[2] != 1) {
        gsz = 1;
    }
    const int64_t n_kvh = k->ne[2];
    while (gsz > 1 && N*NS*n_kvh*((rk2 + gsz - 1)/gsz) < 2*(int64_t) nth) {
        gsz = (gsz + 1)/2;
    }
    const int64_t n_chunk = (rk2 + gsz - 1)/gsz;   // sibling chunks per kv head

    float  VKQ[KPC_GQA_MAX][512];
    int8_t qs8[KPC_GQA_MAX][512];
    float  S[KPC_GQA_MAX], M[KPC_GQA_MAX], dqg[KPC_GQA_MAX], corr[KPC_GQA_MAX], slope[KPC_GQA_MAX];
    const float * pq[KPC_GQA_MAX];
    int64_t iv2[KPC_GQA_MAX];
    float  V32[512];
    float  qsf[512];
    float sc_dec[512], zp_dec[512];   // per-group decoded scale/zp slice (int8 metadata -> f32, kq_scale folded)
    const int64_t KPC_KV_TILE = 64;   // mask block-skip stride (mirrors flash_attn_ext_tiled)
    const int64_t nr = N * n_kvh * n_chunk * NS;
    const int64_t r0 = (nr * ith) / nth;
    const int64_t r1 = (nr * (ith + 1)) / nth;
    for (int64_t ir = r0; ir < r1; ++ir) {
        const int64_t iq3  = ir / (n_kvh*n_chunk*N);
        const int64_t rem  = ir - iq3*n_kvh*n_chunk*N;
        const int64_t ik2  = rem / (n_chunk*N);
        const int64_t rem2 = rem - ik2*n_chunk*N;
        const int64_t ch   = rem2 / N;
        const int64_t iq1  = rem2 - ch*N;
        const int64_t ik3  = iq3 / rk3, iv3 = iq3 / rv3;
        const int64_t hq0  = ik2*rk2 + ch*gsz;                              // first q-head of this chunk
        const int64_t hcnt = gsz < rk2 - ch*gsz ? gsz : rk2 - ch*gsz;       // heads in this chunk

        for (int64_t hi = 0; hi < hcnt; ++hi) {
            const int64_t iq2 = hq0 + hi;
            const uint32_t h = (uint32_t) iq2;   // head index (ALiBi slope / sink logit are per-head)
            slope[hi] = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;
            pq[hi]    = (const float *)((const char *)q->data + iq1*q->nb[1] + iq2*q->nb[2] + iq3*q->nb[3]);
            iv2[hi]   = iq2 / rv2;
            for (int64_t d = 0; d < DV; ++d) VKQ[hi][d] = 0.0f;
            S[hi] = 0.0f;
            M[hi] = -INFINITY;
        }
        const ggml_fp16_t * mp = m ? (const ggml_fp16_t *)((const char *)m->data + iq1*m->nb[1] + (hq0 % m->ne[2])*m->nb[2] + (iq3 % m->ne[3])*m->nb[3]) : NULL;

        const int64_t cbase = ik2 * DK;
        // walk keys in slot order, folding per-channel scale into Q once per run of keys sharing a
        // scalezp pool entry (group_index). contiguous -> 32-key runs, scattered (SWA) -> shorter.
        const int32_t * gidx = (const int32_t *)((const char *)gi->data + ik3*gi->nb[1]);
        const int64_t   ng   = sz->ne[1];           // scalezp pool size (for a defensive clamp)
        int64_t ic = 0;
        while (ic < NKV) {
            // bulk-skip a fully-masked KPC_KV_TILE; partial tiles fall through to the per-cell skip
            if (mp && (ic % KPC_KV_TILE) == 0) {
                const int64_t tend = ic + KPC_KV_TILE < NKV ? ic + KPC_KV_TILE : NKV;
                bool all_masked = true;
                for (int64_t t = ic; t < tend; ++t) {
                    if (mp[t] != KPC_F16_NEG_INF) { all_masked = false; break; }   // raw-bits -inf test
                }
                if (all_masked) { ic = tend; continue; }
            }
            if (mp && mp[ic] == KPC_F16_NEG_INF) { ++ic; continue; }   // masked: skip; group_index may be unset for never-written slots
            const int64_t gkey = gidx[ic];           // raw grouping key: the run-start cell always matches itself
            int64_t pool = gkey;                     // pool index into scalezp
            if (pool < 0 || pool >= ng) pool = 0;    // clamp only for indexing (unwritten / no-mask slots); never the run key
            const uint8_t * slab = (const uint8_t *)((const char *)sz->data + ik3*sz->nb[2] + pool*sz->nb[1]);
            float ss, zmin, szc; kpc_sz_super_read(slab, &ss, &zmin, &szc);
            for (int64_t d = 0; d < DK; ++d) {              // decode this kv-head's DK-channel slice; fold kq_scale
                sc_dec[d] = (slab[6 + cbase + d] * ss) * kq_scale;
                zp_dec[d] = (zmin + slab[6 + C + cbase + d] * szc) * kq_scale;
            }
            for (int64_t hi = 0; hi < hcnt; ++hi) {         // the decoded table is shared; the Q fold is per head
                kpc_prescale(pq[hi], sc_dec, zp_dec, DK, qsf, qs8[hi], &dqg[hi], &corr[hi]);
            }

            const int64_t ic_start = ic;
            for (; ic < NKV; ++ic) {
                if (ic > ic_start && (ic % KPC_KV_TILE) == 0) {
                    break;                            // allow outer loop to attempt a bulk block-skip
                }
                if (mp && mp[ic] == KPC_F16_NEG_INF) {
                    continue;                         // masked within the run: skip (don't read gidx for the break test)
                }
                if ((int64_t) gidx[ic] != gkey) {     // compare the RAW key so ic_start always processes -> forward progress
                    break;                            // attended key in a different group -> re-fold for the next run
                }
                const float mv = mp ? GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
                const uint8_t * krow = (const uint8_t *)((const char *)k->data + ic*k->nb[1] + ik2*k->nb[2] + ik3*k->nb[3]);
                const char * vbase = (const char *)v->data + ic*v->nb[1] + iv3*v->nb[3];
                const char * conv  = NULL;            // last to_float-converted V row (shared across heads)
                for (int64_t hi = 0; hi < hcnt; ++hi) {
                    // already-scaled KQ value
                    float s = dqg[hi] * (float) kpc_dot_i8(qs8[hi], krow, DK) + corr[hi];
                    if (logit_softcap != 0.0f) {
                        s = logit_softcap * tanhf(s / logit_softcap);   // softcap before mask, mirroring flash_attn_ext
                    }
                    s += slope[hi] * mv;                                 // ALiBi: slope-scaled mask (slope==1 -> plain mask)

                    const char * vd = vbase + iv2[hi]*v->nb[2];
                    const float Mold = M[hi];
                    float ms = 1.0f, vs = 1.0f;
                    if (s > M[hi]) { M[hi] = s; ms = expf(Mold - M[hi]); kpc_vscale(VKQ[hi], ms, DV); }
                    else           { vs = expf(s - M[hi]); }
                    if      (vf16) kpc_vmad_f16(VKQ[hi], (const ggml_fp16_t *) vd, vs, DV);
                    else if (vf32) kpc_vmad_f32(VKQ[hi], (const float *) vd, vs, DV);
                    else if (vq41) kpc_vmad_q4_1(VKQ[hi], (const uint8_t *) vd, vs, DV);
                    else if (vq3v) kpc_vmad_q3v_1(VKQ[hi], (const uint8_t *) vd, vs, DV);
                    else if (vq3v2) kpc_vmad_q3v_2(VKQ[hi], (const uint8_t *) vd, vs, DV);
                    else {
                        if (vd != conv) { v_to_float(vd, V32, DV); conv = vd; }            // convert once per distinct row
                        kpc_vmad_f32(VKQ[hi], V32, vs, DV);
                    }
                    S[hi] = S[hi]*ms + vs;
                }
            }
        }

        for (int64_t hi = 0; hi < hcnt; ++hi) {
            const int64_t iq2 = hq0 + hi;
            // attention sink: fold the per-head sink logit into the softmax denominator (no V contribution),
            // matching ggml_flash_attn_ext_add_sinks. applied once per query row (single chunk).
            if (sk) {
                const float sks = ((const float *) sk->data)[iq2];
                float ms = 1.0f, vs = 1.0f;
                if (sks > M[hi]) { ms = expf(M[hi] - sks); M[hi] = sks; kpc_vscale(VKQ[hi], ms, DV); }
                else             { vs = expf(sks - M[hi]); }
                S[hi] = S[hi]*ms + vs;
            }

            const float Sinv = (S[hi] == 0.0f) ? 0.0f : 1.0f / S[hi];
            float * out = (float *)((char *)dst->data + (iq3*dst->ne[2]*dst->ne[1] + iq2 + iq1*dst->ne[1])*dst->nb[1]);
            for (int64_t d = 0; d < DV; ++d) out[d] = VKQ[hi][d] * Sinv;
        }
    }
    #undef KPC_GQA_MAX
}

// in-place per-channel int4 K write: tokens group by logical pos/32 per seq and scatter to the
// k_idxs slots; each (seq,group) owns a scalezp pool, with members incl. the seq's staged f16
// originals (k_resid). work is split into (seq,pool) items, staging updated after a barrier.
void ggml_compute_forward_kpc_write(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * kc  = dst->src[0];   // k_cur   f32 [C, n_tokens] (stream-ordered)
    struct ggml_tensor       * sz  = dst->src[1];   // scalezp F16 [2C, ng_max, n_stream]  (in-place)
    struct ggml_tensor       * rs  = dst->src[2];   // k_resid F16 [C, GROUP, n_seq_max]    (in-place)
    const struct ggml_tensor * idx = dst->src[3];   // k_idxs  I64 [n_tokens] (global slots)
    int32_t                  * gid = (int32_t *) dst->src[4]->data;   // group_index   I32 [kv_size, n_stream] (in-place)
    int32_t                  * rsl = (int32_t *) dst->src[5]->data;   // k_resid_slots I32 [GROUP, n_seq_max]   (in-place)
    int32_t                  * sgp = (int32_t *) dst->src[6]->data;   // staged_group  I32 [n_seq_max]          (in-place)
    int32_t                  * smk = (int32_t *) dst->src[7]->data;   // staged_mask   I32 [n_seq_max]          (in-place)
    const int32_t            * kpc_seq = (const int32_t *) dst->src[8]->data;   // primary seq per token I32 [n_tokens]
    const int32_t            * kpc_pos = (const int32_t *) dst->src[9]->data;   // true position per token I32 [n_tokens]

    int32_t pr[16];
    memcpy(pr, dst->op_params, sizeof(pr));

    // each of the n_seqps sequences sharing a stream owns a band_size scalezp pool band;
    // L = staging-slot count (L < n_seq_max when virtualized)
    const int64_t   kv_size   = dst->ne[1];
    const int64_t   ng_max    = sz->ne[1];
    const int64_t   n_stream  = dst->src[1]->ne[2];
    const int64_t   n_seq_max = pr[0];
    const int64_t   L         = dst->src[6]->ne[0];
    GGML_ASSERT(n_stream >= 1 && n_seq_max >= n_stream);
    const int64_t   n_seqps   = n_seq_max / n_stream;
    GGML_ASSERT(n_seqps >= 1);
    const int64_t   band_size = ng_max / n_seqps;
    GGML_ASSERT(band_size >= 1 && "KPC write: scalezp pool too small for n_seq_max (host must validate kv_size/32 >= n_seq_max/n_stream)");

    const int       ith  = params->ith;
    const int       nth  = params->nth;
    const int64_t   C    = kc->ne[0];
    const int64_t   G    = KPC_GROUP;
    const int64_t   krow = C / 2;                       // KPC4_1 packed row size (bytes)
    const int64_t   n_tokens = kc->ne[1];
    const int64_t * idxd = (const int64_t *) idx->data;
    uint8_t       * szd  = (uint8_t *)       sz->data;   // int8 scalezp metadata (see KPC_SZ_GROUP_BYTES)
    ggml_fp16_t   * rsd  = (ggml_fp16_t *)   rs->data;
    uint8_t       * kd   = (uint8_t *)       dst->data;
    std::vector<float> gsc((size_t) C), gzp((size_t) C);   // per-group scale/zp scratch for int8 metadata encode
    std::vector<float> nsc((size_t) C), nzp((size_t) C);   // decoded effective scale/zp table for the pack loops
    std::vector<int64_t> resc_slots;                       // rescue: live cells re-packed on pool re-encode
    std::vector<float>   resc_vals;

    const uint32_t full_mask = G >= 32 ? 0xFFFFFFFFu : ((1u << (uint32_t) G) - 1u);

    // every token's seq must fit the band namespace and its packed slot must fit the staging slots (host enforces too)
    if (ith == 0) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            const int64_t seq  = kpc_seq[t] & KPC_SEQ_MASK;
            const int64_t slot = (int64_t)(uint32_t) kpc_seq[t] >> KPC_SLOT_SHIFT;
            GGML_ASSERT(seq  >= 0 && seq  < n_seq_max && "KPC write: seq_id out of range");
            GGML_ASSERT(slot >= 0 && slot <= L        && "KPC write: staging slot out of range");   // slot==L -> spilled (no staging)
        }
    }

    // work list: the distinct (seq, lg) pairs of this call (plus the staged open group when touched),
    // grouped by (seq, pool) and round-robined; built identically on every thread.
    std::vector<int64_t> seq_lo((size_t) n_seq_max, INT64_MAX);
    std::vector<int64_t> seq_hi((size_t) n_seq_max, -1);
    std::vector<int64_t> seq_slot((size_t) n_seq_max, -1);
    std::vector<std::pair<int64_t, int64_t>> work;   // (seq, lg)
    work.reserve((size_t) n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        const int64_t sb = kpc_seq[t] & KPC_SEQ_MASK;
        const int64_t lg = (int64_t) kpc_pos[t] / G;
        seq_slot[sb] = (int64_t)(uint32_t) kpc_seq[t] >> KPC_SLOT_SHIFT;   // same for all of sb's tokens
        if (lg < seq_lo[sb]) seq_lo[sb] = lg;
        if (lg > seq_hi[sb]) seq_hi[sb] = lg;
        work.emplace_back(sb, lg);
    }
    for (int64_t sb = 0; sb < n_seq_max; ++sb) {     // staged open group with no tokens this call but in range
        if (seq_hi[sb] < 0 || seq_slot[sb] >= L) {
            continue;
        }
        const int64_t slot = seq_slot[sb];
        if (smk[slot] != 0 && sgp[slot] >= seq_lo[sb] && sgp[slot] <= seq_hi[sb]) {
            work.emplace_back(sb, (int64_t) sgp[slot]);
        }
    }
    auto pool_of = [&](int64_t sb, int64_t lg) -> int64_t {
        return (sb % n_seqps)*band_size + (lg % band_size);   // ring within this sequence's pool band
    };
    std::sort(work.begin(), work.end(), [&](const std::pair<int64_t, int64_t> & a, const std::pair<int64_t, int64_t> & b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        const int64_t pa = pool_of(a.first, a.second);
        const int64_t pb = pool_of(b.first, b.second);
        if (pa != pb) {
            return pa < pb;
        }
        return a.second < b.second;
    });
    work.erase(std::unique(work.begin(), work.end()), work.end());

    // pools this call (re-)encodes; a pool absent here is touched by no thread this seal -> the fresh-pool rescue
    // (below) can relocate shared cells into one without racing another thread's encode.
    std::vector<bool> work_pools((size_t) ng_max, false);
    for (const auto & wk : work) work_pools[pool_of(wk.first, wk.second)] = true;

    // per-(seq,lg) membership: within-group index w -> the k_cur token that wrote it (-1 if staged-only)
    // and its physical slot. folds the staged open-group cells when lg is the staged group.
    int64_t tok_of[32];    // G <= 32
    int64_t slot_of[32];
    auto gather = [&](int64_t sb, int64_t lg, int64_t slot, bool stage) -> uint32_t {
        for (int64_t w = 0; w < G; ++w) { tok_of[w] = -1; slot_of[w] = -1; }
        uint32_t members = 0;
        for (int64_t t = 0; t < n_tokens; ++t) {
            if (((int64_t) kpc_seq[t] & KPC_SEQ_MASK) != sb) continue;
            const int64_t p = (int64_t) kpc_pos[t];
            if (p / G != lg) continue;
            const int64_t w = p % G;
            members  |= (1u << w);
            tok_of[w] = t;
            slot_of[w] = idxd[t] % kv_size;   // k_idxs is global slot incl stream offset
        }
        if (stage && lg == sgp[slot]) {
            const uint32_t prev_mask = (uint32_t) smk[slot];
            for (int64_t w = 0; w < G; ++w) {
                if ((prev_mask & (1u << w)) && !(members & (1u << w))) {
                    members  |= (1u << w);
                    slot_of[w] = rsl[slot*G + w];   // staged slot
                }
            }
        }
        return members;
    };

    // encode + pack; reads the staging state but never writes it, so items of one seq can run on
    // different threads (the staging update is deferred past the barrier below)
    int64_t item = -1;
    int64_t prev_sb = -1, prev_pool = -1;
    for (size_t wi = 0; wi < work.size(); ++wi) {
        const int64_t sb = work[wi].first;
        const int64_t lg = work[wi].second;
        const int64_t pl = pool_of(sb, lg);
        if (sb != prev_sb || pl != prev_pool) { ++item; prev_sb = sb; prev_pool = pl; }
        if ((int) (item % nth) != ith) {
            continue;
        }

        const int64_t s     = sb / n_seqps;                  // physical stream
        const int64_t sbase = s * sz->nb[2];                 // scalezp stream slab (byte offset)
        const int64_t slot  = seq_slot[sb];
        const bool    stage = slot < L;                      // slot == L: spilled, no staging to fold
        const int64_t rbase = stage ? slot * (C*G) : 0;      // k_resid staging slab (f16 elems)

        // original of a member at within-pos w: f32 from k_cur token tk (>=0) else f16 from staged k_resid
        auto orig = [&](int64_t tk, int64_t w, int64_t c) -> float {
            if (tk >= 0) {
                return *(const float *)((const char *)kc->data + tk*kc->nb[1] + c*kc->nb[0]);
            }
            return GGML_CPU_FP16_TO_FP32(rsd[rbase + w*C + c]);
        };

        const uint32_t members = gather(sb, lg, slot, stage);
        if (members == 0) {
            continue;   // fold-only entry whose staging emptied: nothing to write
        }

        {
            const int64_t pool = pl;
            GGML_ASSERT(pool < ng_max);
            uint8_t * szg = szd + sbase + pool*sz->nb[1];

            // rescue: live cells still mapped to this pool but not members of this write (stale staging,
            // band ring reuse, seq_cp sharing) are dequantized against the current slab and repacked with
            // the new scale, so a pool re-encode never orphans a referencing cell. gid < 0 = free cell.
            resc_slots.clear();
            resc_vals.clear();
            for (int64_t p = 0; p < kv_size; ++p) {
                if (gid[s*kv_size + p] != (int32_t) pool) continue;
                bool is_member = false;
                for (int64_t w = 0; w < G; ++w) {
                    if ((members & (1u << w)) && slot_of[w] == p) { is_member = true; break; }
                }
                if (is_member) continue;
                if (resc_slots.empty()) {                // decode the current slab's scale/zp table once
                    float ss, zmin, szc; kpc_sz_super_read(szg, &ss, &zmin, &szc);
                    for (int64_t c = 0; c < C; ++c) {
                        kpc_sz_decode1(szg, C, c, ss, zmin, szc, &nsc[c], &nzp[c]);
                    }
                }
                const uint8_t * row = kd + (s*kv_size + p)*krow;
                const size_t base = resc_vals.size();
                resc_vals.resize(base + (size_t) C);
                for (int64_t c = 0; c < C; ++c) {
                    resc_vals[base + (size_t) c] = ggml_kpc_deq(row[c/2], (int) c, nsc[c], nzp[c]);
                }
                resc_slots.push_back(p);
            }

            // Fresh-pool rescue: these live, non-member cells (e.g. seq_cp-shared cells whose owning seq is re-using
            // this band) would otherwise be re-quantized into THIS pool under a scale combined with the new members
            // -- coarse when the reused content diverges, so the sharing seq's prefix drifts. Instead relocate them to
            // a FREE pool BIT-EXACTLY: copy their existing scale slab there and just repoint gid (the int4 rows are
            // untouched -> no requant). A "free" pool is one no work item of this call (re-)encodes and no live cell
            // references, so no other thread touches it -> moving these (non-written-seq) cells is race-free. Falls
            // back to the combine below when the band is full.
            if (!resc_slots.empty()) {
                int64_t fp = -1;
                for (int64_t cand = 0; cand < ng_max; ++cand) {
                    if ((cand % nth) != ith) continue;   // disjoint per-thread candidate set -> two concurrent
                    if (cand == pool || work_pools[cand]) continue;   // rescues can never claim the same free pool
                    bool used = false;
                    for (int64_t p = 0; p < kv_size; ++p) {
                        if (gid[s*kv_size + p] == (int32_t) cand) { used = true; break; }
                    }
                    if (!used) { fp = cand; break; }
                }
                if (fp >= 0) {
                    memcpy(szd + sbase + fp*sz->nb[1], szg, (size_t) sz->nb[1]);   // copy existing scale -> bit-exact
                    for (size_t r = 0; r < resc_slots.size(); ++r) {
                        gid[s*kv_size + resc_slots[r]] = (int32_t) fp;
                    }
                    resc_slots.clear();
                    resc_vals.clear();
                }
            }

            for (int64_t c = 0; c < C; ++c) {            // per-channel scale/zp over members + rescued cells
                float mn = INFINITY, mx = -INFINITY;
                for (int64_t w = 0; w < G; ++w) {
                    if (!(members & (1u << w))) continue;
                    const float v = orig(tok_of[w], w, c);
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                for (size_t r = 0; r < resc_slots.size(); ++r) {
                    const float v = resc_vals[r*(size_t) C + (size_t) c];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                float scale = (mx - mn) / 15.0f;
                if (scale == 0.0f) scale = 1.0f;
                gsc[c] = scale; gzp[c] = mn;
            }
            kpc_sz_encode(gsc.data(), gzp.data(), C, szg);
            {                                            // decode the int8-rounded effective scale/zp table once
                float ss, zmin, szc; kpc_sz_super_read(szg, &ss, &zmin, &szc);
                for (int64_t c = 0; c < C; ++c) {
                    kpc_sz_decode1(szg, C, c, ss, zmin, szc, &nsc[c], &nzp[c]);
                    if (nsc[c] == 0.0f) nsc[c] = 1.0f;   // tiny scale can underflow to 0
                }
            }
            for (int64_t w = 0; w < G; ++w) {
                if (!(members & (1u << w))) continue;
                const int64_t p    = slot_of[w];
                uint8_t * row      = kd + (s*kv_size + p)*krow;
                gid[s*kv_size + p] = (int32_t) pool;
                for (int64_t c = 0; c < C; ++c) {
                    kpc_pack_nibble(row, c, orig(tok_of[w], w, c), nsc[c], nzp[c]);
                }
            }
            for (size_t r = 0; r < resc_slots.size(); ++r) {
                uint8_t * row = kd + (s*kv_size + resc_slots[r])*krow;
                for (int64_t c = 0; c < C; ++c) {
                    kpc_pack_nibble(row, c, resc_vals[r*(size_t) C + (size_t) c], nsc[c], nzp[c]);
                }
            }
        }
    }

    ggml_barrier(params->threadpool);

    // staging update, after the barrier so the packing above saw the pre-call staging; only the
    // highest group of a staged seq stays open, spilled seqs (slot == L) skip -> frozen
    item = -1; prev_sb = -1; prev_pool = -1;
    for (size_t wi = 0; wi < work.size(); ++wi) {
        const int64_t sb = work[wi].first;
        const int64_t lg = work[wi].second;
        const int64_t pl = pool_of(sb, lg);
        if (sb != prev_sb || pl != prev_pool) { ++item; prev_sb = sb; prev_pool = pl; }
        if ((int) (item % nth) != ith) {
            continue;
        }
        const int64_t slot = seq_slot[sb];
        if (lg != seq_hi[sb] || slot >= L) {
            continue;
        }

        const int64_t rbase  = slot * (C*G);
        const int64_t slbase = slot * G;
        const uint32_t members = gather(sb, lg, slot, /*stage =*/ true);   // old smk/sgp still intact here

        if (members == full_mask) {                  // group complete
            sgp[slot] = (int32_t) lg;
            smk[slot] = 0;
        } else {
            for (int64_t w = 0; w < G; ++w) {
                if (!(members & (1u << w))) continue;
                if (tok_of[w] < 0) continue;         // already staged in a previous call; leave intact
                rsl[slbase + w] = (int32_t) slot_of[w];
                for (int64_t c = 0; c < C; ++c) {
                    const float o = *(const float *)((const char *)kc->data + tok_of[w]*kc->nb[1] + c*kc->nb[0]);
                    rsd[rbase + w*C + c] = GGML_CPU_FP32_TO_FP16(o);
                }
            }
            sgp[slot] = (int32_t) lg;
            smk[slot] = (int32_t) members;
        }
    }
}
