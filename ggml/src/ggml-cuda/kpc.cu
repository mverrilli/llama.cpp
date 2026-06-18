#include "kpc.cuh"
#include "mma.cuh"

// ============================================================================
// KPC CUDA — GPU-native per-channel int4 K-cache (branch kpc-cuda-kv).
//
// Tensor contract (GPU-native; differs from the CPU op — positional, no group_index):
//   KPC_FLASH_ATTN dst->src: [0]=Q f32, [1]=K_packed int4 (KPC4_1), [2]=scalezp u8
//     [SZ, kv/32, ns] (group = token>>5), [3]=V (q4_1/f16), [4]=mask, [5]=residual window
//     f16 [C,32,nseq] (open <32 tail), [6]=sinks. op_params: kq_scale, max_bias, logit_softcap.
//   KPC_WRITE seals a full 32-token group into K_packed + scalezp and resets the window.
//
// Kernel math validated standalone on the P40 (round-trip NMSE 3.5e-4, attn-read NMSE 3.2e-5).
// supports_op is gated OFF below until the kv-cache integration (M3) provides these tensors.
// ============================================================================

#define KPC_SZ_QMAX 255

// per-channel int4 dequant: c even -> low nibble, c odd -> high.  v = q*scale[c] + zp[c]
static __device__ __forceinline__ float kpc_deq_nib(uint8_t nb, int c, float s, float z) {
    const int q = (c & 1) ? (nb >> 4) : (nb & 0x0F);
    return q * s + z;
}

// largest power of two <= n (ALiBi n_head_log2). Integer form: GPU log2f(2^k) can round to just under k, and
// floorf() then truncates to k-1 -> a wrong slope for power-of-2 head counts (e.g. n_head=16 -> 8). The host fattn
// path dodges this by computing on the CPU; KPC computes the slope in-kernel, so it must be exact here.
static __device__ __forceinline__ int kpc_pow2_floor(int n) { int p = 1; while ((p << 1) <= n) p <<= 1; return p; }

static __device__ __forceinline__ void kpc_pack_nib(uint8_t * row, int c, float v, float s, float z) {
    int qv = (int)((v - z) / s + 0.5f);
    qv = qv < 0 ? 0 : (qv > 15 ? 15 : qv);
    uint8_t * b = &row[c >> 1];
    *b = (c & 1) ? ((*b & 0x0F) | (uint8_t)(qv << 4)) : ((*b & 0xF0) | (uint8_t) qv);
}

// read the three fp16 super-params from a group slab
static __device__ __forceinline__ void kpc_slab_super(const uint8_t * s, float * ss, float * zmn, float * sz) {
    half h;
    h = *(const half *)(s + 0); *ss  = __half2float(h);
    h = *(const half *)(s + 2); *zmn = __half2float(h);
    h = *(const half *)(s + 4); *sz  = __half2float(h);
}

// ---- seal one 32-token group: in[32][C] f32 -> int4 nibbles + slab (one block, thread 0) ----
// Mirrors ggml-cpu/kpc.cpp:20-72 (scale=(mx-mn)/15, zp=min, int8 double-quant slab, pack with
// the int8-rounded scale/zp).  Reference-grade (perf is M6).
static __global__ void kpc_seal_kernel(const float * in, uint8_t * nib, uint8_t * slab, int C) {
    const int g = blockIdx.x;
    const float  * gin  = in   + (size_t) g * GGML_KPC_GROUP * C;
    uint8_t      * gnib = nib  + (size_t) g * GGML_KPC_GROUP * (C / 2);
    uint8_t      * s    = slab + (size_t) g * GGML_KPC_SZ_GROUP_BYTES(C);

    extern __shared__ float smem[];           // scale[C], zp[C]
    float * scale = smem;
    float * zp    = smem + C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float mn = INFINITY, mx = -INFINITY;
        for (int t = 0; t < GGML_KPC_GROUP; ++t) { float v = gin[t * C + c]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
        float sc = (mx - mn) / 15.0f; if (sc == 0.0f) sc = 1.0f;
        scale[c] = sc; zp[c] = mn;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float smax = 0.0f;       for (int c = 0; c < C; ++c) smax = fmaxf(smax, scale[c]);
        float ss = smax / (float) KPC_SZ_QMAX; if (ss == 0.0f) ss = 1.0f;
        float zmn = INFINITY, zmx = -INFINITY; for (int c = 0; c < C; ++c) { zmn = fminf(zmn, zp[c]); zmx = fmaxf(zmx, zp[c]); }
        float sz = (zmx - zmn) / (float) KPC_SZ_QMAX; if (sz == 0.0f) sz = 1.0f;
        *(half *)(s + 0) = __float2half(ss);
        *(half *)(s + 2) = __float2half(zmn);
        *(half *)(s + 4) = __float2half(sz);
        uint8_t * qs = s + 6; uint8_t * qz = s + 6 + C;
        for (int c = 0; c < C; ++c) {
            int a = (int)(scale[c] / ss + 0.5f);          a = a < 0 ? 0 : (a > KPC_SZ_QMAX ? KPC_SZ_QMAX : a);
            int b = (int)((zp[c] - zmn) / sz + 0.5f);      b = b < 0 ? 0 : (b > KPC_SZ_QMAX ? KPC_SZ_QMAX : b);
            qs[c] = (uint8_t) a; qz[c] = (uint8_t) b;
        }
    }
    __syncthreads();
    float ss, zmn, sz; kpc_slab_super(s, &ss, &zmn, &sz);
    const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C;
    for (int t = threadIdx.x; t < GGML_KPC_GROUP; t += blockDim.x) {
        uint8_t * row = gnib + t * (C / 2);
        for (int c = 0; c < C; ++c) kpc_pack_nib(row, c, gin[t * C + c], qs[c] * ss, zmn + qz[c] * sz);
    }
}

// ---- flash-decode read: 1 block per (head, query, stream); online softmax over sealed K, q4_1/f16 V ----
// Mirrors the CPU kernel ggml_compute_forward_kpc_flash_attn indexing exactly (strides, GQA head/kv-head
// mapping, positional scalezp pool = key>>5, dst output, alibi/softcap/sinks). Reference-grade single-thread
// inner loop (warp-tile/split-K is M6). K is [head_dim, n_kv, n_kvh, ns] sliced by kv-head via k_nb2; the
// scalezp slab carries all C_full=n_embd_k_gqa channels, so the head's slice starts at cbase=ik2*head_dim.
// DK_T / DV_T > 0 specialize head_dim / DV at compile time so the qp[] and acc[] arrays and their loops have
// constant bounds (register-resident, no local-memory spill). DK_T == 0 falls back to the runtime values.
template<int DK_T, int DV_T>
static __global__ void kpc_flash_decode_kernel(
        const float * Q, const uint8_t * Kp, const uint8_t * sz0, const uint8_t * V, const half * mask,
        const float * sinks, const int32_t * gid, int gi_stride, float kqscale, float max_bias, float logit_softcap, float * dst,
        int head_dim_rt, int DV_rt, int C_full, int n_kv, int n_head, int n_q, int n_kvh, int n_vh, int n_stream,
        int rk2, int rv2, int rk3, int rv3, bool v_q41,
        int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
        int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
        int64_t sz_nb1, int64_t sz_nb2,
        int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
        int64_t m_nb1, int64_t m_nb2, int64_t m_nb3, int m_ne2, int m_ne3,
        int64_t dst_nb1, int dst_ne1, int dst_ne2, int n_split,
        float * parts, float2 * meta_out, int kv_split) {
    const int iq2 = blockIdx.x;   // query head
    // grid-level split-K (decode only): when kv_split>1, blockIdx.y is the KV super-block index and there is a
    // single query (iq1=0); the block reduces its KV slice to a partial that a combine kernel finishes. When
    // kv_split==1, blockIdx.y is the query position (original behaviour) and the block writes dst directly.
    const int kv_g = (kv_split > 1) ? blockIdx.y : 0;
    const int iq1  = (kv_split > 1) ? 0          : (int) blockIdx.y;   // query position
    const int iq3  = blockIdx.z;   // stream
    if (iq2 >= n_head || iq1 >= n_q || iq3 >= n_stream) return;
    const int wid  = threadIdx.x >> 5;                       // warp = key-split partition (split-K)
    const int lane = threadIdx.x & 31;                       // n_split warps per (head, query, stream)

    const int head_dim = DK_T ? DK_T : head_dim_rt;          // compile-time when specialized -> loops unroll
    const int DV       = DV_T ? DV_T : DV_rt;
    constexpr int NQP  = DK_T ? ((DK_T + 31) / 32) : 8;      // qp[] channels per lane  (head_dim <= 256)
    constexpr int NACC = DV_T ? ((DV_T + 31) / 32) : 16;     // acc[] V elements per lane (DV <= 512)

    const int ik2 = iq2 / rk2, iv2 = iq2 / rv2;
    const int ik3 = iq3 / rk3, iv3 = iq3 / rv3;
    const int cbase = ik2 * head_dim;

    // ALiBi slope (max_bias==0 -> 1).
    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const int n_head_log2 = kpc_pow2_floor(n_head);
        const float m0 = exp2f(-(max_bias)        / n_head_log2);
        const float m1 = exp2f(-(max_bias / 2.0f) / n_head_log2);
        slope = (iq2 < n_head_log2) ? powf(m0, iq2 + 1) : powf(m1, 2 * (iq2 - n_head_log2) + 1);
    }

    const float * qrow = (const float *)((const char *) Q + (size_t) iq1 * q_nb1 + (size_t) iq2 * q_nb2 + (size_t) iq3 * q_nb3);
    const half  * mp   = mask ? (const half *)((const char *) mask + (size_t) iq1 * m_nb1 + (size_t)(iq2 % m_ne2) * m_nb2 + (size_t)(iq3 % m_ne3) * m_nb3) : nullptr;

    // each lane owns DV elements {lane, lane+32, ...} of the running VKQ accumulator (registers)
    const int ndv = NACC;                                    // (DV+31)/32, compile-time when DV_T specialized
    float acc[NACC];
    for (int j = 0; j < ndv; ++j) acc[j] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    // KPC groups 32 keys under one scalezp slab, so the per-channel Q*scale fold + zp correction is constant
    // across a group: compute it once per pool (qp[] = q*scale for this lane's channels, corr = sum q*zp) and
    // reuse it for every key in the group. Per key then costs only int4 loads + a MAD (was a full re-fold/key).
    float qp[NQP];                                           // q*scale per lane channel (sized at compile time)
    float corr = 0.0f;
    int   cur_pool = -1;

    // this block owns the KV slice [kv_lo, kv_hi); kv_split==1 -> the whole cache (original behaviour).
    const int kv_slice = (n_kv + kv_split - 1) / kv_split;
    const int kv_lo    = kv_g * kv_slice;
    const int kv_hi    = (kv_lo + kv_slice < n_kv) ? (kv_lo + kv_slice) : n_kv;
    for (int t = kv_lo + wid; t < kv_hi; t += n_split) {    // split-K: warp wid owns keys {kv_lo+wid, +n_split, ...}
        const float mv = mp ? __half2float(mp[t]) : 0.0f;
        if (mp && mv == -INFINITY) continue;                 // same for every lane -> no warp divergence
        int pool = gid ? gid[(size_t) ik3 * gi_stride + t] : (t / GGML_KPC_GROUP);   // cell->pool (positional if no map)
        if (pool < 0) pool = 0;
        if (pool != cur_pool) {                              // entered a new group -> refold Q against its scale/zp
            const uint8_t * s = sz0 + (size_t) ik3 * sz_nb2 + (size_t) pool * sz_nb1;
            float ss, zmn, szc; kpc_slab_super(s, &ss, &zmn, &szc);
            const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C_full;
            float lc = 0.0f; int ci = 0;
            for (int c = lane; c < head_dim; c += 32, ++ci) {
                const int fc = cbase + c;
                qp[ci] = qrow[c] * (qs[fc] * ss);
                lc    += qrow[c] * (zmn + qz[fc] * szc);
            }
            for (int o = 16; o > 0; o >>= 1) lc += __shfl_xor_sync(0xffffffff, lc, o);
            corr = lc;                                       // sum_c q[c]*zp[c] (same for every key in the group)
            cur_pool = pool;
        }
        const uint8_t * krow = (const uint8_t *) Kp + (size_t) t * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
        float part = 0.0f; int ci = 0;                       // lane's slice of sum_c qp[c]*nibble[c]
        for (int c = lane; c < head_dim; c += 32, ++ci) {
            const uint8_t nb = krow[c >> 1];
            part += qp[ci] * (float) ((c & 1) ? (nb >> 4) : (nb & 0x0F));
        }
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);   // warp-reduce -> all lanes
        float score = (part + corr) * kqscale;
        if (logit_softcap != 0.0f) score = logit_softcap * tanhf(score / logit_softcap);
        score += slope * mv;

        const float mn = fmaxf(m, score), a = expf(m - mn), p = expf(score - mn);
        l = l * a + p;
        const uint8_t * vrow = (const uint8_t *) V + (size_t) t * v_nb1 + (size_t) iv2 * v_nb2 + (size_t) iv3 * v_nb3;
        for (int j = 0; j < ndv; ++j) {
            const int d = lane + j * 32;
            if (d >= DV) break;
            float vv;
            if (v_q41) {
                const uint8_t * blk = vrow + (d / 32) * 20;          // q4_1 block: [f16 d][f16 m][16B nibbles]
                const float vd = __half2float(*(const half *)(blk + 0));
                const float vm = __half2float(*(const half *)(blk + 2));
                const int dj = d % 32;
                const uint8_t b = blk[4 + (dj % 16)];
                vv = ((dj < 16) ? (b & 0x0F) : (b >> 4)) * vd + vm;
            } else {
                vv = __half2float(((const half *) vrow)[d]);
            }
            acc[j] = acc[j] * a + p * vv;
        }
        m = mn;
    }
    // attention sink: fold the per-head sink logit into the denominator once. With grid-split only the g==0
    // block folds it (still on warp 0), so the sink is counted exactly once across all partials and the combine
    // kernel stays sink-agnostic. No V contribution.
    if (sinks && wid == 0 && (kv_split == 1 || kv_g == 0)) {
        const float sk = sinks[iq2];
        const float mn = fmaxf(m, sk), a = expf(m - mn), p = expf(sk - mn);
        l = l * a + p;
        for (int j = 0; j < ndv; ++j) acc[j] *= a;
        m = mn;
    }

    // grid-split (kv_split>1): each block emits an un-normalised numerator + (m,l) meta into transient scratch
    // at part_idx; the combine kernel renormalises across the kv_split blocks. kv_split==1: write dst directly.
    const size_t part_idx = (size_t)(iq3 * n_head + iq2) * kv_split + kv_g;
    float * out = (float *)((char *) dst + (size_t)((size_t) iq3 * dst_ne2 * dst_ne1 + iq2 + (size_t) iq1 * dst_ne1) * dst_nb1);
    if (n_split == 1) {                                             // single warp -> block result is this warp's
        if (kv_split == 1) {
            const float Sinv = (l == 0.0f) ? 0.0f : 1.0f / l;
            for (int j = 0; j < ndv; ++j) { const int d = lane + j * 32; if (d < DV) out[d] = acc[j] * Sinv; }
        } else {
            float * pb = parts + part_idx * DV;
            for (int j = 0; j < ndv; ++j) { const int d = lane + j * 32; if (d < DV) pb[d] = acc[j]; }
            if (lane == 0) meta_out[part_idx] = make_float2(m, l);
        }
        return;
    }

    // split-K reduction: combine the per-warp (m, l, VKQ) partials with an online-softmax rescale.
    extern __shared__ float smem[];
    float * s_acc = smem;                  // [n_split][DV]
    float * s_m   = smem + (size_t) n_split * DV;
    float * s_l   = s_m + n_split;
    for (int j = 0; j < ndv; ++j) { const int d = lane + j * 32; if (d < DV) s_acc[(size_t) wid * DV + d] = acc[j]; }
    if (lane == 0) { s_m[wid] = m; s_l[wid] = l; }
    __syncthreads();

    float m_g = -INFINITY;
    for (int w = 0; w < n_split; ++w) m_g = fmaxf(m_g, s_m[w]);
    float l_g = 0.0f;
    for (int w = 0; w < n_split; ++w) l_g += (s_m[w] == -INFINITY) ? 0.0f : s_l[w] * expf(s_m[w] - m_g);
    if (kv_split == 1) {
        const float Sinv = (l_g == 0.0f) ? 0.0f : 1.0f / l_g;
        for (int d = threadIdx.x; d < DV; d += blockDim.x) {
            float a = 0.0f;
            for (int w = 0; w < n_split; ++w) a += (s_m[w] == -INFINITY) ? 0.0f : s_acc[(size_t) w * DV + d] * expf(s_m[w] - m_g);
            out[d] = a * Sinv;
        }
    } else {
        float * pb = parts + part_idx * DV;                        // un-normalised numerator; combine renormalises
        for (int d = threadIdx.x; d < DV; d += blockDim.x) {
            float a = 0.0f;
            for (int w = 0; w < n_split; ++w) a += (s_m[w] == -INFINITY) ? 0.0f : s_acc[(size_t) w * DV + d] * expf(s_m[w] - m_g);
            pb[d] = a;
        }
        if (threadIdx.x == 0) meta_out[part_idx] = make_float2(m_g, l_g);
    }
}

// ---- grid-split combine: renormalise the kv_split per-block partials (numerator + (m,l) meta) into dst with
// an online-softmax rescale. One block per (head, stream); DV threads. Mirrors flash_attn_combine_results. ----
template<int DV_T>
static __global__ void kpc_flash_combine_kernel(
        const float * parts, const float2 * meta_out, float * dst,
        int kv_split, int n_head, int DV_rt, int64_t dst_nb1, int dst_ne1, int dst_ne2) {
    const int iq2 = blockIdx.x;   // head
    const int iq3 = blockIdx.y;   // stream
    const int DV  = DV_T ? DV_T : DV_rt;
    const size_t base = (size_t)(iq3 * n_head + iq2) * kv_split;
    extern __shared__ float2 s_meta[];                             // [kv_split]
    for (int i = threadIdx.x; i < kv_split; i += blockDim.x) s_meta[i] = meta_out[base + i];
    __syncthreads();
    float M = -INFINITY;
    for (int g = 0; g < kv_split; ++g) M = fmaxf(M, s_meta[g].x);
    float denom = 0.0f;
    for (int g = 0; g < kv_split; ++g) denom += (s_meta[g].x == -INFINITY) ? 0.0f : s_meta[g].y * expf(s_meta[g].x - M);
    const float Sinv = (denom == 0.0f) ? 0.0f : 1.0f / denom;
    float * out = (float *)((char *) dst + (size_t)((size_t) iq3 * dst_ne2 * dst_ne1 + iq2) * dst_nb1);
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        float num = 0.0f;
        for (int g = 0; g < kv_split; ++g) num += (s_meta[g].x == -INFINITY) ? 0.0f : parts[(base + g) * DV + d] * expf(s_meta[g].x - M);
        out[d] = num * Sinv;
    }
}

// ---- query-tiled flash-decode read: 1 warp per (head, QUERY-TILE of QT, stream). For prefill the original
// 1-warp-per-query kernel reloads the WHOLE K/V cache once PER QUERY (n_kv x n_q redundant loads) -- the
// dominant cost at long context. Here each key's K nibbles + V row are loaded ONCE into registers and reused
// for all QT queries in the tile, so K/V load traffic drops ~QT x. Single warp only (no split-K) -> used for
// prefill (n_split==1, n_q>1); decode (n_split>1) keeps the original kernel. Same math/parity per query. ----
template<int DK_T, int DV_T, int QT>
static __global__ void kpc_flash_decode_qt_kernel(
        const float * Q, const uint8_t * Kp, const uint8_t * sz0, const uint8_t * V, const half * mask,
        const float * sinks, const int32_t * gid, int gi_stride, float kqscale, float max_bias, float logit_softcap, float * dst,
        int head_dim_rt, int DV_rt, int C_full, int n_kv, int n_head, int n_q, int n_kvh, int n_vh, int n_stream,
        int rk2, int rv2, int rk3, int rv3, bool v_q41,
        int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
        int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
        int64_t sz_nb1, int64_t sz_nb2,
        int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
        int64_t m_nb1, int64_t m_nb2, int64_t m_nb3, int m_ne2, int m_ne3,
        int64_t dst_nb1, int dst_ne1, int dst_ne2, int n_split) {
    const int iq2     = blockIdx.x;                  // query head
    const int iq1base = blockIdx.y * QT;             // first query of this tile
    const int iq3     = blockIdx.z;                  // stream
    if (iq2 >= n_head || iq1base >= n_q || iq3 >= n_stream) return;
    const int lane = threadIdx.x & 31;

    const int head_dim = DK_T ? DK_T : head_dim_rt;
    const int DV       = DV_T ? DV_T : DV_rt;
    constexpr int NQP  = DK_T ? ((DK_T + 31) / 32) : 8;
    constexpr int NACC = DV_T ? ((DV_T + 31) / 32) : 16;
    const int ndv = NACC;

    const int ik2 = iq2 / rk2, iv2 = iq2 / rv2;
    const int ik3 = iq3 / rk3, iv3 = iq3 / rv3;
    const int cbase = ik2 * head_dim;

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const int n_head_log2 = kpc_pow2_floor(n_head);
        const float m0 = exp2f(-(max_bias)        / n_head_log2);
        const float m1 = exp2f(-(max_bias / 2.0f) / n_head_log2);
        slope = (iq2 < n_head_log2) ? powf(m0, iq2 + 1) : powf(m1, 2 * (iq2 - n_head_log2) + 1);
    }

    const int nq_tile = (n_q - iq1base) < QT ? (n_q - iq1base) : QT;   // valid queries in this tile (tail)
    const float * qrow[QT];
    const half  * mp[QT];
    float qp[QT][NQP];
    float corr[QT];
    float acc[QT][NACC];
    float m[QT], l[QT];
    #pragma unroll
    for (int r = 0; r < QT; ++r) {
        const int iq1 = iq1base + r;
        const int rr  = r < nq_tile ? iq1 : iq1base;   // clamp tail lanes to a valid row (results discarded)
        qrow[r] = (const float *)((const char *) Q + (size_t) rr * q_nb1 + (size_t) iq2 * q_nb2 + (size_t) iq3 * q_nb3);
        mp[r]   = mask ? (const half *)((const char *) mask + (size_t) rr * m_nb1 + (size_t)(iq2 % m_ne2) * m_nb2 + (size_t)(iq3 % m_ne3) * m_nb3) : nullptr;
        for (int j = 0; j < ndv; ++j) acc[r][j] = 0.0f;
        m[r] = -INFINITY; l[r] = 0.0f; corr[r] = 0.0f;
    }
    int cur_pool = -1;

    for (int t = 0; t < n_kv; ++t) {
        // skip a key only if EVERY query in the tile masks it (causal: t beyond the tile's last query)
        bool any = false;
        float mv[QT];
        #pragma unroll
        for (int r = 0; r < QT; ++r) { mv[r] = mp[r] ? __half2float(mp[r][t]) : 0.0f; if (r < nq_tile && mv[r] != -INFINITY) any = true; }
        if (!any) continue;

        int pool = gid ? gid[(size_t) ik3 * gi_stride + t] : (t / GGML_KPC_GROUP);
        if (pool < 0) pool = 0;
        if (pool != cur_pool) {                          // refold every query's Q against this group's scale/zp
            const uint8_t * s = sz0 + (size_t) ik3 * sz_nb2 + (size_t) pool * sz_nb1;
            float ss, zmn, szc; kpc_slab_super(s, &ss, &zmn, &szc);
            const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C_full;
            float lc[QT];
            #pragma unroll
            for (int r = 0; r < QT; ++r) lc[r] = 0.0f;
            int ci = 0;
            for (int c = lane; c < head_dim; c += 32, ++ci) {
                const int fc = cbase + c;
                const float sca = qs[fc] * ss, zpc = zmn + qz[fc] * szc;
                #pragma unroll
                for (int r = 0; r < QT; ++r) { qp[r][ci] = qrow[r][c] * sca; lc[r] += qrow[r][c] * zpc; }
            }
            #pragma unroll
            for (int r = 0; r < QT; ++r) { for (int o = 16; o > 0; o >>= 1) lc[r] += __shfl_xor_sync(0xffffffff, lc[r], o); corr[r] = lc[r]; }
            cur_pool = pool;
        }
        // load this key's K nibbles for the lane's channels ONCE, reuse across the tile
        const uint8_t * krow = (const uint8_t *) Kp + (size_t) t * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
        float nbv[NQP]; { int ci = 0; for (int c = lane; c < head_dim; c += 32, ++ci) { const uint8_t nb = krow[c >> 1]; nbv[ci] = (float) ((c & 1) ? (nb >> 4) : (nb & 0x0F)); } }
        // load this key's V row for the lane's elements ONCE
        const uint8_t * vrow = (const uint8_t *) V + (size_t) t * v_nb1 + (size_t) iv2 * v_nb2 + (size_t) iv3 * v_nb3;
        float vv[NACC];
        for (int j = 0; j < ndv; ++j) {
            const int d = lane + j * 32;
            if (d >= DV) { vv[j] = 0.0f; continue; }
            if (v_q41) {
                const uint8_t * blk = vrow + (d / 32) * 20;
                const float vd = __half2float(*(const half *)(blk + 0));
                const float vm = __half2float(*(const half *)(blk + 2));
                const int dj = d % 32;
                const uint8_t b = blk[4 + (dj % 16)];
                vv[j] = ((dj < 16) ? (b & 0x0F) : (b >> 4)) * vd + vm;
            } else {
                vv[j] = __half2float(((const half *) vrow)[d]);
            }
        }
        #pragma unroll
        for (int r = 0; r < QT; ++r) {
            if (r >= nq_tile || mv[r] == -INFINITY) continue;       // this query does not attend key t
            float part = 0.0f;
            int ci = 0;
            for (int c = lane; c < head_dim; c += 32, ++ci) part += qp[r][ci] * nbv[ci];
            for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
            float score = (part + corr[r]) * kqscale;
            if (logit_softcap != 0.0f) score = logit_softcap * tanhf(score / logit_softcap);
            score += slope * mv[r];
            const float mn = fmaxf(m[r], score), a = expf(m[r] - mn), p = expf(score - mn);
            l[r] = l[r] * a + p;
            for (int j = 0; j < ndv; ++j) acc[r][j] = acc[r][j] * a + p * vv[j];
            m[r] = mn;
        }
    }

    #pragma unroll
    for (int r = 0; r < QT; ++r) {
        if (r >= nq_tile) continue;
        if (sinks) {
            const float sk = sinks[iq2];
            const float mn = fmaxf(m[r], sk), a = expf(m[r] - mn), p = expf(sk - mn);
            l[r] = l[r] * a + p;
            for (int j = 0; j < ndv; ++j) acc[r][j] *= a;
            m[r] = mn;
        }
        const int iq1 = iq1base + r;
        float * out = (float *)((char *) dst + (size_t)((size_t) iq3 * dst_ne2 * dst_ne1 + iq2 + (size_t) iq1 * dst_ne1) * dst_nb1);
        const float Sinv = (l[r] == 0.0f) ? 0.0f : 1.0f / l[r];
        for (int j = 0; j < ndv; ++j) { const int d = lane + j * 32; if (d < DV) out[d] = acc[r][j] * Sinv; }
    }
}

// ============================================================================
// Tensor-core prefill path (sm_75+). Built incrementally and PPL-gated:
//   M1 (this kernel, scalar): dequant K(int4 per-chan)+V(q4_1) into f16 smem tiles, online softmax over KN-key
//       tiles, SCALAR f32-accumulate QK and PV. Isolates the KPC-dequant-to-smem path from MMA mechanics.
//   M2/M3 swap the QK then PV matmuls for ggml_cuda_mma. One warp per (head, QM-query tile, stream).
// f16 K/Q inputs with f32 accumulation mirror the f32.f16.f16.f32 MMA, so PPL should track the scalar kernel.
// ============================================================================
#define KPC_QM 16          // queries per block tile (MMA M dim)
#define KPC_KN 16          // keys per inner tile
#define KPC_NW 8           // warps per block: dequant runs on all NW warps (amortizes the ~24KB smem -> more
                           // warps/SM to hide dequant latency); the QK/PV MMA + softmax stay on warp 0.

#ifdef KPC_CLOCKPROF
// dev-only phase profiler (M0b): cumulative cycle counts per phase across all prefill MMA launches.
// [0]=prologue [1]=dequantK [2]=QK-MMA [3]=rescale+Pconvert [4]=PV [5]=epilogue [6]=dequantV [7]=softmax
__device__ unsigned long long g_kpc_prof[8];
#define KPC_PROF(slot) do { if (tid == 0) { long long _n = clock64(); _acc[slot] += (unsigned long long)(_n - _tk); _tk = _n; } } while (0)
#else
#define KPC_PROF(slot)
#endif

template<int DK_T, int DV_T>
static __global__ void kpc_flash_prefill_kernel(
        const float * Q, const uint8_t * Kp, const uint8_t * sz0, const uint8_t * V, const half * mask,
        const float * sinks, const int32_t * gid, int gi_stride, float kqscale, float max_bias, float logit_softcap, float * dst,
        int head_dim_rt, int DV_rt, int C_full, int n_kv, int n_head, int n_q, int n_kvh, int n_vh, int n_stream,
        int rk2, int rv2, int rk3, int rv3, bool v_q41,
        int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
        int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
        int64_t sz_nb1, int64_t sz_nb2,
        int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
        int64_t m_nb1, int64_t m_nb2, int64_t m_nb3, int m_ne2, int m_ne3,
        int64_t dst_nb1, int dst_ne1, int dst_ne2, int n_split) {
    const int HD = DK_T ? DK_T : head_dim_rt;
    const int DV = DV_T ? DV_T : DV_rt;
    const int iq2     = blockIdx.x;                  // head
    const int iq1base = blockIdx.y * KPC_QM;         // first query of this tile
    const int iq3     = blockIdx.z;                  // stream
    if (iq2 >= n_head || iq1base >= n_q || iq3 >= n_stream) return;
    const int tid = threadIdx.x;                     // 0..blockDim.x-1 (NW warps); warp 0 (tid<32) does the MMAs

    const int ik2 = iq2 / rk2, iv2 = iq2 / rv2;
    const int ik3 = iq3 / rk3, iv3 = iq3 / rv3;
    const int cbase = ik2 * HD;
    const int nqv = (n_q - iq1base) < KPC_QM ? (n_q - iq1base) : KPC_QM;   // valid queries (tail)

#ifdef KPC_CLOCKPROF
    unsigned long long _acc[8] = {0,0,0,0,0,0,0,0};
    long long _tk = clock64();
#endif

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const int n_head_log2 = kpc_pow2_floor(n_head);
        const float m0 = exp2f(-(max_bias)        / n_head_log2);
        const float m1 = exp2f(-(max_bias / 2.0f) / n_head_log2);
        slope = (iq2 < n_head_log2) ? powf(m0, iq2 + 1) : powf(m1, 2 * (iq2 - n_head_log2) + 1);
    }

    // smem: floats first (4-byte aligned from the 16-byte smem base) so the f16 tiles that follow are half2-aligned.
    extern __shared__ char kpc_smem[];
    float * Acc = (float *) kpc_smem;                // [QM][DV]  running output accumulator
    float * Sc  = Acc + KPC_QM * DV;                 // [QM][KN]  scores then probs
    float * scaZ= Sc  + KPC_QM * KPC_KN;             // [HD] per-channel scale (current group)
    float * zpcZ= scaZ + HD;                         // [HD] per-channel zp    (current group)
    float * Mx  = zpcZ + HD;                         // [QM] running max
    float * Ln  = Mx  + KPC_QM;                      // [QM] running denom
    float * Av  = Ln  + KPC_QM;                      // [QM] this-tile rescale a
    half  * Qf  = (half  *)(Av + KPC_QM);            // [QM][HD] row-major
    half  * Kf  = Qf  + KPC_QM * HD;                 // [KN][HD] row-major
    half  * Vf  = Kf  + KPC_KN * HD;                 // [DV][KN] d-major (so the PV MMA contracts over keys)
    half  * Pf  = Vf  + DV * KPC_KN;                 // [QM][KN] softmax probs in f16 (PV MMA operand)

    // load Q -> Qf (f16), zero running state + accumulator (all warps cooperate)
    for (int idx = tid; idx < nqv * HD; idx += blockDim.x) {
        const int q = idx / HD, c = idx % HD;
        const float * qr = (const float *)((const char *) Q + (size_t)(iq1base + q) * q_nb1 + (size_t) iq2 * q_nb2 + (size_t) iq3 * q_nb3);
        Qf[q * HD + c] = __float2half(qr[c]);
    }
    for (int q = tid; q < KPC_QM; q += blockDim.x) { Mx[q] = -INFINITY; Ln[q] = 0.0f; }
    for (int idx = tid; idx < KPC_QM * DV; idx += blockDim.x) Acc[idx] = 0.0f;
    __syncthreads();
    KPC_PROF(0);

    for (int kt = 0; kt < n_kv; kt += KPC_KN) {
        const int knv = (n_kv - kt) < KPC_KN ? (n_kv - kt) : KPC_KN;
        // ---- dequant K tile -> Kf (per-channel: nibble*scale + zp; group scale/zp cached in smem) ----
        // Positional tiles (gid==null) are always single-pool (GROUP=32, KN=16, 16-aligned kt); gid tiles are too
        // when the cell->pool map is positional. Single-pool => dequant K over a FLAT (key,channel) grid-stride loop:
        // all NW warps stay busy (the per-key loop left threads >= HD idle at HD=128) and all knv key-rows' global
        // loads are issued concurrently instead of one latency-bound key at a time. Output is bit-identical.
        int pool0 = gid ? gid[(size_t) ik3 * gi_stride + kt] : (kt / GGML_KPC_GROUP);
        if (pool0 < 0) pool0 = 0;
        bool single_pool = true;
        if (gid) {
            for (int k = 1; k < knv; ++k) {
                int p = gid[(size_t) ik3 * gi_stride + kt + k]; if (p < 0) p = 0;
                if (p != pool0) { single_pool = false; break; }
            }
        }
        if (single_pool) {
            const uint8_t * s = sz0 + (size_t) ik3 * sz_nb2 + (size_t) pool0 * sz_nb1;
            float ss, zmn, szc; kpc_slab_super(s, &ss, &zmn, &szc);
            const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C_full;
            for (int c = tid; c < HD; c += blockDim.x) { scaZ[c] = qs[cbase + c] * ss; zpcZ[c] = zmn + qz[cbase + c] * szc; }
            __syncthreads();
            for (int idx = tid; idx < knv * HD; idx += blockDim.x) {
                const int k = idx / HD, c = idx % HD;
                const uint8_t * krow = (const uint8_t *) Kp + (size_t)(kt + k) * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
                const uint8_t nb = krow[c >> 1];
                const float nibble = (float) ((c & 1) ? (nb >> 4) : (nb & 0x0F));
                Kf[k * HD + c] = __float2half(nibble * scaZ[c] + zpcZ[c]);
            }
        } else {
            int tile_pool = -1;                         // rare gid multi-pool tile: per-key serial fallback
            for (int k = 0; k < knv; ++k) {
                const int key = kt + k;
                int pool = gid[(size_t) ik3 * gi_stride + key];
                if (pool < 0) pool = 0;
                if (pool != tile_pool) {
                    const uint8_t * s = sz0 + (size_t) ik3 * sz_nb2 + (size_t) pool * sz_nb1;
                    float ss, zmn, szc; kpc_slab_super(s, &ss, &zmn, &szc);
                    const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C_full;
                    for (int c = tid; c < HD; c += blockDim.x) { scaZ[c] = qs[cbase + c] * ss; zpcZ[c] = zmn + qz[cbase + c] * szc; }
                    __syncthreads(); tile_pool = pool;
                }
                const uint8_t * krow = (const uint8_t *) Kp + (size_t) key * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
                for (int c = tid; c < HD; c += blockDim.x) {
                    const uint8_t nb = krow[c >> 1];
                    const float nibble = (float) ((c & 1) ? (nb >> 4) : (nb & 0x0F));
                    Kf[k * HD + c] = __float2half(nibble * scaZ[c] + zpcZ[c]);
                }
            }
        }
        KPC_PROF(1);                                    // K dequant (no barrier; tid==0 local estimate)
        // ---- dequant V tile -> Vf, D-MAJOR [d][k] (q4_1 or f16); zero padded keys so P=0 * V can't make NaN ----
        for (int idx = tid; idx < DV * KPC_KN; idx += blockDim.x) {
            const int d = idx / KPC_KN, k = idx % KPC_KN;
            float vv = 0.0f;
            if (k < knv) {
                const uint8_t * vrow = (const uint8_t *) V + (size_t)(kt + k) * v_nb1 + (size_t) iv2 * v_nb2 + (size_t) iv3 * v_nb3;
                if (v_q41) {
                    const uint8_t * blk = vrow + (d / 32) * 20;
                    const float vd = __half2float(*(const half *)(blk + 0));
                    const float vm = __half2float(*(const half *)(blk + 2));
                    const int dj = d % 32;
                    const uint8_t b = blk[4 + (dj % 16)];
                    vv = ((dj < 16) ? (b & 0x0F) : (b >> 4)) * vd + vm;
                } else {
                    vv = __half2float(((const half *) vrow)[d]);
                }
            }
            Vf[d * KPC_KN + k] = __float2half(vv);
        }
        __syncthreads();
        KPC_PROF(6);                                    // V dequant

        // ---- QK MMA + per-query softmax: warp 0 only (MMA is warp-collective; lane q owns query q) ----
        if (tid < 32) {
            using namespace ggml_cuda_mma;
            const half2 * Qh2 = (const half2 *) Qf;     // [QM][HD/2] row-major
            const half2 * Kh2 = (const half2 *) Kf;     // [KN][HD/2] row-major
            const int hd2 = HD / 2;
            #pragma unroll
            for (int n = 0; n < KPC_KN / 8; ++n) {      // D[16q x 8k] = Q @ K^T over 16-channel steps, 2 N-tiles
                tile<16, 8, float> D;
                #pragma unroll
                for (int s = 0; s < HD / 16; ++s) {
                    tile<16, 8, half2> A;
                    tile<8,  8, half2> B;
                    load_generic(A, Qh2 + 8 * s,                  hd2);
                    load_generic(B, Kh2 + (8 * n) * hd2 + 8 * s,  hd2);
                    mma(D, A, B);
                }
                #pragma unroll
                for (int l = 0; l < D.ne; ++l) {
                    Sc[D.get_i(l) * KPC_KN + 8 * n + D.get_j(l)] = D.x[l];
                }
            }
            __syncwarp();                               // warp-0 internal: all lanes' Sc writes visible
            KPC_PROF(2);                                // QK MMA (warp-0 collective)
            if (tid < nqv) {
                const int q = tid;
                const int qpos = iq1base + q;
                const half * mp = mask ? (const half *)((const char *) mask + (size_t) qpos * m_nb1 + (size_t)(iq2 % m_ne2) * m_nb2 + (size_t)(iq3 % m_ne3) * m_nb3) : nullptr;
                float tilemax = -INFINITY;
                float proc[KPC_KN];
                for (int k = 0; k < knv; ++k) {
                    const int key = kt + k;
                    float sc = Sc[q * KPC_KN + k] * kqscale;
                    if (logit_softcap != 0.0f) sc = logit_softcap * tanhf(sc / logit_softcap);
                    const float mv = mp ? __half2float(mp[key]) : 0.0f;
                    sc += slope * mv;
                    proc[k] = sc;
                    tilemax = fmaxf(tilemax, sc);
                }
                const float mn = fmaxf(Mx[q], tilemax);
                const float a  = expf(Mx[q] - mn);
                float sump = 0.0f;
                for (int k = 0; k < knv; ++k) { const float p = expf(proc[k] - mn); Sc[q * KPC_KN + k] = p; sump += p; }
                for (int k = knv; k < KPC_KN; ++k) Sc[q * KPC_KN + k] = 0.0f;
                Ln[q] = Ln[q] * a + sump;
                Mx[q] = mn;
                Av[q] = a;
            }
        }
        __syncthreads();                                // Av, Sc(probs), Mx, Ln from warp 0 -> all warps
        KPC_PROF(7);                                    // softmax

        // ---- PV: rescale Acc by this tile's softmax factor (all warps), convert P -> f16, then warp-0 P@V MMA ----
        for (int idx = tid; idx < nqv * DV; idx += blockDim.x) Acc[idx] *= Av[idx / DV];
        for (int idx = tid; idx < KPC_QM * KPC_KN; idx += blockDim.x) {
            const int q = idx / KPC_KN;
            Pf[idx] = (q < nqv) ? __float2half(Sc[idx]) : __float2half(0.0f);
        }
        __syncthreads();
        KPC_PROF(3);
        if (tid < 32) {
            using namespace ggml_cuda_mma;
            const int kn2 = KPC_KN / 2;                                            // half2 cols (= keys/2)
            tile<16, 8, half2> A;                                                  // P: 16 queries x KN keys
            load_generic(A, (const half2 *) Pf, kn2);
            #pragma unroll 1
            for (int m = 0; m < DV / 8; ++m) {                                     // DV/8 output N-tiles of 8
                tile<8, 8, half2> B;                                               // V: 8 d-out x KN keys (d-major)
                load_generic(B, (const half2 *) Vf + 8 * m * kn2, kn2);
                tile<16, 8, float> D;
                mma(D, A, B);
                #pragma unroll
                for (int l = 0; l < D.ne; ++l) {
                    Acc[D.get_i(l) * DV + 8 * m + D.get_j(l)] += D.x[l];
                }
            }
        }
        __syncthreads();                                // Acc from PV done; safe to overwrite Kf/Vf next tile
        KPC_PROF(4);
    }

    // ---- attention sink (fold once per query) + write normalized output ----
    if (tid < nqv) {
        const int q = tid;
        if (sinks) {
            const float sk = sinks[iq2];
            const float mn = fmaxf(Mx[q], sk), a = expf(Mx[q] - mn), p = expf(sk - mn);
            Ln[q] = Ln[q] * a + p;
            Av[q] = a;   // reuse Av to carry the sink rescale for Acc below
            Mx[q] = mn;
        } else {
            Av[q] = 1.0f;
        }
    }
    __syncthreads();
    for (int idx = tid; idx < nqv * DV; idx += blockDim.x) {
        const int q = idx / DV, d = idx % DV;
        const float Sinv = (Ln[q] == 0.0f) ? 0.0f : 1.0f / Ln[q];
        const int iq1 = iq1base + q;
        float * out = (float *)((char *) dst + (size_t)((size_t) iq3 * dst_ne2 * dst_ne1 + iq2 + (size_t) iq1 * dst_ne1) * dst_nb1);
        out[d] = Acc[q * DV + d] * Av[q] * Sinv;
    }
#ifdef KPC_CLOCKPROF
    KPC_PROF(5);
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) atomicAdd(&g_kpc_prof[i], _acc[i]);
    }
#endif
}

// ---- host wrappers (thin launchers; exercised once M3 wires the GPU-native tensors) ----

void ggml_cuda_kpc_flash_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * Kp    = dst->src[1];
    const ggml_tensor * sz    = dst->src[2];
    const ggml_tensor * V     = dst->src[3];
    const ggml_tensor * mask  = dst->src[4];
    const ggml_tensor * gi    = dst->src[5];   // group_index [n_kv, ns] (cell->pool); null -> positional
    const ggml_tensor * sinks = dst->src[6];
    float params[4]; memcpy(params, dst->op_params, sizeof(params));   // kq_scale, max_bias, logit_softcap, n_seq_max

    const int head_dim = Q->ne[0];
    const int n_q      = Q->ne[1];
    const int n_head   = Q->ne[2];
    const int n_stream = Q->ne[3];
    const int DV       = dst->ne[0];
    const int C_full   = (sz->ne[0] - 6) / 2;        // n_embd_k_gqa (all kv-heads share one slab per group)
    const int n_kv     = Kp->ne[1];
    const int n_kvh    = Kp->ne[2];
    const int n_vh     = V->ne[2];
    const int rk2 = n_head / n_kvh, rv2 = n_head / n_vh;
    const int rk3 = n_stream / (int) Kp->ne[3], rv3 = n_stream / (int) V->ne[3];
    const bool v_q41 = V->type == GGML_TYPE_Q4_1;

    const int gi_stride = gi ? (int) (gi->nb[1] / sizeof(int32_t)) : 0;   // per-stream stride (= kv_size)
    // split-K: when there are few query blocks (decode), fan the key loop across warps within the block so the
    // SM stays busy; the per-warp partials are combined in shared memory (no VRAM scratch). At decode the grid
    // is only n_head(*n_stream) blocks, so without this the SMs sit at single-digit warp occupancy and can hide
    // neither the K/V loads nor the per-key reduce -- which is why KPC decode was >2x slower than f16 despite
    // moving ~4x less data. Grow n_split while it fits the hardware limits (1024 threads/block = 32*n_split, and
    // the 48 KB default dynamic-smem budget) and does not oversubscribe the grid. The 1024-thread ceiling caps
    // it at 32 warps; raising the old cap of 8 -> 32 ~doubled decode tg (e.g. 38->76 t/s @ ctx 4096). Prefill
    // has many (head x query) blocks so the grid-budget test keeps it at n_split==1 (the tiled kernel, no reduce).
    const long blocks = (long) n_head * n_q * n_stream;
    int n_split = 1;
    while (32 * (n_split * 2) <= 1024 &&                                                    // threads/block limit
           (size_t)((n_split * 2) * DV + 2 * (n_split * 2)) * sizeof(float) <= 48 * 1024 && // dynamic-smem budget
           blocks * (n_split * 2) <= 2048 &&                                                // don't oversubscribe
           (n_split * 2) <= n_kv) n_split *= 2;
    // register-file cap: 32*n_split threads must fit the SM's 64K-register/block file. Large head_dim kernels
    // (DV=256 uses ~72 regs/thread) blow past it at n_split=32 (1024 thr) -> "too many resources requested for
    // launch". maxThreadsPerBlock is the driver's register-derived ceiling for the chosen instantiation; shrink
    // n_split to fit it (head_dim 64/128 keep 32 warps; 256 backs off).
    if (n_split > 1) {
        cudaFuncAttributes fa{};
        cudaError_t fe;
        if      (head_dim == DV && head_dim ==  64) fe = cudaFuncGetAttributes(&fa, kpc_flash_decode_kernel< 64, 64>);
        else if (head_dim == DV && head_dim == 128) fe = cudaFuncGetAttributes(&fa, kpc_flash_decode_kernel<128,128>);
        else if (head_dim == DV && head_dim == 256) fe = cudaFuncGetAttributes(&fa, kpc_flash_decode_kernel<256,256>);
        else                                        fe = cudaFuncGetAttributes(&fa, kpc_flash_decode_kernel<  0,  0>);
        if (fe == cudaSuccess && fa.maxThreadsPerBlock > 0)
            while (n_split > 1 && 32 * n_split > fa.maxThreadsPerBlock) n_split /= 2;
    }
    // Determinism gate: split-K's per-warp partition + combine (BOTH the in-block n_split here and the grid-level
    // kv_split below) is not bit-reproducible across SEPARATE context instances, so when the cache holds >1 sequence
    // (state restore / continuous batching) a restored or sibling sequence could see another sequence's decode drift
    // by ~0.2 logits. Force single-warp + single-block (deterministic) decode there. Single-stream inference
    // (n_seq_max==1) keeps both split-K paths. Env override for A/B.
    const int  n_seq_max     = (int) params[3];
    const bool kpc_det_decode = (n_seq_max > 1 && !getenv("KPC_FORCE_SPLITK")) || getenv("KPC_NSPLIT1");
    if (kpc_det_decode) n_split = 1;
    const int    nthreads = 32 * n_split;
    const size_t smem = (n_split > 1) ? ((size_t) n_split * DV + 2 * n_split) * sizeof(float) : 0;
    dim3 grid(n_head, n_q, n_stream);
    // specialize on (head_dim, DV) so qp[]/acc[] are compile-time sized (no register spill); <0,0> = runtime.
    #define KPC_FA_ARGS \
        (const float *) Q->data, (const uint8_t *) Kp->data, (const uint8_t *) sz->data, \
        (const uint8_t *) V->data, mask ? (const half *) mask->data : nullptr, \
        sinks ? (const float *) sinks->data : nullptr, gi ? (const int32_t *) gi->data : nullptr, gi_stride, \
        params[0], params[1], params[2], (float *) dst->data, \
        head_dim, DV, C_full, n_kv, n_head, n_q, n_kvh, n_vh, n_stream, \
        rk2, rv2, rk3, rv3, v_q41, \
        Q->nb[1], Q->nb[2], Q->nb[3], \
        Kp->nb[1], Kp->nb[2], Kp->nb[3], \
        sz->nb[1], sz->nb[2], \
        V->nb[1], V->nb[2], V->nb[3], \
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0, \
        mask ? (int) mask->ne[2] : 1, mask ? (int) mask->ne[3] : 1, \
        dst->nb[1], (int) dst->ne[1], (int) dst->ne[2], n_split
    // Prefill (n_split==1, many queries): query-tiled kernel reuses each key's K/V load across QT queries.
    // Decode / runtime / unspecialized: original 1-query-per-warp kernel (with split-K).
    const bool tiled = (n_split == 1 && n_q > 1 && head_dim == DV &&
                        (head_dim == 64 || head_dim == 128 || head_dim == 256));

    // grid-level split-K for decode: the in-block split-K above fills at most one SM per (head,stream) block,
    // but decode has only n_head*n_stream blocks (~n_head/nsm of the SMs), so its attention grows O(n_kv) on a
    // fraction of the GPU. Fan each (head,stream)'s key range across kv_split *blocks* to use all SMs, then
    // renormalise the per-block partials with the combine kernel. The partials live in a small TRANSIENT pool
    // buffer freed at function exit -- the persistent int4 KV-cache VRAM saving (KPC's whole point) is untouched,
    // same idiom as mainline f16 flash-attention's dst_tmp. Only when deep enough that the 2nd launch pays off.
    int kv_split = 1;
    // ...and the grid-level split too: with the in-block split forced to n_split=1 above, leaving the grid-split on
    // (kv_split>1) still drifts the restored sequence ~0.08 across context instances in the unified banded cache at
    // depth (n_kv>=2048). Both split-K paths must be off together under multi-seq for a deterministic decode.
    if (!tiled && n_q == 1 && n_kv >= 2048 && !kpc_det_decode && head_dim == DV &&
        (head_dim == 64 || head_dim == 128 || head_dim == 256)) {
        const int  id  = ggml_cuda_get_device();
        const int  nsm = ggml_cuda_info().devices[id].nsm;
        const long base_blocks = (long) n_head * n_stream;
        int want = (int) ((2L * nsm + base_blocks - 1) / base_blocks);   // ~2 resident blocks per SM
        const int max_by_keys = n_kv / (32 * n_split);                   // keep >= 32 keys per warp-split per block
        if (want > max_by_keys) want = max_by_keys;
        static const int kv_cap = []{ const char * e = getenv("KPC_KVSPLIT_MAX"); return e ? atoi(e) : 0; }();
        if (kv_cap > 0 && want > kv_cap) want = kv_cap;                  // A/B knob: =1 forces the no-grid-split oracle
        if (want > 1) kv_split = want;
    }

    ggml_cuda_pool_alloc<float>  parts_buf(ctx.pool());
    ggml_cuda_pool_alloc<float2> meta_buf (ctx.pool());
    float  * parts = nullptr;
    float2 * meta  = nullptr;
    if (kv_split > 1) {
        const size_t nparts = (size_t) kv_split * n_head * n_stream;
        parts = parts_buf.alloc(nparts * DV);
        meta  = meta_buf.alloc(nparts);
    }
    const dim3 gridR = (kv_split > 1) ? dim3(n_head, kv_split, n_stream) : grid;
    #define KPC_FA_DEC , parts, meta, kv_split

    // tensor-core prefill path. cc>=Turing is a hard CORRECTNESS floor: the MMA intrinsics compile to NO_DEVICE_CODE
    // (trap) on older arches (Pascal etc.), so MMA must never *launch* there -- those devices fall back to the
    // query-tiled scalar kernel. On capable hardware MMA wins once there are enough keys to amortise the f16 staging
    // (n_kv>=2048; below that the query-tiled kernel is faster AND lossless -- see the crossover in the perf handoff).
    // KPC_MMA tri-state override: unset = auto(n_kv>=2048); 1 = force MMA at any n_kv; 0 = force query-tiled.
    static const int kpc_mma = []{ const char * e = getenv("KPC_MMA"); return e ? atoi(e) : -1; }();
    bool use_mma = false;
    if (tiled) {
        const int  dev_cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const bool cc_mma = dev_cc >= GGML_CUDA_CC_TURING;
        use_mma = cc_mma && ((kpc_mma == 1) || (kpc_mma == -1 && n_kv >= 2048));
        if (kpc_mma == 1 && !cc_mma) {                              // forced on hardware that can't run MMA
            static int warned = 0;
            if (!warned) { warned = 1;
                fprintf(stderr, "KPC: KPC_MMA=1 but device cc %d < Turing (750) -- using query-tiled kernel\n", dev_cc); }
        }
    }
    if (use_mma) {
        dim3 gridM(n_head, (n_q + KPC_QM - 1) / KPC_QM, n_stream);
        const size_t smem_m = (size_t)(KPC_QM * head_dim + KPC_KN * head_dim + DV * KPC_KN + KPC_QM * KPC_KN) * sizeof(half)
                            + (size_t)(KPC_QM * DV + KPC_QM * KPC_KN + 2 * head_dim + 3 * KPC_QM) * sizeof(float);
        const int nthr_m = 32 * KPC_NW;
        if      (head_dim ==  64) kpc_flash_prefill_kernel< 64, 64><<<gridM, nthr_m, smem_m, ctx.stream()>>>(KPC_FA_ARGS);
        else if (head_dim == 128) kpc_flash_prefill_kernel<128,128><<<gridM, nthr_m, smem_m, ctx.stream()>>>(KPC_FA_ARGS);
        else                      kpc_flash_prefill_kernel<256,256><<<gridM, nthr_m, smem_m, ctx.stream()>>>(KPC_FA_ARGS);
#ifdef KPC_CLOCKPROF
        {   static int pc = 0;
            if ((++pc % 64) == 0) {                       // periodic cumulative-ratio dump (dev build only)
                cudaStreamSynchronize(ctx.stream());
                unsigned long long h[8] = {0,0,0,0,0,0,0,0};
                cudaMemcpyFromSymbol(h, g_kpc_prof, sizeof(h));
                unsigned long long tot = h[0]+h[1]+h[2]+h[3]+h[4]+h[5]+h[6]+h[7];
                if (tot) fprintf(stderr, "[KPC-PROF] prologue=%.1f dequantK=%.1f dequantV=%.1f QKmma=%.1f softmax=%.1f rescale=%.1f pv=%.1f epilogue=%.1f  (tot=%llu)\n",
                    100.0*h[0]/tot, 100.0*h[1]/tot, 100.0*h[6]/tot, 100.0*h[2]/tot, 100.0*h[7]/tot, 100.0*h[3]/tot, 100.0*h[4]/tot, 100.0*h[5]/tot, tot);
            }
        }
#endif
    }
    else if (tiled) {
        const int QT = 3;
        dim3 gridT(n_head, (n_q + QT - 1) / QT, n_stream);
        if      (head_dim ==  64) kpc_flash_decode_qt_kernel< 64, 64, 3><<<gridT, 32, 0, ctx.stream()>>>(KPC_FA_ARGS);
        else if (head_dim == 128) kpc_flash_decode_qt_kernel<128,128, 3><<<gridT, 32, 0, ctx.stream()>>>(KPC_FA_ARGS);
        else                      kpc_flash_decode_qt_kernel<256,256, 3><<<gridT, 32, 0, ctx.stream()>>>(KPC_FA_ARGS);
    }
    else if (head_dim == DV && head_dim ==  64) kpc_flash_decode_kernel< 64, 64><<<gridR, nthreads, smem, ctx.stream()>>>(KPC_FA_ARGS KPC_FA_DEC);
    else if (head_dim == DV && head_dim == 128) kpc_flash_decode_kernel<128,128><<<gridR, nthreads, smem, ctx.stream()>>>(KPC_FA_ARGS KPC_FA_DEC);
    else if (head_dim == DV && head_dim == 256) kpc_flash_decode_kernel<256,256><<<gridR, nthreads, smem, ctx.stream()>>>(KPC_FA_ARGS KPC_FA_DEC);
    else                                        kpc_flash_decode_kernel<  0,  0><<<gridR, nthreads, smem, ctx.stream()>>>(KPC_FA_ARGS KPC_FA_DEC);

    if (kv_split > 1) {                                              // renormalise the kv_split partials into dst
        const int    cthreads = DV < 1024 ? DV : 1024;
        const size_t csmem    = (size_t) kv_split * sizeof(float2);
        dim3 gridC(n_head, n_stream);
        #define KPC_COMBINE_ARGS parts, meta, (float *) dst->data, kv_split, n_head, DV, dst->nb[1], (int) dst->ne[1], (int) dst->ne[2]
        if      (DV ==  64) kpc_flash_combine_kernel< 64><<<gridC, cthreads, csmem, ctx.stream()>>>(KPC_COMBINE_ARGS);
        else if (DV == 128) kpc_flash_combine_kernel<128><<<gridC, cthreads, csmem, ctx.stream()>>>(KPC_COMBINE_ARGS);
        else                kpc_flash_combine_kernel<256><<<gridC, cthreads, csmem, ctx.stream()>>>(KPC_COMBINE_ARGS);
        #undef KPC_COMBINE_ARGS
    }
    #undef KPC_FA_DEC
    #undef KPC_FA_ARGS
}

// shared member scan: find this (slot,g) block's new members (k_cur tokens with pos/32==g and this slot),
// whether a later group exists (g is then not the slot's open group), and the per-member token/cell indices.
// Fills the caller's shared newmask/has_later/tok_of/cell_of and barriers so they are ready on return.
static __device__ __forceinline__ void kpc_scan_members(
        const int32_t * kpc_seq, const int32_t * kpc_pos, const int64_t * k_idxs,
        int g, int slot, int slot_shift, int nt,
        uint32_t * s_newmask, int * s_has_later, int * tok_of, int64_t * cell_of) {
    if (threadIdx.x == 0) { *s_newmask = 0; *s_has_later = 0; }
    if ((int) threadIdx.x < GGML_KPC_GROUP) { tok_of[threadIdx.x] = -1; cell_of[threadIdx.x] = -1; }
    __syncthreads();
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
        if ((int) ((uint32_t) kpc_seq[i] >> slot_shift) != slot) continue;
        const int tg = kpc_pos[i] / GGML_KPC_GROUP;
        if (tg > g) { *s_has_later = 1; continue; }             // a later token exists -> g is not the open group
        if (tg < g) continue;
        const int w = kpc_pos[i] % GGML_KPC_GROUP;
        tok_of[w]  = i;
        cell_of[w] = k_idxs[i];
        atomicOr(s_newmask, 1u << w);
    }
    __syncthreads();
}

// device write (group-parallel seal). ONE block per (logical group g, staging slot). Each block gathers the
// group's members once -- this ubatch's tokens (from k_cur) plus any previously-staged members (from k_resid
// when g is the slot's open group) -- computes the per-channel scale/zp a single time, encodes the scalezp
// slab and packs every member into its GLOBAL K cell (k_idxs for new, k_resid_slots for staged). Uses only
// on-chip shared memory (no device scratch -> no extra VRAM). Multi-stream: stream = seq/n_seqps,
// pool = (seq%n_seqps)*band_size + g%band_size (mirrors the CPU pool_of). No rescue/virtualization.
// NOTE: this kernel only READS the staging tensors (staged_group/staged_mask/k_resid/resid_slots) -- the
// staging UPDATE for the next ubatch is a SEPARATE launch (kpc_write_stage_kernel) so the seal never races a
// concurrent staging write (the open-group block used to overwrite staging that sibling blocks were still
// reading -> non-deterministic corruption, exposed by Blackwell's independent thread scheduling).
static __global__ void kpc_write_kernel(
        const float * k_cur, const int32_t * kpc_seq, const int32_t * kpc_pos, const int64_t * k_idxs,
        const half * k_resid, uint8_t * scalezp, const int32_t * resid_slots,
        const int32_t * staged_group, const int32_t * staged_mask,
        uint8_t * K, int C, int nt, int slot_shift, int seq_mask, int L,
        int n_seqps, int band_size, int64_t sz_nb1, int64_t sz_nb2, int64_t k_nb1,
        int64_t kc_nb0, int64_t kc_nb1) {
    // k_cur may be a non-contiguous view (e.g. ALiBi models have no RoPE op to repack K) -> read it via strides,
    // mirroring the CPU write. Reading it as flat [tok*C+c] silently grabs wrong channels -> inflated seal range.
    #define KPC_KC(tk, ch) (*(const float *)((const char *) k_cur + (size_t)(tk)*kc_nb1 + (size_t)(ch)*kc_nb0))
    const int g    = blockIdx.x;     // logical 32-token group
    const int slot = blockIdx.y;     // staging slot (== seq for the supported non-virtualized cache)
    if (slot >= L) return;
    const int krow = C / 2;

    extern __shared__ float smem[];  // scale[C], zp[C]
    float * scale = smem; float * zp = smem + C;
    __shared__ uint32_t newmask;
    __shared__ int      has_later;
    __shared__ int      tok_of[GGML_KPC_GROUP];    // k_cur token index that wrote within-group slot w (-1 none)
    __shared__ int64_t  cell_of[GGML_KPC_GROUP];   // global K cell for a new member w
    kpc_scan_members(kpc_seq, kpc_pos, k_idxs, g, slot, slot_shift, nt, &newmask, &has_later, tok_of, cell_of);

    const uint32_t omask   = (g == staged_group[slot]) ? (uint32_t) staged_mask[slot] : 0u;   // staged members of g
    const uint32_t members = newmask | omask;
    if (members == 0) return;

    const int sb   = slot;                                      // slot == seq (non-virtualized)
    const int st   = sb / n_seqps;                              // physical stream
    const int pool = (sb % n_seqps) * band_size + (g % band_size);
    const half * rsd = k_resid + (size_t) slot * C * GGML_KPC_GROUP;

    // per-channel min/max over members (new from k_cur f32, staged-only from k_resid f16)
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float mn = INFINITY, mx = -INFINITY;
        for (int w = 0; w < GGML_KPC_GROUP; ++w) {
            if (!(members & (1u << w))) continue;
            float v = (newmask & (1u << w)) ? KPC_KC(tok_of[w], c)
                                            : __half2float(rsd[(size_t) w * C + c]);
            mn = fminf(mn, v); mx = fmaxf(mx, v);
        }
        float sc = (mx - mn) / 15.0f; if (sc == 0.0f) sc = 1.0f;
        scale[c] = sc; zp[c] = mn;
    }
    __syncthreads();
    uint8_t * s = scalezp + (size_t) st * sz_nb2 + (size_t) pool * sz_nb1;
    if (threadIdx.x == 0) {
        float smax = 0.0f; for (int c = 0; c < C; ++c) smax = fmaxf(smax, scale[c]);
        float ss = smax / 255.0f; if (ss == 0.0f) ss = 1.0f;
        float zmn = INFINITY, zmx = -INFINITY; for (int c = 0; c < C; ++c) { zmn = fminf(zmn, zp[c]); zmx = fmaxf(zmx, zp[c]); }
        float sz = (zmx - zmn) / 255.0f; if (sz == 0.0f) sz = 1.0f;
        *(half *)(s + 0) = __float2half(ss); *(half *)(s + 2) = __float2half(zmn); *(half *)(s + 4) = __float2half(sz);
        uint8_t * qs = s + 6; uint8_t * qz = s + 6 + C;
        for (int c = 0; c < C; ++c) {
            int a = (int)(scale[c]/ss + 0.5f); a = a<0?0:(a>255?255:a);
            int b = (int)((zp[c]-zmn)/sz + 0.5f); b = b<0?0:(b>255?255:b);
            qs[c] = (uint8_t) a; qz[c] = (uint8_t) b;
        }
    }
    __syncthreads();
    float ss, zmn, sz; kpc_slab_super(s, &ss, &zmn, &sz);
    const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C;
    // pack every member into its GLOBAL K cell, one whole byte per thread (no nibble RMW race)
    for (int w = 0; w < GGML_KPC_GROUP; ++w) {
        if (!(members & (1u << w))) continue;
        const bool    isnew = newmask & (1u << w);
        const int64_t cell  = isnew ? cell_of[w] : (int64_t) resid_slots[slot * GGML_KPC_GROUP + w];
        uint8_t * row = K + (size_t) cell * k_nb1;
        for (int b = threadIdx.x; b < krow; b += blockDim.x) {
            const int c0 = 2 * b, c1 = 2 * b + 1;
            float v0 = isnew ? KPC_KC(tok_of[w], c0) : __half2float(rsd[(size_t) w * C + c0]);
            float v1 = isnew ? KPC_KC(tok_of[w], c1) : __half2float(rsd[(size_t) w * C + c1]);
            int q0 = (int)((v0 - (zmn + qz[c0]*sz)) / (qs[c0]*ss) + 0.5f);
            int q1 = (int)((v1 - (zmn + qz[c1]*sz)) / (qs[c1]*ss) + 0.5f);
            q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
            q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
            row[b] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

// staging update (SEPARATE launch, runs after kpc_write_kernel on the same stream so the seal has finished and
// its reads of the OLD staging are complete). Only the slot's OPEN group (no later token this ubatch) persists
// the residual window + staged_group/staged_mask for the next ubatch. Non-open blocks bail before touching the
// staging tensors, so exactly one block per slot writes them -> no inter-block race.
static __global__ void kpc_write_stage_kernel(
        const float * k_cur, const int32_t * kpc_seq, const int32_t * kpc_pos, const int64_t * k_idxs,
        half * k_resid, int32_t * resid_slots, int32_t * staged_group, int32_t * staged_mask,
        int C, int nt, int slot_shift, int L, int64_t kc_nb0, int64_t kc_nb1) {
    const int g    = blockIdx.x;
    const int slot = blockIdx.y;
    if (slot >= L) return;

    __shared__ uint32_t newmask;
    __shared__ int      has_later;
    __shared__ int      tok_of[GGML_KPC_GROUP];
    __shared__ int64_t  cell_of[GGML_KPC_GROUP];
    kpc_scan_members(kpc_seq, kpc_pos, k_idxs, g, slot, slot_shift, nt, &newmask, &has_later, tok_of, cell_of);

    if (has_later) return;                                      // not the open group -> never touches staging
    const uint32_t omask   = (g == staged_group[slot]) ? (uint32_t) staged_mask[slot] : 0u;
    const uint32_t members = newmask | omask;
    if (members == 0) return;

    half * rsd = k_resid + (size_t) slot * C * GGML_KPC_GROUP;
    for (int w = 0; w < GGML_KPC_GROUP; ++w) {
        if (!(newmask & (1u << w))) continue;                  // staged members already have residual + slot
        for (int c = threadIdx.x; c < C; c += blockDim.x) rsd[(size_t) w * C + c] = __float2half(KPC_KC(tok_of[w], c));
        if (threadIdx.x == 0) resid_slots[slot * GGML_KPC_GROUP + w] = (int32_t) cell_of[w];
    }
    if (threadIdx.x == 0) { staged_group[slot] = g; staged_mask[slot] = (int32_t) members; }
}

void ggml_cuda_kpc_write(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k_cur   = dst->src[0];
    const ggml_tensor * scalezp = dst->src[1];
    const ggml_tensor * k_resid = dst->src[2];
    const ggml_tensor * k_idxs  = dst->src[3];
    const ggml_tensor * rslots  = dst->src[5];
    const ggml_tensor * sgrp    = dst->src[6];
    const ggml_tensor * smask   = dst->src[7];
    const ggml_tensor * kseq    = dst->src[8];
    const ggml_tensor * kpos    = dst->src[9];
    int32_t pr[16]; memcpy(pr, dst->op_params, sizeof(pr));
    const int   C         = k_cur->ne[0];
    const int   nt        = k_cur->ne[1];
    const int   L         = sgrp->ne[0];               // staging-slot count
    const int   n_seq_max = pr[0];
    const int   n_stream  = scalezp->ne[2];
    const int   ng_max    = scalezp->ne[1];
    const int   n_seqps   = n_seq_max / n_stream;
    const int   band_size = ng_max / n_seqps;
    const size_t smem = 2 * C * sizeof(float);
    dim3 grid(ng_max, L, 1);   // one block per (logical group, staging slot); empty blocks early-return
    kpc_write_kernel<<<grid, 256, smem, ctx.stream()>>>(
        (const float *) k_cur->data, (const int32_t *) kseq->data, (const int32_t *) kpos->data,
        (const int64_t *) k_idxs->data, (half *) k_resid->data, (uint8_t *) scalezp->data,
        (int32_t *) rslots->data, (int32_t *) sgrp->data, (int32_t *) smask->data, (uint8_t *) dst->data,
        C, nt, GGML_KPC_SLOT_SHIFT, GGML_KPC_SEQ_MASK, L,
        n_seqps, band_size, scalezp->nb[1], scalezp->nb[2], dst->nb[1], k_cur->nb[0], k_cur->nb[1]);
    // SECOND launch: persist the staging window. Serialized after the seal on the same stream, so the seal's
    // reads of the OLD staging are done before this writes the NEW staging -> no inter-block staging race.
    kpc_write_stage_kernel<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) k_cur->data, (const int32_t *) kseq->data, (const int32_t *) kpos->data,
        (const int64_t *) k_idxs->data, (half *) k_resid->data,
        (int32_t *) rslots->data, (int32_t *) sgrp->data, (int32_t *) smask->data,
        C, nt, GGML_KPC_SLOT_SHIFT, L, k_cur->nb[0], k_cur->nb[1]);
}
#undef KPC_KC

// ---- dequant: packed int4 K [C,n_kv,ns] -> f16, positional pool = cell>>5 per stream (RoPE shift / state) ----
static __global__ void kpc_dequant_kernel(
        const uint8_t * pk, const uint8_t * sz0, const int32_t * gid, int gi_stride, half * dst,
        int C, int n_kv, int ns, int64_t k_nb1, int64_t k_nb2, int64_t sz_nb1, int64_t sz_nb2,
        int64_t d_nb1, int64_t d_nb2) {
    const int t = blockIdx.x;
    const int s = blockIdx.y;
    if (t >= n_kv || s >= ns) return;
    int pool = gid ? gid[(size_t) s * gi_stride + t] : (t / GGML_KPC_GROUP);   // cell->pool (positional if no map)
    if (pool < 0) pool = 0;
    const uint8_t * slab = sz0 + (size_t) s * sz_nb2 + (size_t) pool * sz_nb1;
    float ss, zmn, szc; kpc_slab_super(slab, &ss, &zmn, &szc);
    const uint8_t * qs = slab + 6; const uint8_t * qz = slab + 6 + C;
    const uint8_t * row  = pk  + (size_t) s * k_nb2 + (size_t) t * k_nb1;
    half          * drow = (half *)((char *) dst + (size_t) s * d_nb2 + (size_t) t * d_nb1);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        drow[c] = __float2half(kpc_deq_nib(row[c >> 1], c, qs[c] * ss, zmn + qz[c] * szc));
    }
}

void ggml_cuda_kpc_dequant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * pk = dst->src[0];
    const ggml_tensor * sz = dst->src[1];
    const ggml_tensor * gi = dst->src[2];   // group_index [n_kv, ns] (cell->pool); null -> positional
    const int C    = pk->ne[0];
    const int n_kv = pk->ne[1];
    const int ns   = pk->ne[2];
    const int gi_stride = gi ? (int) (gi->nb[1] / sizeof(int32_t)) : 0;
    dim3 grid(n_kv, ns, 1);
    kpc_dequant_kernel<<<grid, 256, 0, ctx.stream()>>>(
        (const uint8_t *) pk->data, (const uint8_t *) sz->data, gi ? (const int32_t *) gi->data : nullptr, gi_stride,
        (half *) dst->data, C, n_kv, ns, pk->nb[1], pk->nb[2], sz->nb[1], sz->nb[2], dst->nb[1], dst->nb[2]);
}

// ---- requant roped f32 K [C,n_kv,ns] -> packed int4 + scalezp, in place. One block per (pool, stream);
// cells are bucketed by group_index[cell]==pool (mirrors the CPU requant), so a RoPE shift that regroups
// cells re-encodes each referenced pool consistently. min/max -> scale=(mx-mn)/15 + int8 double-quant slab. ----
static __global__ void kpc_requant_kernel(
        const float * roped, const int32_t * gid, int gi_stride, uint8_t * pk, uint8_t * sz0,
        int C, int n_kv, int ns, int64_t r_nb1, int64_t r_nb2, int64_t k_nb1, int64_t k_nb2,
        int64_t sz_nb1, int64_t sz_nb2) {
    const int p = blockIdx.x;          // pool
    const int s = blockIdx.y;          // stream
    if (s >= ns) return;
    const int krow = C / 2;
    const int32_t * gid_s = gid + (size_t) s * gi_stride;
    extern __shared__ float smem[];    // scale[C], zp[C]
    float * scale = smem; float * zp = smem + C;
    const float * rbase = (const float *)((const char *) roped + (size_t) s * r_nb2);

    __shared__ int n_cells;
    if (threadIdx.x == 0) { n_cells = 0; for (int i = 0; i < n_kv; ++i) if (gid_s[i] == p) ++n_cells; }
    __syncthreads();
    if (n_cells == 0) return;          // pool unused -> leave its slab/K untouched

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float mn = INFINITY, mx = -INFINITY;
        for (int i = 0; i < n_kv; ++i) {
            if (gid_s[i] != p) continue;
            const float v = *(const float *)((const char *) rbase + (size_t) i * r_nb1 + (size_t) c * sizeof(float));
            mn = fminf(mn, v); mx = fmaxf(mx, v);
        }
        float sc = (mx - mn) / 15.0f; if (sc == 0.0f) sc = 1.0f;
        scale[c] = sc; zp[c] = mn;
    }
    __syncthreads();
    uint8_t * slab = sz0 + (size_t) s * sz_nb2 + (size_t) p * sz_nb1;
    if (threadIdx.x == 0) {
        float smax = 0.0f; for (int c = 0; c < C; ++c) smax = fmaxf(smax, scale[c]);
        float ss = smax / 255.0f; if (ss == 0.0f) ss = 1.0f;
        float zmn = INFINITY, zmx = -INFINITY; for (int c = 0; c < C; ++c) { zmn = fminf(zmn, zp[c]); zmx = fmaxf(zmx, zp[c]); }
        float sz = (zmx - zmn) / 255.0f; if (sz == 0.0f) sz = 1.0f;
        *(half *)(slab + 0) = __float2half(ss); *(half *)(slab + 2) = __float2half(zmn); *(half *)(slab + 4) = __float2half(sz);
        uint8_t * qs = slab + 6; uint8_t * qz = slab + 6 + C;
        for (int c = 0; c < C; ++c) {
            int a = (int)(scale[c]/ss + 0.5f); a = a<0?0:(a>255?255:a);
            int b = (int)((zp[c]-zmn)/sz + 0.5f); b = b<0?0:(b>255?255:b);
            qs[c] = (uint8_t) a; qz[c] = (uint8_t) b;
        }
    }
    __syncthreads();
    float ss, zmn, sz; kpc_slab_super(slab, &ss, &zmn, &sz);
    const uint8_t * qs = slab + 6; const uint8_t * qz = slab + 6 + C;
    for (int i = 0; i < n_kv; ++i) {   // repack each pool cell, one whole byte per thread (no nibble RMW race)
        if (gid_s[i] != p) continue;
        uint8_t     * row = pk + (size_t) s * k_nb2 + (size_t) i * k_nb1;
        const float * rc  = (const float *)((const char *) rbase + (size_t) i * r_nb1);
        for (int b = threadIdx.x; b < krow; b += blockDim.x) {
            float d0 = qs[2*b]  *ss; if (d0 == 0.0f) d0 = 1.0f;   // match CPU requant: tiny scale underflows to 0 (kpc.cpp)
            float d1 = qs[2*b+1]*ss; if (d1 == 0.0f) d1 = 1.0f;
            int q0 = (int)((rc[2*b]   - (zmn + qz[2*b]  *sz)) / d0 + 0.5f);
            int q1 = (int)((rc[2*b+1] - (zmn + qz[2*b+1]*sz)) / d1 + 0.5f);
            q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
            q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
            row[b] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

void ggml_cuda_kpc_requant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * roped = dst->src[0];   // f32 [C, n_kv, ns]
    const ggml_tensor * gi    = dst->src[1];   // group_index [n_kv, ns] (cell->pool)
    const ggml_tensor * sz    = dst->src[2];   // scalezp (in place)
    const int C    = roped->ne[0];
    const int n_kv = roped->ne[1];
    const int ns   = roped->ne[2];
    const int ng   = (int) sz->ne[1];          // pool count (ng_max)
    const int gi_stride = (int) (gi->nb[1] / sizeof(int32_t));
    const size_t smem = 2 * C * sizeof(float);
    dim3 grid(ng, ns, 1);
    kpc_requant_kernel<<<grid, 256, smem, ctx.stream()>>>(
        (const float *) roped->data, (const int32_t *) gi->data, gi_stride, (uint8_t *) dst->data, (uint8_t *) sz->data,
        C, n_kv, ns, roped->nb[1], roped->nb[2], dst->nb[1], dst->nb[2], sz->nb[1], sz->nb[2]);
}

bool ggml_cuda_kpc_supported(const ggml_tensor * op) {
    // GPU-native KPC: READ + WRITE + DEQUANT + REQUANT all on the GPU (K/scalezp/k_resid/staging are
    // device-resident and mutated in place). The shift's dequant->rope->requant therefore stays on-device
    // (requant writes the packed K in place, which a CPU op cannot do for a device tensor).
    if (getenv("KPC_CUDA_OFF")) return false;   // DEBUG: force KPC onto CPU for A/B comparison
    return op->op == GGML_OP_KPC_FLASH_ATTN || op->op == GGML_OP_KPC_WRITE ||
           op->op == GGML_OP_KPC_DEQUANT    || op->op == GGML_OP_KPC_REQUANT;
}
