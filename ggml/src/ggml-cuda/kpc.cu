#include "kpc.cuh"
#include "mma.cuh"
#include "fattn.cuh"
#include "convert.cuh"

// GPU-native per-channel int4 K-cache.
//
// KPC_FLASH_ATTN dst->src: [0]=Q f32, [1]=K_packed int4 (KPC4_1), [2]=scalezp u8 [SZ, kv/32, ns]
//   (group = token>>5), [3]=V (q4_1/f16), [4]=mask, [5]=group_index, [6]=sinks.
//   op_params: kq_scale, max_bias, logit_softcap.
// KPC_WRITE seals a full 32-token group into K_packed + scalezp and resets the window.

// prefill: dequant the int4 K-cache (+ q4_1 V) to an f16 scratch, then run the stock flash-attn kernel.
static bool kpc_flash_prefill_stock(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * Q, const ggml_tensor * Kp, const ggml_tensor * sz, const ggml_tensor * V,
        const ggml_tensor * mask, const ggml_tensor * gi, const ggml_tensor * sinks, const float * params,
        int head_dim, int DV, int C_full, int n_kv, int n_kvh, int n_vh, int n_stream, bool v_q41);

// per-channel int4 dequant: c even -> low nibble, c odd -> high.  v = q*scale[c] + zp[c]
static __device__ __forceinline__ float kpc_deq_nib(uint8_t nb, int c, float s, float z) {
    const int q = (c & 1) ? (nb >> 4) : (nb & 0x0F);
    return q * s + z;
}

// largest power of two <= n (ALiBi n_head_log2). Integer form: the float log2f/floorf can round 2^k to k-1
// and pick the wrong slope for power-of-2 head counts; the slope is computed in-kernel so it must be exact.
static __device__ __forceinline__ int kpc_pow2_floor(int n) { int p = 1; while ((p << 1) <= n) p <<= 1; return p; }

// read the three fp16 super-params from a group slab
static __device__ __forceinline__ void kpc_slab_super(const uint8_t * s, float * ss, float * zmn, float * sz) {
    half h;
    h = *(const half *)(s + 0); *ss  = __half2float(h);
    h = *(const half *)(s + 2); *zmn = __half2float(h);
    h = *(const half *)(s + 4); *sz  = __half2float(h);
}

// flash-decode read (mirrors the CPU kpc flash-attn indexing): 1 block per (head, query, stream); online softmax over sealed K, q4_1/f16 V.
// K is [head_dim, n_kv, n_kvh, ns] sliced by kv-head via k_nb2; the scalezp slab carries all
// C_full=n_embd_k_gqa channels, so the head's slice starts at cbase=ik2*head_dim.
// DK_T / DV_T > 0 specialize head_dim / DV so qp[]/acc[] are constant-bound (no local-memory spill);
// DK_T == 0 falls back to the runtime values.
// Decode is occupancy/latency-bound: forcing the register allocator down with __launch_bounds__ (min 3 blocks/SM
// -> ~40 regs from the unconstrained ~56) raises occupancy and decode throughput by ~10% on Blackwell (measured
// deepseek-coder hd128: 86.6 -> 95.6 t/s). The bound only changes register allocation, not arithmetic, so output is
// bit-identical. The #if gates the *attribute's presence* per compiled arch: only the sm_120 slice carries it; every
// older arch (incl. P40 sm_61, the kernel's only decode path there) compiles the original unbounded kernel verbatim,
// so those SASS slices are byte-identical to before. The bound caps registers to ~42/thread (3x512 threads/SM);
// the read kernel uses ~40 (measured sm_120), so it FITS without spilling on the same-register-file Ada-class arches
// -> applied to Ada+ (Ada/Hopper/Blackwell), the floor where the hint helps without risk. Turing/Ampere keep the
// compiler default (architecturally further; un-gate after on-hardware validation).
// The min-blocks/SM hint (3) caps registers to ~40 to raise occupancy: a win for hd<=128 (deepseek hd128 spills only
// 24B), but at hd256 the kernel needs 24 register-arrays (qp/kb/kbn/acc, each NQP=NACC=8) and 40 regs forces 120B of
// local-memory spill, hammered every key in the latency-bound loop -> hd256 in-place was 191us/step vs the materialize
// pipeline's 154us. So make the hint DV-aware: hd256 drops the cap (min 1 block/SM) so the allocator can keep the
// arrays in registers; hd<=128 keeps the occupancy cap. Template-driven so it specializes per instantiation.
template<int DV_T> struct kpc_dec_bounds { static constexpr int min_blocks = (DV_T >= 256) ? 1 : 3; };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
#  define KPC_DEC_LAUNCH_BOUNDS __launch_bounds__(512, kpc_dec_bounds<DV_T>::min_blocks)
#else
#  define KPC_DEC_LAUNCH_BOUNDS
#endif
template<int DK_T, int DV_T>
static __global__ void KPC_DEC_LAUNCH_BOUNDS kpc_flash_decode_kernel(
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
    // grid-split (kv_split>1): blockIdx.y is the KV super-block, query is fixed at iq1=0 and the block emits a
    // partial for the combine kernel. kv_split==1: blockIdx.y is the query position, write dst directly.
    const int kv_g = (kv_split > 1) ? blockIdx.y : 0;
    const int iq1  = (kv_split > 1) ? 0          : (int) blockIdx.y;   // query position
    const int iq3  = blockIdx.z;   // stream
    if (iq2 >= n_head || iq1 >= n_q || iq3 >= n_stream) return;
    const int wid  = threadIdx.x >> 5;                       // warp = key-split partition (split-K)
    const int lane = threadIdx.x & 31;                       // n_split warps per (head, query, stream)

    const int head_dim = DK_T ? DK_T : head_dim_rt;          // compile-time when specialized
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
    const int ndv = NACC;
    float acc[NACC];
    for (int j = 0; j < ndv; ++j) acc[j] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    // 32 keys share one scalezp slab, so the per-channel Q*scale fold + zp correction is constant across a
    // group: compute it once per pool (qp[] = q*scale, corr = sum q*zp), then each key costs int4 loads + a MAD.
    float qp[NQP];                                           // q*scale per lane channel
    float corr = 0.0f;
    int   cur_pool = -1;

    // this block owns the KV slice [kv_lo, kv_hi); kv_split==1 -> the whole cache.
    const int kv_slice = (n_kv + kv_split - 1) / kv_split;
    const int kv_lo    = kv_g * kv_slice;
    const int kv_hi    = (kv_lo + kv_slice < n_kv) ? (kv_lo + kv_slice) : n_kv;
    // rolling int4-K prefetch: load the next key's nibbles into kbn[] before consuming kb[], so each K load is
    // in flight behind the previous key's QK/softmax/V (decode is latency-bound).
    uint8_t kb[NQP], kbn[NQP];
    if (kv_lo + wid < kv_hi) {
        const uint8_t * kr0 = (const uint8_t *) Kp + (size_t)(kv_lo + wid) * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
        int cj = 0; for (int c = lane; c < head_dim; c += 32, ++cj) kb[cj] = kr0[c >> 1];
    }
    for (int t = kv_lo + wid; t < kv_hi; t += n_split) {    // split-K: warp wid owns keys {kv_lo+wid, +n_split, ...}
        const int tn = t + n_split;                          // prefetch the next key's int4 bytes
        if (tn < kv_hi) {
            const uint8_t * krn = (const uint8_t *) Kp + (size_t) tn * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
            int cj = 0; for (int c = lane; c < head_dim; c += 32, ++cj) kbn[cj] = krn[c >> 1];
        }
        const float mv = mp ? __half2float(mp[t]) : 0.0f;
        if (mp && mv == -INFINITY) { for (int cj = 0; cj < NQP; ++cj) kb[cj] = kbn[cj]; continue; }   // masked, keep prefetch aligned
        int pool = gid ? gid[(size_t) ik3 * gi_stride + t] : (t / GGML_KPC_GROUP);   // cell->pool (positional if no map)
        if (pool < 0) pool = 0;
        if (pool != cur_pool) {                              // new group -> refold Q against its scale/zp
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
            corr = lc;                                       // sum_c q[c]*zp[c], same for every key in the group
            cur_pool = pool;
        }
        float part = 0.0f; int ci = 0;                       // lane's slice of sum_c qp[c]*nibble[c]
        for (int c = lane; c < head_dim; c += 32, ++ci)
            part += qp[ci] * (float) ((c & 1) ? (kb[ci] >> 4) : (kb[ci] & 0x0F));
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
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
        for (int cj = 0; cj < NQP; ++cj) kb[cj] = kbn[cj];   // advance the rolling prefetch buffer
    }
    // attention sink: fold the per-head sink logit into the denominator once (no V contribution). Under
    // grid-split only the kv_g==0 block folds it, so it counts once and the combine kernel stays sink-agnostic.
    if (sinks && wid == 0 && (kv_split == 1 || kv_g == 0)) {
        const float sk = sinks[iq2];
        const float mn = fmaxf(m, sk), a = expf(m - mn), p = expf(sk - mn);
        l = l * a + p;
        for (int j = 0; j < ndv; ++j) acc[j] *= a;
        m = mn;
    }

    // grid-split (kv_split>1): emit an un-normalised numerator + (m,l) meta at part_idx for the combine kernel.
    // kv_split==1: write dst directly.
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

// GQA query-group-fused decode: one warp owns a KV head + a tile of QT query heads that SHARE that KV head's
// K, V, and per-channel scale slab. Per key it loads+unpacks the int4 K row ONCE, folds the per-channel scale
// ONCE per group, and loads V ONCE -- all amortised across the QT queries (the dominant memory cost AND kpc's
// per-channel-scale cost, divided by QT). Each query keeps its own qp/corr/m/l/acc. Grid x =
// n_kvh*ceil(rk2/QT); 1 warp/block; grid split-K via blockIdx.y; the existing combine renormalises per query
// head (partials emitted at the per-head part_idx). Positional pools; GQA only (rk2>=2, rk2==rv2).
template<int DK_T, int DV_T, int QT>
static __global__ void KPC_DEC_LAUNCH_BOUNDS kpc_flash_decode_gqa(
        const float * Q, const uint8_t * Kp, const uint8_t * sz0, const uint8_t * V, const half * mask,
        const float * sinks, const int32_t * gid, int gi_stride, float kqscale, float max_bias, float logit_softcap, float * dst,
        int head_dim_rt, int DV_rt, int C_full, int n_kv, int n_head, int n_q, int n_kvh, int n_vh, int n_stream,
        int rk2, int rv2, int rk3, int rv3, bool v_q41,
        int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
        int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
        int64_t sz_nb1, int64_t sz_nb2,
        int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
        int64_t m_nb1, int64_t m_nb2, int64_t m_nb3, int m_ne2, int m_ne3,
        int64_t dst_nb1, int dst_ne1, int dst_ne2, int n_split_unused, float * parts, float2 * meta_out, int kv_split) {
    (void) n_split_unused;
    constexpr int DK = DK_T, DV = DV_T, NQP = DK_T / 32, NACC = (DV_T + 31) / 32;
    const int tiles = (rk2 + QT - 1) / QT;
    const int ikv   = blockIdx.x / tiles, qtile = blockIdx.x % tiles;   // KV head + query-tile within its group
    const int kv_g  = (kv_split > 1) ? blockIdx.y : 0;
    const int iq3   = blockIdx.z;
    if (ikv >= n_kvh || iq3 >= n_stream) return;
    const int lane  = threadIdx.x & 31;
    const int qt0   = qtile * QT;
    const int ik2 = ikv, iv2 = ikv, ik3 = iq3 / rk3, iv3 = iq3 / rv3;
    const int cbase = ik2 * DK;

    int           qh_id[QT];
    const float * qrow_i[QT];
    const half  * mp_i[QT];
    float slope_i[QT], corr[QT], m[QT], l[QT], qp[QT][NQP], acc[QT][NACC];
    #pragma unroll
    for (int i = 0; i < QT; ++i) {
        const int qh = ikv * rk2 + qt0 + i;
        const bool ok = (qt0 + i < rk2) && (qh < n_head);
        qh_id[i]  = ok ? qh : -1;
        qrow_i[i] = ok ? (const float *)((const char *) Q + (size_t) qh * q_nb2 + (size_t) iq3 * q_nb3) : nullptr;
        mp_i[i]   = (ok && mask) ? (const half *)((const char *) mask + (size_t)(qh % m_ne2) * m_nb2 + (size_t)(iq3 % m_ne3) * m_nb3) : nullptr;
        float sl = 1.0f;
        if (max_bias > 0.0f && ok) {
            const int nhl2 = kpc_pow2_floor(n_head);
            const float a0 = exp2f(-(max_bias) / nhl2), a1 = exp2f(-(max_bias / 2.0f) / nhl2);
            sl = (qh < nhl2) ? powf(a0, qh + 1) : powf(a1, 2 * (qh - nhl2) + 1);
        }
        slope_i[i] = sl; m[i] = -INFINITY; l[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < NACC; ++j) acc[i][j] = 0.0f;
    }
    int cur_pool = -1;

    const int kv_slice = (n_kv + kv_split - 1) / kv_split;
    const int kv_lo = kv_g * kv_slice, kv_hi = (kv_lo + kv_slice < n_kv) ? (kv_lo + kv_slice) : n_kv;
    uint8_t kb[NQP], kbn[NQP];
    if (kv_lo < kv_hi) {
        const uint8_t * kr0 = (const uint8_t *) Kp + (size_t) kv_lo * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
        int cj = 0; for (int c = lane; c < DK; c += 32, ++cj) kb[cj] = kr0[c >> 1];
    }
    for (int t = kv_lo; t < kv_hi; ++t) {
        if (t + 1 < kv_hi) {
            const uint8_t * krn = (const uint8_t *) Kp + (size_t)(t + 1) * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
            int cj = 0; for (int c = lane; c < DK; c += 32, ++cj) kbn[cj] = krn[c >> 1];
        }
        int pool = gid ? gid[(size_t) ik3 * gi_stride + t] : (t / GGML_KPC_GROUP);
        if (pool < 0) pool = 0;
        if (pool != cur_pool) {                              // refold scale into all QT queries (shared slab)
            const uint8_t * s = sz0 + (size_t) ik3 * sz_nb2 + (size_t) pool * sz_nb1;
            float ss, zmn, szc; kpc_slab_super(s, &ss, &zmn, &szc);
            const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C_full;
            float lc[QT];
            #pragma unroll
            for (int i = 0; i < QT; ++i) lc[i] = 0.0f;
            int ci = 0;
            for (int c = lane; c < DK; c += 32, ++ci) {
                const int fc = cbase + c; const float sc = qs[fc] * ss, zc = zmn + qz[fc] * szc;
                #pragma unroll
                for (int i = 0; i < QT; ++i) { if (qh_id[i] < 0) continue; const float q = qrow_i[i][c]; qp[i][ci] = q * sc; lc[i] += q * zc; }
            }
            #pragma unroll
            for (int i = 0; i < QT; ++i) { for (int o = 16; o > 0; o >>= 1) lc[i] += __shfl_xor_sync(0xffffffff, lc[i], o); corr[i] = lc[i]; }
            cur_pool = pool;
        }
        float part[QT];
        #pragma unroll
        for (int i = 0; i < QT; ++i) part[i] = 0.0f;
        int ci = 0;
        for (int c = lane; c < DK; c += 32, ++ci) {          // unpack nibble once, feed QT dots
            const float nib = (float) ((c & 1) ? (kb[ci] >> 4) : (kb[ci] & 0x0F));
            #pragma unroll
            for (int i = 0; i < QT; ++i) part[i] += qp[i][ci] * nib;
        }
        #pragma unroll
        for (int i = 0; i < QT; ++i) { for (int o = 16; o > 0; o >>= 1) part[i] += __shfl_xor_sync(0xffffffff, part[i], o); }
        float p_i[QT], a_i[QT];
        #pragma unroll
        for (int i = 0; i < QT; ++i) {
            if (qh_id[i] < 0) { p_i[i] = 0.0f; a_i[i] = 1.0f; continue; }
            const float mv = mp_i[i] ? __half2float(mp_i[i][t]) : 0.0f;
            if (mp_i[i] && mv == -INFINITY) { p_i[i] = 0.0f; a_i[i] = 1.0f; continue; }
            float score = (part[i] + corr[i]) * kqscale;
            if (logit_softcap != 0.0f) score = logit_softcap * tanhf(score / logit_softcap);
            score += slope_i[i] * mv;
            const float mn = fmaxf(m[i], score); a_i[i] = expf(m[i] - mn); p_i[i] = expf(score - mn);
            l[i] = l[i] * a_i[i] + p_i[i]; m[i] = mn;
        }
        const uint8_t * vrow = (const uint8_t *) V + (size_t) t * v_nb1 + (size_t) iv2 * v_nb2 + (size_t) iv3 * v_nb3;
        for (int j = 0; j < NACC; ++j) {                     // load each V element once, apply to QT queries
            const int d = lane + j * 32;
            if (d >= DV) break;
            float vv;
            if (v_q41) {
                const uint8_t * blk = vrow + (d / 32) * 20;
                const float vd = __half2float(*(const half *)(blk + 0)), vm = __half2float(*(const half *)(blk + 2));
                const int dj = d % 32; const uint8_t b = blk[4 + (dj % 16)];
                vv = ((dj < 16) ? (b & 0x0F) : (b >> 4)) * vd + vm;
            } else {
                vv = __half2float(((const half *) vrow)[d]);
            }
            #pragma unroll
            for (int i = 0; i < QT; ++i) acc[i][j] = acc[i][j] * a_i[i] + p_i[i] * vv;
        }
        for (int cj = 0; cj < NQP; ++cj) kb[cj] = kbn[cj];
    }
    if (sinks && (kv_split == 1 || kv_g == 0)) {
        #pragma unroll
        for (int i = 0; i < QT; ++i) {
            if (qh_id[i] < 0) continue;
            const float sk = sinks[qh_id[i]], mn = fmaxf(m[i], sk), a = expf(m[i] - mn), p = expf(sk - mn);
            l[i] = l[i] * a + p;
            for (int j = 0; j < NACC; ++j) acc[i][j] *= a;
            m[i] = mn;
        }
    }
    #pragma unroll
    for (int i = 0; i < QT; ++i) {
        if (qh_id[i] < 0) continue;
        const int iq2 = qh_id[i];
        const size_t part_idx = (size_t)(iq3 * n_head + iq2) * kv_split + kv_g;
        float * out = (float *)((char *) dst + (size_t)((size_t) iq3 * dst_ne2 * dst_ne1 + iq2) * dst_nb1);
        if (kv_split == 1) {
            const float Sinv = (l[i] == 0.0f) ? 0.0f : 1.0f / l[i];
            for (int j = 0; j < NACC; ++j) { const int d = lane + j * 32; if (d < DV) out[d] = acc[i][j] * Sinv; }
        } else {
            float * pb = parts + part_idx * DV;
            for (int j = 0; j < NACC; ++j) { const int d = lane + j * 32; if (d < DV) pb[d] = acc[i][j]; }
            if (lane == 0) meta_out[part_idx] = make_float2(m[i], l[i]);
        }
    }
}

// grid-split combine: renormalise the kv_split per-block partials into dst with an online-softmax rescale.
// One block per (head, stream); DV threads. Mirrors flash_attn_combine_results.
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

// query-tiled flash-decode read: 1 warp per (head, query-tile of QT, stream). Each key's K nibbles + V row
// are loaded once and reused for all QT queries in the tile. Single warp only (no split-K) -> prefill path.
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
        if (pool != cur_pool) {                          // new group -> refold every query's Q against its scale/zp
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
        // load this key's K nibbles for the lane's channels once, reuse across the tile
        const uint8_t * krow = (const uint8_t *) Kp + (size_t) t * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
        float nbv[NQP]; { int ci = 0; for (int c = lane; c < head_dim; c += 32, ++ci) { const uint8_t nb = krow[c >> 1]; nbv[ci] = (float) ((c & 1) ? (nb >> 4) : (nb & 0x0F)); } }
        // load this key's V row for the lane's elements once
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

// Tensor-core prefill path (sm_75+): dequant K (int4 per-chan) + V (q4_1) into f16 smem tiles, online
// softmax over KN-key tiles, QK and PV via ggml_cuda_mma. One warp per (head, QM-query tile, stream).
#define KPC_QM 16          // queries per block tile (MMA M dim)
#define KPC_KN 16          // keys per inner tile
#define KPC_NW 8           // warps per block: dequant on all NW warps; QK/PV MMA + softmax on warp 0


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


    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const int n_head_log2 = kpc_pow2_floor(n_head);
        const float m0 = exp2f(-(max_bias)        / n_head_log2);
        const float m1 = exp2f(-(max_bias / 2.0f) / n_head_log2);
        slope = (iq2 < n_head_log2) ? powf(m0, iq2 + 1) : powf(m1, 2 * (iq2 - n_head_log2) + 1);
    }

    // smem: floats first so the f16 tiles that follow stay half2-aligned.
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

    // top query of this tile and its causal key horizon: keys past it are future for every query in the tile.
    const int qtop    = iq1base + nqv - 1;
    const int qmaxpos = (n_kv - n_q) + qtop;            // n_kv-n_q = n_past; query qtop attends to keys <= this
    for (int kt = 0; kt < n_kv; kt += KPC_KN) {
        const int knv = (n_kv - kt) < KPC_KN ? (n_kv - kt) : KPC_KN;
        // skip a key-tile past the top query's horizon (fully masked -> contributes 0). Depends only on kt so
        // barriers stay matched; the mask at (top query, kt) being -inf proves the whole tile is masked, and a
        // non-causal mask is finite there so we never skip.
        if (mask && kt > qmaxpos) {
            const half * mtop = (const half *)((const char *) mask + (size_t) qtop * m_nb1
                              + (size_t)(iq2 % m_ne2) * m_nb2 + (size_t)(iq3 % m_ne3) * m_nb3);
            if (__half2float(mtop[kt]) == -INFINITY) continue;
        }
        // dequant K tile -> Kf (per-channel: nibble*scale + zp; group scale/zp cached in smem).
        // Positional tiles (gid==null) are always single-pool (GROUP=32, KN=16, 16-aligned kt); gid tiles are too
        // when the cell->pool map is positional. Single-pool dequants over a flat (key,channel) grid-stride loop.
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
            // one byte holds both nibbles of a channel pair (c=2*cp low, 2*cp+1 high); read it once, dequant both
            // channels and store as one half2.
            for (int idx = tid; idx < knv * (HD / 2); idx += blockDim.x) {
                const int k = idx / (HD / 2), cp = idx % (HD / 2);
                const uint8_t * krow = (const uint8_t *) Kp + (size_t)(kt + k) * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
                const uint8_t nb = krow[cp];
                const float2 sc = ((const float2 *) scaZ)[cp];   // scaZ[2cp], scaZ[2cp+1]
                const float2 zp = ((const float2 *) zpcZ)[cp];
                const float v0 = (float) (nb & 0x0F) * sc.x + zp.x;   // channel 2cp   (low nibble)
                const float v1 = (float) (nb >>   4) * sc.y + zp.y;   // channel 2cp+1 (high nibble)
                ((half2 *) Kf)[k * (HD / 2) + cp] = __floats2half2_rn(v0, v1);
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
        __syncthreads();                                // K dequant (all warps) visible -> warp-0 QK MMA

        // warp 0: QK MMA + per-query softmax. warps 1-7: dequant V into Vf. V is not read until the PV MMA
        // below, so produce it on the otherwise-idle warps during the QK+softmax phase. (MMA is warp-collective;
        // lane q owns query q.)
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
                if (tilemax == -INFINITY) {
                    // fully-masked tile for this query: contributes nothing, so skip the softmax update. Mx[q] may
                    // still be -inf, so expf(Mx[q]-mn) would be NaN; leave the accumulator untouched.
                    for (int k = 0; k < KPC_KN; ++k) Sc[q * KPC_KN + k] = 0.0f;
                    Av[q] = 1.0f;   // no rescale of the accumulator
                } else {
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
        } else {
            // dequant V tile -> Vf, d-major [d][k] (q4_1 or f16); zero padded keys so P=0 * V can't make NaN.
            // warps 1-7, concurrent with warp 0's QK MMA + softmax above.
            for (int idx = tid - 32; idx < DV * KPC_KN; idx += blockDim.x - 32) {
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
        }
        __syncthreads();                                // Vf (warps 1-7) + softmax state Av,Sc,Mx,Ln (warp 0) -> all warps

        // PV: rescale Acc by this tile's softmax factor, convert P -> f16, then warp-0 P@V MMA
        for (int idx = tid; idx < nqv * DV; idx += blockDim.x) Acc[idx] *= Av[idx / DV];
        for (int idx = tid; idx < KPC_QM * KPC_KN; idx += blockDim.x) {
            const int q = idx / KPC_KN;
            Pf[idx] = (q < nqv) ? __float2half(Sc[idx]) : __float2half(0.0f);
        }
        __syncthreads();
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
    }

    // attention sink (fold once per query) + write normalized output
    if (tid < nqv) {
        const int q = tid;
        if (sinks) {
            const float sk = sinks[iq2];
            const float mn = fmaxf(Mx[q], sk), a = expf(Mx[q] - mn), p = expf(sk - mn);
            Ln[q] = Ln[q] * a + p;
            Av[q] = a;   // carries the sink rescale for Acc below
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
}

// ---- host wrappers (thin launchers) ----

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

    // Single-token decode (n_q==1) kernel choice. The dequant-once -> stock VEC helper must rewrite the whole int4
    // cache to an f16 scratch every step (materialise ~ n_kv*n_kvh*head_dim); the custom kernel reads in place but
    // needs the key range fanned across many grid blocks (kv_split, capped by n_kv/32/n_split) to fill the SMs.
    // So custom wins only when ALL of: head_dim<=128 (custom cheap per key), n_kvh>=8 (f16 materialise expensive),
    // and n_kv large enough to feed the grid split-K. Below ~2k context the custom kernel is occupancy-starved and
    // f16 wins; above it custom pulls ahead and grows with depth. Measured RTX 5060: custom beats f16 for
    // deepseek-coder (hd128,n_kvh16) and llama-3.2 (hd64,n_kvh8) at n_kv>=2048 (deepseek d4096 +65%, d2048 +36%;
    // f16 wins both below d2048). f16 still wins few-kv-head (qwen2/glm4 n_kvh=2). hd256 with LIGHT GQA (rk2<=2,
    // e.g. gemma2 8h/4kvh) now ALSO routes to custom at n_kv>=2048: the DV-aware __launch_bounds__ (above,
    // min_blocks=1 for DV>=256) removed a 120-byte register spill that had crippled the hd256 custom kernel, so it
    // beats the f16 materialise (gemma2 d4096 75.5->84.1 = +11%, 0.86->0.96x). Heavy-GQA hd256 (qwen3.5 rk2=4) stays
    // f16: the custom read re-reads the shared KV per query head, so high rk2 makes it a wash/-1%. f16 wins < d2048.
    // Measured on Ada/Blackwell, but the gate is the helper's FUNCTIONALITY floor, not the test card: the stock
    // flash_attn_ext_vec has a non-tensor-core path, so the f16 route runs on any cc>=TURING. Turing/Ampere get it by
    // default (validate). Pascal (cc<TURING) routes to custom/GQA above.
    const bool prefer_custom_dec = (head_dim <= 128 && n_kvh >= 8 && n_kv >= 2048)
                                || (head_dim == 256 && rk2 <= 2 && n_kv >= 2048);
    if (!prefer_custom_dec && n_q == 1 && n_stream == 1 && head_dim == DV &&
        (head_dim == 64 || head_dim == 128 || head_dim == 256) && n_kv >= 256 &&
        ggml_cuda_info().devices[ggml_cuda_get_device()].cc >= GGML_CUDA_CC_TURING &&
        kpc_flash_prefill_stock(ctx, dst, Q, Kp, sz, V, mask, gi, sinks, params,
                                head_dim, DV, C_full, n_kv, n_kvh, n_vh, n_stream, v_q41)) {
        return;
    }

    const int gi_stride = gi ? (int) (gi->nb[1] / sizeof(int32_t)) : 0;   // per-stream stride (= kv_size)
    // split-K: with few query blocks (decode), fan the key loop across warps within the block; the per-warp
    // partials are combined in shared memory. Grow n_split while it fits the thread/smem limits and does not
    // oversubscribe the grid. Prefill has many (head x query) blocks so the grid test keeps it at n_split==1.
    const long blocks = (long) n_head * n_q * n_stream;
    int n_split = 1;
    while (32 * (n_split * 2) <= 1024 &&                                                    // threads/block limit
           (size_t)((n_split * 2) * DV + 2 * (n_split * 2)) * sizeof(float) <= 48 * 1024 && // dynamic-smem budget
           blocks * (n_split * 2) <= 2048 &&                                                // don't oversubscribe
           (n_split * 2) <= n_kv) n_split *= 2;
    // register-file cap: shrink n_split so 32*n_split fits the instantiation's maxThreadsPerBlock.
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
    // split-K's per-warp partition + combine is not bit-reproducible across context instances, so with >1
    // sequence in the cache decode single-warp + single-block; n_seq_max==1 keeps both split-K paths.
    const int  n_seq_max      = (int) params[3];
    const bool kpc_det_decode = n_seq_max > 1;
    if (kpc_det_decode) n_split = 1;
    // Occupancy: decode wants the grid (n_head*kv_split blocks) to fill all SMs several deep. Maximising warps/block
    // (n_split up to 16) makes few large blocks -> only ~2 resident/SM -> ~60% occupancy. Capping n_split small and
    // fanning the key range across MORE grid blocks (kv_split, below) saturates the SMs: ~10% faster on Blackwell
    // (deepseek hd128 d4096 95->104) and big on Pascal P40 (deepseek 51->70; small-head GQA models beat the GQA-fused
    // kernel this way -- qwen2 +9%, llama-1b +14%). This is a UNIVERSAL latency-hiding split -- no special hardware --
    // so cap whenever n_split>4 on every arch; only the kv_split VALUE below is arch-tuned. (Measured Blackwell + all
    // Pascal; Turing/Ampere/Ada/Hopper extrapolated -- validate.)
    const int dec_cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!kpc_det_decode && n_split > 4) n_split = 4;
    const int    nthreads = 32 * n_split;
    const size_t smem = (n_split > 1) ? ((size_t) n_split * DV + 2 * n_split) * sizeof(float) : 0;
    dim3 grid(n_head, n_q, n_stream);
    // specialize on (head_dim, DV) so qp[]/acc[] are compile-time sized; <0,0> = runtime.
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

    // grid-level split-K for decode: decode has only n_head*n_stream blocks, so fan each (head,stream)'s key
    // range across kv_split blocks to use all SMs, then renormalise the partials with the combine kernel.
    // Partials live in a transient pool buffer (same idiom as f16 FA's dst_tmp). Off under multi-seq.
    int kv_split = 1;
    if (!tiled && n_q == 1 && n_kv >= 2048 && !kpc_det_decode && head_dim == DV &&
        (head_dim == 64 || head_dim == 128 || head_dim == 256)) {
        const int  id  = ggml_cuda_get_device();
        const int  nsm = ggml_cuda_info().devices[id].nsm;
        const long base_blocks = (long) n_head * n_stream;
        // Target blocks/SM for the grid split-K (fill every SM several deep). Capable arches without a measured
        // value default to the Blackwell-class 9 pending a per-card sweep.
        const int blk_per_sm =
            (dec_cc >= GGML_CUDA_CC_BLACKWELL) ? 9  :   // measured (sm_120)
            (dec_cc <  GGML_CUDA_CC_TURING)    ? 16 :   // measured (P40 sweep), all Pascal/Volta
            9;                                          // Turing/Ampere/Ada/Hopper: untested default (validate per card)
        int want = (int) (((long) blk_per_sm * nsm + base_blocks - 1) / base_blocks);
        const int max_by_keys = n_kv / (32 * n_split);                   // keep >= 32 keys per warp-split per block
        if (want > max_by_keys) want = max_by_keys;
        if (want > 1) kv_split = want;
    }

    // GQA query-group-fused decode: grid over KV-heads x query-tiles, 1 warp/block, amortises K/scale/V loads across
    // the QT queries sharing a KV head. A WASH on sm_120 (compute-bound). On Pascal it beats the per-head default, but
    // the standard kernel + grid split-K (above) beats it further for SMALL heads (the 1-warp/block GQA kernel is
    // occupancy-starved: hd64 qwen2 +9%, llama-1b +14% on the split path vs GQA); for hd>=128 GQA (glm4) the two tie.
    // This is a default, not a hardware gate. Default ON for the lower-bandwidth pre-Ada arches (memory-bound,
    // win-or-wash) but only head_dim>=128 (small-head loses, above); OFF for compute-bound Ada/Hopper/Blackwell
    // (sm_120 wash). Turing/Ampere extrapolated -- validate.
    const int kpc_gqa = (dec_cc < GGML_CUDA_CC_ADA_LOVELACE && head_dim >= 128) ? 1 : 0;
    constexpr int GQA_QT = 4;
    if (kpc_gqa && n_q == 1 && n_stream == 1 && !kpc_det_decode && head_dim == DV &&
        (head_dim == 64 || head_dim == 128 || head_dim == 256) && rk2 >= 2 && rk2 == rv2 && n_kv >= 2048) {
        const int  tiles = (rk2 + GQA_QT - 1) / GQA_QT;
        const int  nsm   = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        const long base_g = (long) n_kvh * tiles * n_stream;
        int kvs = (int)((9L * nsm + base_g - 1) / base_g);
        const int max_by_keys = n_kv / 32;                              // 1 warp/block -> >=32 keys/block
        if (kvs > max_by_keys) kvs = max_by_keys;
        if (kvs < 1) kvs = 1;
        ggml_cuda_pool_alloc<float>  pbuf(ctx.pool());
        ggml_cuda_pool_alloc<float2> mbuf(ctx.pool());
        float * pg = nullptr; float2 * mg = nullptr;
        if (kvs > 1) { const size_t np = (size_t) kvs * n_head * n_stream; pg = pbuf.alloc(np * DV); mg = mbuf.alloc(np); }
        const dim3 gridG(n_kvh * tiles, kvs, n_stream);
        const int gi_stride2 = gi ? (int) (gi->nb[1] / sizeof(int32_t)) : 0;
        #define KPC_GQA_ARGS \
            (const float *) Q->data, (const uint8_t *) Kp->data, (const uint8_t *) sz->data, \
            (const uint8_t *) V->data, mask ? (const half *) mask->data : nullptr, \
            sinks ? (const float *) sinks->data : nullptr, gi ? (const int32_t *) gi->data : nullptr, gi_stride2, \
            params[0], params[1], params[2], (float *) dst->data, \
            head_dim, DV, C_full, n_kv, n_head, n_q, n_kvh, n_vh, n_stream, rk2, rv2, rk3, rv3, v_q41, \
            Q->nb[1], Q->nb[2], Q->nb[3], Kp->nb[1], Kp->nb[2], Kp->nb[3], sz->nb[1], sz->nb[2], \
            V->nb[1], V->nb[2], V->nb[3], \
            mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0, \
            mask ? (int) mask->ne[2] : 1, mask ? (int) mask->ne[3] : 1, \
            dst->nb[1], (int) dst->ne[1], (int) dst->ne[2], 1, pg, mg, kvs
        if      (head_dim ==  64) kpc_flash_decode_gqa< 64, 64, GQA_QT><<<gridG, 32, 0, ctx.stream()>>>(KPC_GQA_ARGS);
        else if (head_dim == 128) kpc_flash_decode_gqa<128,128, GQA_QT><<<gridG, 32, 0, ctx.stream()>>>(KPC_GQA_ARGS);
        else                      kpc_flash_decode_gqa<256,256, GQA_QT><<<gridG, 32, 0, ctx.stream()>>>(KPC_GQA_ARGS);
        #undef KPC_GQA_ARGS
        if (kvs > 1) {
            const int    cthreads = DV < 1024 ? DV : 1024;
            const size_t csmem    = (size_t) kvs * sizeof(float2);
            dim3 gridC(n_head, n_stream);
            if      (DV ==  64) kpc_flash_combine_kernel< 64><<<gridC, cthreads, csmem, ctx.stream()>>>(pg, mg, (float *) dst->data, kvs, n_head, DV, dst->nb[1], (int) dst->ne[1], (int) dst->ne[2]);
            else if (DV == 128) kpc_flash_combine_kernel<128><<<gridC, cthreads, csmem, ctx.stream()>>>(pg, mg, (float *) dst->data, kvs, n_head, DV, dst->nb[1], (int) dst->ne[1], (int) dst->ne[2]);
            else                kpc_flash_combine_kernel<256><<<gridC, cthreads, csmem, ctx.stream()>>>(pg, mg, (float *) dst->data, kvs, n_head, DV, dst->nb[1], (int) dst->ne[1], (int) dst->ne[2]);
        }
        return;
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

    // tensor-core prefill path. cc>=Turing is a correctness floor: the MMA intrinsics compile to NO_DEVICE_CODE
    // on older arches, which fall back to the non-MMA path below (dequant-once stock FA on Pascal, else the
    // query-tiled scalar kernel). MMA amortises the f16 staging once there are enough keys (>=256).
    bool use_mma = false;
    if (tiled) {
        const int  dev_cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const bool cc_mma = dev_cc >= GGML_CUDA_CC_TURING;
        use_mma = cc_mma && n_kv >= 256;
    }
    if (use_mma) {
        // dequant-once -> stock flash-attn: re-dequanting int4 K per query-tile is O(n_q*n_kv) and dominates at
        // long context, so dequant K (+ q4_1 V) to an f16 scratch once (O(n_kv)) and ride the stock kernel. The
        // persistent KV cache stays int4; the f16 scratch is one-op-sized pool memory. Gated to n_stream==1: the
        // dequant does not apply the custom kernel's stream-broadcast ratios (rk3/rv3), so multi-stream prefill
        // keeps the custom kernel.
        const bool done = n_stream == 1 &&
            kpc_flash_prefill_stock(ctx, dst, Q, Kp, sz, V, mask, gi, sinks, params,
                                    head_dim, DV, C_full, n_kv, n_kvh, n_vh, n_stream, v_q41);
        if (!done) {
            dim3 gridM(n_head, (n_q + KPC_QM - 1) / KPC_QM, n_stream);
            const size_t smem_m = (size_t)(KPC_QM * head_dim + KPC_KN * head_dim + DV * KPC_KN + KPC_QM * KPC_KN) * sizeof(half)
                                + (size_t)(KPC_QM * DV + KPC_QM * KPC_KN + 2 * head_dim + 3 * KPC_QM) * sizeof(float);
            const int nthr_m = 32 * KPC_NW;
            if      (head_dim ==  64) kpc_flash_prefill_kernel< 64, 64><<<gridM, nthr_m, smem_m, ctx.stream()>>>(KPC_FA_ARGS);
            else if (head_dim == 128) kpc_flash_prefill_kernel<128,128><<<gridM, nthr_m, smem_m, ctx.stream()>>>(KPC_FA_ARGS);
            else                      kpc_flash_prefill_kernel<256,256><<<gridM, nthr_m, smem_m, ctx.stream()>>>(KPC_FA_ARGS);
        }
    }
    else if (tiled) {
        // Pascal (no MMA): dequant-once -> stock f16 flash-attn (the tile path) is far faster than the query-tiled
        // scalar kernel here -- same idea as the MMA path above, but the stock kernel has a non-tensor-core path so
        // it needs no Turing. The helper returns false if the stock FA cannot take the config on this device, then
        // we fall back to the scalar kernel. Gated cc<Turing so the Turing+ use_mma routing is untouched;
        // n_stream==1 (the dequant omits the custom kernel's stream-broadcast ratios).
        const bool done = dec_cc < GGML_CUDA_CC_TURING && n_stream == 1 &&
            kpc_flash_prefill_stock(ctx, dst, Q, Kp, sz, V, mask, gi, sinks, params,
                                    head_dim, DV, C_full, n_kv, n_kvh, n_vh, n_stream, v_q41);
        if (!done) {
            const int QT = 3;
            dim3 gridT(n_head, (n_q + QT - 1) / QT, n_stream);
            if      (head_dim ==  64) kpc_flash_decode_qt_kernel< 64, 64, 3><<<gridT, 32, 0, ctx.stream()>>>(KPC_FA_ARGS);
            else if (head_dim == 128) kpc_flash_decode_qt_kernel<128,128, 3><<<gridT, 32, 0, ctx.stream()>>>(KPC_FA_ARGS);
            else                      kpc_flash_decode_qt_kernel<256,256, 3><<<gridT, 32, 0, ctx.stream()>>>(KPC_FA_ARGS);
        }
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

// scan this (seq,g) block's new members (k_cur tokens with pos/32==g and this seq) into shared
// newmask/tok_of/cell_of, set has_later if a later group exists (g is then not the seq's open group).
static __device__ __forceinline__ void kpc_scan_members(
        const int32_t * kpc_seq, const int32_t * kpc_pos, const int64_t * k_idxs,
        int g, int seq, int nt,
        uint32_t * s_newmask, int * s_has_later, int * tok_of, int64_t * cell_of) {
    if (threadIdx.x == 0) { *s_newmask = 0; *s_has_later = 0; }
    if ((int) threadIdx.x < GGML_KPC_GROUP) { tok_of[threadIdx.x] = -1; cell_of[threadIdx.x] = -1; }
    __syncthreads();
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
        if (kpc_seq[i] != seq) continue;
        const int tg = kpc_pos[i] / GGML_KPC_GROUP;
        if (tg > g) { *s_has_later = 1; continue; }             // later token exists -> g is not the open group
        if (tg < g) continue;
        const int w = kpc_pos[i] % GGML_KPC_GROUP;
        tok_of[w]  = i;
        cell_of[w] = k_idxs[i];
        atomicOr(s_newmask, 1u << w);
    }
    __syncthreads();
}

// group-parallel seal. One block per (logical group g, sequence). Each block gathers the group's members
// once -- this ubatch's tokens (k_cur) plus any previously-staged members (k_resid, when g is the seq's open
// group) -- computes the per-channel scale/zp, encodes the scalezp slab and packs every member into its global
// K cell (k_idxs for new, resid_slots for staged). Multi-stream: stream = seq/n_seqps,
// pool = (seq%n_seqps)*band_size + g%band_size (mirrors the CPU pool_of).
// Only reads the staging tensors; the staging update is a separate launch (kpc_write_stage_kernel) so the seal
// never reads staging a concurrent write is mutating.
// DO_PACK=true (prefill): also packs every member into K in this same kernel (one members scan). DO_PACK=false
// (single-group decode): packs in a separate multi-block kpc_pack_kernel instead, to fan the pack's bandwidth
// across SMs. Gated because the separate pack re-scans members (O(nt)/block) -- cheap at nt=1, but doubles the
// prefill scan, so prefill stays fused.
template<bool DO_PACK>
static __global__ void kpc_write_kernel(
        const float * k_cur, const int32_t * kpc_seq, const int32_t * kpc_pos, const int64_t * k_idxs,
        const half * k_resid, uint8_t * scalezp, const int32_t * resid_slots,
        const int32_t * staged_group, const int32_t * staged_mask,
        uint8_t * K, int C, int nt,
        int n_seqps, int band_size, int64_t sz_nb1, int64_t sz_nb2, int64_t k_nb1,
        int64_t kc_nb0, int64_t kc_nb1) {
    // k_cur may be a non-contiguous view (ALiBi models have no RoPE op to repack K) -> read it via strides.
    #define KPC_KC(tk, ch) (*(const float *)((const char *) k_cur + (size_t)(tk)*kc_nb1 + (size_t)(ch)*kc_nb0))
    const int g   = blockIdx.x;     // logical 32-token group
    const int seq = blockIdx.y;     // sequence (one staging buffer per seq)
    const int krow = C / 2;

    extern __shared__ float smem[];  // scale[C], zp[C]
    float * scale = smem; float * zp = smem + C;
    __shared__ uint32_t newmask;
    __shared__ int      has_later;
    __shared__ int      tok_of[GGML_KPC_GROUP];    // k_cur token index that wrote within-group slot w (-1 none)
    __shared__ int64_t  cell_of[GGML_KPC_GROUP];   // global K cell for a new member w
    kpc_scan_members(kpc_seq, kpc_pos, k_idxs, g, seq, nt, &newmask, &has_later, tok_of, cell_of);

    const uint32_t omask   = (g == staged_group[(1 + GGML_KPC_GROUP) * seq]) ? (uint32_t) staged_mask[seq] : 0u;   // staged members of g
    const uint32_t members = newmask | omask;
    if (members == 0) return;

    const int st   = seq / n_seqps;                             // physical stream
    const int pool = (seq % n_seqps) * band_size + (g % band_size);
    const half * rsd = k_resid + (size_t) seq * C * GGML_KPC_GROUP;
    uint8_t * s = scalezp + (size_t) st * sz_nb2 + (size_t) pool * sz_nb1;

    // survivor-rescue: committed cells still in this pool but not in this write (staged_group rows 1.., -1-terminated)
    // must be folded into the re-encode and repacked, else they decode against a foreign scale (silent corruption on
    // mid-group seq_rm + continue). Preserve the OLD slab (in smem after scale[C]/zp[C]) to dequant them vs it.
    const int32_t * surv     = staged_group + (size_t)(1 + GGML_KPC_GROUP) * seq + 1;
    const bool      has_surv = surv[0] >= 0;
    uint8_t * old_slab = (uint8_t *)(zp + C);
    if (has_surv) {
        for (int b = threadIdx.x; b < 2 * C + 6; b += blockDim.x) old_slab[b] = s[b];
    }
    __syncthreads();
    float oss = 0.0f, ozmn = 0.0f, osz = 0.0f;
    if (has_surv) kpc_slab_super(old_slab, &oss, &ozmn, &osz);

    // the host fill is a plain gid==pool scan with no member exclusion, so the survivor list can include cells that
    // are members of this write -- flag those once so the fold and repack skip them (members come from k_cur/k_resid).
    // member cell = cell_of[w] (new) or resid_slots[..] (staged), the same mapping the pack uses.
    __shared__ bool surv_skip[GGML_KPC_GROUP];
    if (has_surv) {
        for (int i = threadIdx.x; i < GGML_KPC_GROUP; i += blockDim.x) {
            bool ism = false;
            if (surv[i] >= 0) {
                for (int w = 0; w < GGML_KPC_GROUP; ++w) {
                    if (!(members & (1u << w))) continue;
                    const int64_t mc = (newmask & (1u << w)) ? cell_of[w]
                                                             : (int64_t) resid_slots[seq * GGML_KPC_GROUP + w];
                    if ((int64_t) surv[i] == mc) { ism = true; break; }
                }
            }
            // skip cross-group cells: under -kvu the banded pool is shared across groups (pool = g % band_size), so a
            // gid==pool scan returns other groups' committed cells. They were correct pre-rescue (shared-slab,
            // last-writer-wins); folding/repacking them here races across the concurrent per-group write blocks. Only
            // same-group survivors (the seq_rm re-encode) need the rescue. Members are at g*KPC_GROUP+w, so same group
            // == cell/KPC_GROUP == g.
            surv_skip[i] = ism || (surv[i] >= 0 && (uint32_t)(surv[i] / GGML_KPC_GROUP) != (uint32_t) g);
        }
        __syncthreads();
    }

    // per-channel min/max over members (new from k_cur f32, staged-only from k_resid f16). Single block: this is
    // bounded by the all-C super-quant reduction below, which needs every channel's scale/zp in shared memory.
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float mn = INFINITY, mx = -INFINITY;
        for (int w = 0; w < GGML_KPC_GROUP; ++w) {
            if (!(members & (1u << w))) continue;
            float v = (newmask & (1u << w)) ? KPC_KC(tok_of[w], c)
                                            : __half2float(rsd[(size_t) w * C + c]);
            mn = fminf(mn, v); mx = fmaxf(mx, v);
        }
        if (has_surv) {                                         // fold committed survivors (dequant vs the OLD slab)
            const float osc = old_slab[6 + c] * oss, ozp = ozmn + old_slab[6 + C + c] * osz;
            for (int i = 0; i < GGML_KPC_GROUP && surv[i] >= 0; ++i) {   // bound to the 32-slot row (may be unterminated when full)
                if (surv_skip[i]) continue;
                const uint8_t * row = K + (size_t) surv[i] * k_nb1;
                const float v = kpc_deq_nib(row[c >> 1], c, osc, ozp);
                mn = fminf(mn, v); mx = fmaxf(mx, v);
            }
        }
        float sc = (mx - mn) / 15.0f; if (sc == 0.0f) sc = 1.0f;
        scale[c] = sc; zp[c] = mn;
    }
    __syncthreads();
    // super-quantize the per-channel scale[]/zp[] into the slab. Parallel block reduction (smax over scale,
    // min/max over zp) + parallel per-channel quant -- this was a thread-0 serial O(C) loop, the dominant decode
    // KV-write cost at C=n_kvh*head_dim (2048 for deepseek). Math is identical (min/max are order-independent).
    {
        float lmax = 0.0f, lzmn = INFINITY, lzmx = -INFINITY;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            lmax = fmaxf(lmax, scale[c]); lzmn = fminf(lzmn, zp[c]); lzmx = fmaxf(lzmx, zp[c]);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, o));
            lzmn = fminf(lzmn, __shfl_xor_sync(0xffffffff, lzmn, o));
            lzmx = fmaxf(lzmx, __shfl_xor_sync(0xffffffff, lzmx, o));
        }
        __shared__ float r_max[32], r_zn[32], r_zx[32];
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
        if (lane == 0) { r_max[warp] = lmax; r_zn[warp] = lzmn; r_zx[warp] = lzmx; }
        __syncthreads();
        __shared__ float ss_s, zmn_s, sz_s;
        if (warp == 0) {
            float m  = (lane < nw) ? r_max[lane] : 0.0f;
            float zn = (lane < nw) ? r_zn[lane]  : INFINITY;
            float zx = (lane < nw) ? r_zx[lane]  : -INFINITY;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                m  = fmaxf(m,  __shfl_xor_sync(0xffffffff, m,  o));
                zn = fminf(zn, __shfl_xor_sync(0xffffffff, zn, o));
                zx = fmaxf(zx, __shfl_xor_sync(0xffffffff, zx, o));
            }
            if (lane == 0) {
                float ss = m / 255.0f; if (ss == 0.0f) ss = 1.0f;
                float sz = (zx - zn) / 255.0f; if (sz == 0.0f) sz = 1.0f;
                ss_s = ss; zmn_s = zn; sz_s = sz;
                *(half *)(s + 0) = __float2half(ss); *(half *)(s + 2) = __float2half(zn); *(half *)(s + 4) = __float2half(sz);
            }
        }
        __syncthreads();
        const float ss = ss_s, zmn0 = zmn_s, sz = sz_s;
        uint8_t * qs = s + 6; uint8_t * qz = s + 6 + C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            int a = (int)(scale[c]/ss + 0.5f); a = a<0?0:(a>255?255:a);
            int b = (int)((zp[c]-zmn0)/sz + 0.5f); b = b<0?0:(b>255?255:b);
            qs[c] = (uint8_t) a; qz[c] = (uint8_t) b;
        }
    }
    // repack the rescued survivors against the NEW slab (dequant vs the preserved OLD slab -> requant vs s). Always
    // runs (decode packs members in kpc_pack_kernel, but the survivor repack needs the OLD slab, which lives only in
    // this kernel). has_surv is uniform per block, so the barrier is safe.
    if (has_surv) {
        __syncthreads();
        float ss, zmn, sz; kpc_slab_super(s, &ss, &zmn, &sz);
        const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C;
        for (int i = 0; i < GGML_KPC_GROUP && surv[i] >= 0; ++i) {   // bound to the 32-slot row (may be unterminated when full)
            if (surv_skip[i]) continue;
            uint8_t * row = K + (size_t) surv[i] * k_nb1;
            for (int b = threadIdx.x; b < krow; b += blockDim.x) {
                const int c0 = 2 * b, c1 = 2 * b + 1;
                const float v0 = kpc_deq_nib(row[b], c0, old_slab[6 + c0] * oss, ozmn + old_slab[6 + C + c0] * osz);
                const float v1 = kpc_deq_nib(row[b], c1, old_slab[6 + c1] * oss, ozmn + old_slab[6 + C + c1] * osz);
                int q0 = (int)((v0 - (zmn + qz[c0]*sz)) / (qs[c0]*ss) + 0.5f);
                int q1 = (int)((v1 - (zmn + qz[c1]*sz)) / (qs[c1]*ss) + 0.5f);
                q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
                q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
                row[b] = (uint8_t)(q0 | (q1 << 4));
            }
        }
    }
    // pack: prefill packs here (fused, one scan); decode skips this and uses the multi-block kpc_pack_kernel.
    if (DO_PACK) {
        __syncthreads();
        float ss, zmn, sz; kpc_slab_super(s, &ss, &zmn, &sz);
        const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C;
        for (int w = 0; w < GGML_KPC_GROUP; ++w) {
            if (!(members & (1u << w))) continue;
            const bool    isnew = newmask & (1u << w);
            const int64_t cell  = isnew ? cell_of[w] : (int64_t) resid_slots[seq * GGML_KPC_GROUP + w];
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
    // (decode: pack runs in kpc_pack_kernel, fanned multi-block across SMs since this kernel is 1 active block on
    // 1 SM for single-group decode -- ~5% SM-active per Nsight gpu-metrics. A further min/max split via a global
    // scratch helped decode +0.6% but cost prefill ~16% on the scratch round-trip, so min/max stays fused here.)
}

// pack pass (separate launch after kpc_write_kernel wrote the slab): reads the slab + members and writes the int4
// K cells. blockIdx.z splits the krow bytes across blocks so single-group decode (1 active group) spreads its
// member-read/K-write bandwidth across many SMs instead of one. Same packing math as the old in-kernel loop.
static __global__ void kpc_pack_kernel(
        const float * k_cur, const int32_t * kpc_seq, const int32_t * kpc_pos, const int64_t * k_idxs,
        const half * k_resid, const uint8_t * scalezp, const int32_t * resid_slots,
        const int32_t * staged_group, const int32_t * staged_mask,
        uint8_t * K, int C, int nt,
        int n_seqps, int band_size, int64_t sz_nb1, int64_t sz_nb2, int64_t k_nb1,
        int64_t kc_nb0, int64_t kc_nb1) {
    // KPC_KC is the file-scope macro defined in kpc_write_kernel (still in scope until after kpc_write_stage_kernel).
    const int g   = blockIdx.x;
    const int seq = blockIdx.y;
    const int krow = C / 2;
    __shared__ uint32_t newmask; __shared__ int has_later;
    __shared__ int      tok_of[GGML_KPC_GROUP];
    __shared__ int64_t  cell_of[GGML_KPC_GROUP];
    kpc_scan_members(kpc_seq, kpc_pos, k_idxs, g, seq, nt, &newmask, &has_later, tok_of, cell_of);
    const uint32_t omask   = (g == staged_group[(1 + GGML_KPC_GROUP) * seq]) ? (uint32_t) staged_mask[seq] : 0u;
    const uint32_t members = newmask | omask;
    if (members == 0) return;
    const int st = seq / n_seqps;
    const int pool = (seq % n_seqps) * band_size + (g % band_size);
    const half * rsd = k_resid + (size_t) seq * C * GGML_KPC_GROUP;
    const uint8_t * s = scalezp + (size_t) st * sz_nb2 + (size_t) pool * sz_nb1;
    float ss, zmn, sz; kpc_slab_super(s, &ss, &zmn, &sz);
    const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C;
    const int gtid = blockIdx.z * blockDim.x + threadIdx.x;   // byte index, strided across the z-blocks
    const int gstride = gridDim.z * blockDim.x;
    for (int w = 0; w < GGML_KPC_GROUP; ++w) {
        if (!(members & (1u << w))) continue;
        const bool    isnew = newmask & (1u << w);
        const int64_t cell  = isnew ? cell_of[w] : (int64_t) resid_slots[seq * GGML_KPC_GROUP + w];
        uint8_t * row = K + (size_t) cell * k_nb1;
        for (int b = gtid; b < krow; b += gstride) {
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

// staging update (separate launch after kpc_write_kernel on the same stream, so the seal's reads of the old
// staging are complete). Only the seq's open group (no later token this ubatch) persists the residual window +
// staged_group/staged_mask for the next ubatch; non-open blocks bail, so exactly one block per seq writes them.
static __global__ void kpc_write_stage_kernel(
        const float * k_cur, const int32_t * kpc_seq, const int32_t * kpc_pos, const int64_t * k_idxs,
        half * k_resid, int32_t * resid_slots, int32_t * staged_group, int32_t * staged_mask,
        int C, int nt, int64_t kc_nb0, int64_t kc_nb1) {
    const int g   = blockIdx.x;
    const int seq = blockIdx.y;

    __shared__ uint32_t newmask;
    __shared__ int      has_later;
    __shared__ int      tok_of[GGML_KPC_GROUP];
    __shared__ int64_t  cell_of[GGML_KPC_GROUP];
    kpc_scan_members(kpc_seq, kpc_pos, k_idxs, g, seq, nt, &newmask, &has_later, tok_of, cell_of);

    if (has_later) return;                                      // not the open group -> never touches staging
    const uint32_t omask   = (g == staged_group[(1 + GGML_KPC_GROUP) * seq]) ? (uint32_t) staged_mask[seq] : 0u;
    const uint32_t members = newmask | omask;
    if (members == 0) return;

    half * rsd = k_resid + (size_t) seq * C * GGML_KPC_GROUP;
    for (int w = 0; w < GGML_KPC_GROUP; ++w) {
        if (!(newmask & (1u << w))) continue;                  // staged members already have residual + cell
        for (int c = threadIdx.x; c < C; c += blockDim.x) rsd[(size_t) w * C + c] = __float2half(KPC_KC(tok_of[w], c));
        if (threadIdx.x == 0) resid_slots[seq * GGML_KPC_GROUP + w] = (int32_t) cell_of[w];
    }
    if (threadIdx.x == 0) { staged_group[(1 + GGML_KPC_GROUP) * seq] = g; staged_mask[seq] = (int32_t) members; }
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
    const int   n_seq_max = pr[0];
    const int   n_stream  = scalezp->ne[2];
    const int   ng_max    = scalezp->ne[1];
    const int   n_seqps   = n_seq_max / n_stream;
    const int   band_size = ng_max / n_seqps;
    // smem = scale[C] + zp[C] (member min/max) + the preserved OLD slab (2*C+6 bytes) the survivor rescue dequants against.
    const size_t smem = 2 * C * sizeof(float) + (size_t)(2 * C + 6);
    dim3 grid(ng_max, n_seq_max, 1);   // one block per (logical group, sequence); empty blocks early-return
    // size the write block to the channel count: the per-channel min/max and super-quant parallelise over C
    // (=n_kvh*head_dim). 256 starves wide caches (deepseek C=2048 -> +28% decode going 256->1024 threads).
    int wblk = ((C + 31) / 32) * 32; if (wblk > 1024) wblk = 1024; if (wblk < 64) wblk = 64;
    // Single-group decode (nt<=32): seal without packing, then fan the pack across SMs (kpc_pack_kernel). Prefill:
    // pack fused in the seal (one members scan -- a separate pack would re-scan O(nt)/block and cost ~13% prefill).
    if (nt <= GGML_KPC_GROUP) {
        kpc_write_kernel<false><<<grid, wblk, smem, ctx.stream()>>>(
            (const float *) k_cur->data, (const int32_t *) kseq->data, (const int32_t *) kpos->data,
            (const int64_t *) k_idxs->data, (half *) k_resid->data, (uint8_t *) scalezp->data,
            (int32_t *) rslots->data, (int32_t *) sgrp->data, (int32_t *) smask->data, (uint8_t *) dst->data,
            C, nt,
            n_seqps, band_size, scalezp->nb[1], scalezp->nb[2], dst->nb[1], k_cur->nb[0], k_cur->nb[1]);
        // scale the pack fan-out to the row width (krow = C/2 bytes): wide caches (deepseek krow=1024) need ~16
        // blocks to spread the bandwidth; tiny caches (qwen2 krow=64) are launch-bound and want a single block.
        int nblk_z = C / 128; if (nblk_z < 1) nblk_z = 1; if (nblk_z > 16) nblk_z = 16;
        kpc_pack_kernel<<<dim3(ng_max, n_seq_max, nblk_z), 256, 0, ctx.stream()>>>(
            (const float *) k_cur->data, (const int32_t *) kseq->data, (const int32_t *) kpos->data,
            (const int64_t *) k_idxs->data, (const half *) k_resid->data, (const uint8_t *) scalezp->data,
            (int32_t *) rslots->data, (int32_t *) sgrp->data, (int32_t *) smask->data, (uint8_t *) dst->data,
            C, nt,
            n_seqps, band_size, scalezp->nb[1], scalezp->nb[2], dst->nb[1], k_cur->nb[0], k_cur->nb[1]);
    } else {
        kpc_write_kernel<true><<<grid, wblk, smem, ctx.stream()>>>(
            (const float *) k_cur->data, (const int32_t *) kseq->data, (const int32_t *) kpos->data,
            (const int64_t *) k_idxs->data, (half *) k_resid->data, (uint8_t *) scalezp->data,
            (int32_t *) rslots->data, (int32_t *) sgrp->data, (int32_t *) smask->data, (uint8_t *) dst->data,
            C, nt,
            n_seqps, band_size, scalezp->nb[1], scalezp->nb[2], dst->nb[1], k_cur->nb[0], k_cur->nb[1]);
    }
    // second launch: persist the staging window, serialized after the seal on the same stream.
    kpc_write_stage_kernel<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) k_cur->data, (const int32_t *) kseq->data, (const int32_t *) kpos->data,
        (const int64_t *) k_idxs->data, (half *) k_resid->data,
        (int32_t *) rslots->data, (int32_t *) sgrp->data, (int32_t *) smask->data,
        C, nt, k_cur->nb[0], k_cur->nb[1]);
}
#undef KPC_KC

// One block per 32-token group: the group shares one scalezp slab, so read it + compute the per-channel scale/zp
// ONCE into shared memory, then dequant all 32 keys. Amortises the slab read 32x -- re-reading the C-wide slab per
// key dominates at small C (qwen2 C=128: slab ~= output, ~17% peak BW).
static __global__ void kpc_dequant_grouped_kernel(
        const uint8_t * pk, const uint8_t * sz0, const int32_t * gid, int gi_stride, half * dst,
        int C, int n_kv, int ns, int64_t k_nb1, int64_t k_nb2, int64_t sz_nb1, int64_t sz_nb2,
        int64_t d_nb1, int64_t d_nb2) {
    const int g = blockIdx.x;     // logical 32-token group
    const int s = blockIdx.y;
    if (s >= ns) return;
    const int k0 = g * GGML_KPC_GROUP;
    if (k0 >= n_kv) return;
    // pools are per-group (one scalezp slab per 32-token group), so gid is uniform across a group -> read it once.
    int pool = gid ? gid[(size_t) s * gi_stride + k0] : g;
    if (pool < 0) pool = 0;
    const uint8_t * slab = sz0 + (size_t) s * sz_nb2 + (size_t) pool * sz_nb1;
    float ss, zmn, szc; kpc_slab_super(slab, &ss, &zmn, &szc);
    const uint8_t * qs = slab + 6; const uint8_t * qz = slab + 6 + C;
    extern __shared__ float smem[];   // scale[C], zp[C] -- computed once, reused by all 32 keys
    float * scale = smem; float * zp = smem + C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { scale[c] = qs[c] * ss; zp[c] = zmn + qz[c] * szc; }
    __syncthreads();
    const int kmax = (k0 + GGML_KPC_GROUP <= n_kv) ? GGML_KPC_GROUP : (n_kv - k0);
    for (int idx = threadIdx.x; idx < kmax * C; idx += blockDim.x) {
        const int kk = idx / C, c = idx - kk * C;
        const int t = k0 + kk;
        const uint8_t * row  = pk  + (size_t) s * k_nb2 + (size_t) t * k_nb1;
        half          * drow = (half *)((char *) dst + (size_t) s * d_nb2 + (size_t) t * d_nb1);
        drow[c] = __float2half(kpc_deq_nib(row[c >> 1], c, scale[c], zp[c]));
    }
}

// dispatch the grouped dequant kernel.
static void kpc_dequant_launch(ggml_backend_cuda_context & ctx, const uint8_t * pk, const uint8_t * sz0,
        const int32_t * gid, int gi_stride, half * dstp, int C, int n_kv, int ns,
        int64_t k_nb1, int64_t k_nb2, int64_t sz_nb1, int64_t sz_nb2, int64_t d_nb1, int64_t d_nb2) {
    // grouped fast path (one 32-key group per block, slab read once). Pools are per-group so gid is uniform within
    // a group -> valid for both positional (gid==null) and remapped (gid!=null) caches.
    const int n_groups = (n_kv + GGML_KPC_GROUP - 1) / GGML_KPC_GROUP;
    dim3 grid(n_groups, ns, 1);
    kpc_dequant_grouped_kernel<<<grid, 256, 2 * (size_t) C * sizeof(float), ctx.stream()>>>(
        pk, sz0, gid, gi_stride, dstp, C, n_kv, ns, k_nb1, k_nb2, sz_nb1, sz_nb2, d_nb1, d_nb2);
}

void ggml_cuda_kpc_dequant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * pk = dst->src[0];
    const ggml_tensor * sz = dst->src[1];
    const ggml_tensor * gi = dst->src[2];   // group_index [n_kv, ns] (cell->pool); null -> positional
    const int C    = pk->ne[0];
    const int n_kv = pk->ne[1];
    const int ns   = pk->ne[2];
    const int gi_stride = gi ? (int) (gi->nb[1] / sizeof(int32_t)) : 0;
    kpc_dequant_launch(ctx, (const uint8_t *) pk->data, (const uint8_t *) sz->data,
        gi ? (const int32_t *) gi->data : nullptr, gi_stride, (half *) dst->data,
        C, n_kv, ns, pk->nb[1], pk->nb[2], sz->nb[1], sz->nb[2], dst->nb[1], dst->nb[2]);
}

// Coalesced q4_1 V-cache dequant -> contiguous f16 [DV, n_kv, n_vh, ns]. The stock to_fp16_nc converter runs this
// at ~19% peak BW for the decode V shape (3x slower than the grouped K dequant on the same element count); a
// warp's 32 lanes here read consecutive d (same q4_1 block's d/m broadcast, consecutive nibbles) and write
// consecutive f16. q4_1 block = 20 bytes: half d, half m, then 16 nibble-bytes (val[j] = d*nib + m).
static __global__ void kpc_vdeq_q41_kernel(
        const uint8_t * V, half * dst, int DV, int n_kv, int n_vh, int ns,
        int64_t s1, int64_t s2, int64_t s3) {   // q4_1-block strides for kv, vh, ns
    const long total = (long) DV * n_kv * n_vh * ns;
    for (long idx = (long) blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long) gridDim.x * blockDim.x) {
        const int  d  = (int)(idx % DV); long t = idx / DV;
        const int  kv = (int)(t % n_kv); t /= n_kv;
        const int  vh = (int)(t % n_vh); const int s = (int)(t / n_vh);
        const uint8_t * blk = V + (size_t) 20 * ((size_t) s * s3 + (size_t) vh * s2 + (size_t) kv * s1 + (size_t)(d >> 5));
        const int j = d & 31;
        const float dd = __half2float(*(const half *)(blk + 0));
        const float mm = __half2float(*(const half *)(blk + 2));
        const uint8_t q = blk[4 + (j & 15)];
        const int nib = (j < 16) ? (q & 0xF) : (q >> 4);
        dst[(size_t) DV * ((size_t) kv + (size_t) n_kv * ((size_t) vh + (size_t) n_vh * s)) + d] = __float2half(dd * nib + mm);
    }
}

// prefill fast path: dequant-once -> stock flash-attn.
// Materialise the int4 K-cache ([C_full, n_kv, ns]) and the q4_1 V-cache as f16 in pool scratch (kpc_dequant_launch
// for K, the stock q4_1->f16 converter for V), then hand a synthesized GGML_OP_FLASH_ATTN_EXT over strided f16
// K/V to the stock CUDA kernel, which amortises the dequant across all queries. Returns false if the stock path
// does not support this config (caller falls back to the custom MMA prefill kernel).
// n_kv is padded to a multiple of FATTN_KQ_STRIDE by the kv-cache, so the stock tiling never over-reads.
static bool kpc_flash_prefill_stock(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * Q, const ggml_tensor * Kp, const ggml_tensor * sz, const ggml_tensor * V,
        const ggml_tensor * mask, const ggml_tensor * gi, const ggml_tensor * sinks, const float * params,
        int head_dim, int DV, int C_full, int n_kv, int n_kvh, int n_vh, int n_stream, bool v_q41) {
    // head_dim == DV is guaranteed by the caller's `tiled` gate; the stock path also requires V->ne[0]==K->ne[0].
    // --- K: int4 -> f16 [C_full, n_kv, ns] (one dequant pass over the whole cache) ---
    ggml_cuda_pool_alloc<half> Kf16(ctx.pool(), (size_t) C_full * n_kv * n_stream);
    {
        const int gi_stride = gi ? (int) (gi->nb[1] / sizeof(int32_t)) : 0;
        kpc_dequant_launch(ctx, (const uint8_t *) Kp->data, (const uint8_t *) sz->data,
            gi ? (const int32_t *) gi->data : nullptr, gi_stride, Kf16.ptr, C_full, n_kv, n_stream,
            Kp->nb[1], Kp->nb[3], sz->nb[1], sz->nb[2],
            (int64_t) C_full * sizeof(half), (int64_t) C_full * (size_t) n_kv * sizeof(half));
    }
    // --- V: q4_1 -> f16 [DV, n_kv, n_vh, ns], or pass f16 V through unchanged ---
    ggml_cuda_pool_alloc<half> Vf16(ctx.pool());
    const half * V_data = (const half *) V->data;
    int64_t v_nb1 = V->nb[1], v_nb2 = V->nb[2], v_nb3 = V->nb[3];
    if (v_q41) {
        Vf16.alloc((size_t) DV * n_kv * n_vh * n_stream);
        const size_t ts = ggml_type_size(GGML_TYPE_Q4_1);
        // Small heads (qwen2 DV=64): the stock to_fp16_nc is launch/overhead-bound (~19% peak BW) and the custom
        // coalesced dequant is ~3x faster (qwen2 decode +6%). Large heads (gemma2 DV=256): the stock kernel's
        // vectorized loads win, so keep it.
        const bool use_custom_vdeq = (DV <= 64);
        if (use_custom_vdeq) {
            const long vtot = (long) DV * n_kv * n_vh * n_stream;
            int vblocks = (int)((vtot + 255) / 256); if (vblocks < 1) vblocks = 1; if (vblocks > 65535) vblocks = 65535;
            kpc_vdeq_q41_kernel<<<vblocks, 256, 0, ctx.stream()>>>(
                (const uint8_t *) V->data, Vf16.ptr, DV, n_kv, n_vh, n_stream,
                V->nb[1] / ts, V->nb[2] / ts, V->nb[3] / ts);
        } else {
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(GGML_TYPE_Q4_1);
            to_fp16(V->data, Vf16.ptr, DV, n_kv, n_vh, n_stream,
                    V->nb[1] / ts, V->nb[2] / ts, V->nb[3] / ts, ctx.stream());
        }
        V_data = Vf16.ptr;
        v_nb1 = (int64_t) DV * sizeof(half);
        v_nb2 = (int64_t) DV * (size_t) n_kv * sizeof(half);
        v_nb3 = (int64_t) DV * (size_t) n_kv * (size_t) n_vh * sizeof(half);
    }
    // --- synthesize f16 K/V tensors (strided views over the scratch) + a FLASH_ATTN_EXT dst ---
    ggml_tensor Kf = *Kp;
    Kf.type = GGML_TYPE_F16; Kf.data = Kf16.ptr; Kf.view_src = nullptr; Kf.view_offs = 0;
    Kf.ne[0] = head_dim; Kf.ne[1] = n_kv; Kf.ne[2] = n_kvh; Kf.ne[3] = n_stream;
    Kf.nb[0] = sizeof(half);                          Kf.nb[1] = (size_t) C_full   * sizeof(half);
    Kf.nb[2] = (size_t) head_dim * sizeof(half);      Kf.nb[3] = (size_t) C_full * (size_t) n_kv * sizeof(half);

    ggml_tensor Vf = *V;
    Vf.type = GGML_TYPE_F16; Vf.data = (void *) V_data; Vf.view_src = nullptr; Vf.view_offs = 0;
    Vf.ne[0] = DV; Vf.ne[1] = n_kv; Vf.ne[2] = n_vh; Vf.ne[3] = n_stream;
    Vf.nb[0] = sizeof(half); Vf.nb[1] = v_nb1; Vf.nb[2] = v_nb2; Vf.nb[3] = v_nb3;

    ggml_tensor dt = *dst;
    dt.op = GGML_OP_FLASH_ATTN_EXT;
    dt.src[0] = (ggml_tensor *) Q;
    dt.src[1] = &Kf;
    dt.src[2] = &Vf;
    dt.src[3] = (ggml_tensor *) mask;
    dt.src[4] = (ggml_tensor *) sinks;
    for (int i = 5; i < GGML_MAX_SRC; ++i) dt.src[i] = nullptr;
    // op_params [scale, max_bias, logit_softcap] already match; slot 3 carries n_seq_max in the KPC op but
    // precision in the stock op -> set F32 to match the custom kernel's accumulation.
    ((int32_t *) dt.op_params)[3] = (int32_t) GGML_PREC_F32;
    GGML_UNUSED(params);

    if (!ggml_cuda_flash_attn_ext_supported(ctx.device, &dt)) {
        return false;
    }
    ggml_cuda_flash_attn_ext(ctx, &dt);
    return true;
}

// requant roped f32 K [C,n_kv,ns] -> packed int4 + scalezp, in place. One block per (pool, stream);
// cells are bucketed by group_index[cell]==pool, so a RoPE shift that regroups cells re-encodes each
// referenced pool consistently. min/max -> scale=(mx-mn)/15 + int8 double-quant slab.
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
    for (int i = 0; i < n_kv; ++i) {   // repack each pool cell, one whole byte per thread (no nibble RMW)
        if (gid_s[i] != p) continue;
        uint8_t     * row = pk + (size_t) s * k_nb2 + (size_t) i * k_nb1;
        const float * rc  = (const float *)((const char *) rbase + (size_t) i * r_nb1);
        for (int b = threadIdx.x; b < krow; b += blockDim.x) {
            float d0 = qs[2*b]  *ss; if (d0 == 0.0f) d0 = 1.0f;   // match CPU requant: tiny scale underflows to 0
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
    // all four KPC ops run on the GPU: K/scalezp/k_resid/staging are device-resident and mutated in place, so the
    // shift's dequant->rope->requant must stay on-device.
    switch (op->op) {
        case GGML_OP_KPC_FLASH_ATTN:
            // the runtime <0,0> decode kernel sizes its per-lane registers for head_dim <= 256; reject above so the
            // op falls back cleanly instead of writing past the fixed arrays.
            return op->src[0]->ne[0] <= 256;
        case GGML_OP_KPC_WRITE:
            // the seal kernel stages scale[C]+zp[C] AND the preserved OLD slab (2*C+6 B, for the survivor rescue) in
            // dynamic shared memory; cap C at the 48 KB per-block default so a very wide GQA cache falls back cleanly.
            return 2 * op->src[0]->ne[0] * (int64_t) sizeof(float) + 2 * op->src[0]->ne[0] + 6 <= 48 * 1024;
        case GGML_OP_KPC_DEQUANT:
        case GGML_OP_KPC_REQUANT:
            // dequant/requant stage 2*C scale/zp floats in dynamic shared memory; same 48 KB per-block cap.
            return 2 * op->src[0]->ne[0] * (int64_t) sizeof(float) <= 48 * 1024;
        default:
            return false;
    }
}
