#include "kpc.cuh"

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
        int64_t dst_nb1, int dst_ne1, int dst_ne2, int n_split) {
    const int iq2 = blockIdx.x;   // query head
    const int iq1 = blockIdx.y;   // query position
    const int iq3 = blockIdx.z;   // stream
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
        const int n_head_log2 = 1 << (int) floorf(log2f((float) n_head));
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

    for (int t = wid; t < n_kv; t += n_split) {              // split-K: warp wid owns keys {wid, wid+n_split, ...}
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
    // attention sink: fold the per-head sink logit into the denominator once (on warp 0 only, so it is
    // counted a single time across the split). No V contribution.
    if (sinks && wid == 0) {
        const float sk = sinks[iq2];
        const float mn = fmaxf(m, sk), a = expf(m - mn), p = expf(sk - mn);
        l = l * a + p;
        for (int j = 0; j < ndv; ++j) acc[j] *= a;
        m = mn;
    }

    float * out = (float *)((char *) dst + (size_t)((size_t) iq3 * dst_ne2 * dst_ne1 + iq2 + (size_t) iq1 * dst_ne1) * dst_nb1);
    if (n_split == 1) {                                             // single warp -> emit directly, no reduction
        const float Sinv = (l == 0.0f) ? 0.0f : 1.0f / l;
        for (int j = 0; j < ndv; ++j) { const int d = lane + j * 32; if (d < DV) out[d] = acc[j] * Sinv; }
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
    const float Sinv = (l_g == 0.0f) ? 0.0f : 1.0f / l_g;
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        float a = 0.0f;
        for (int w = 0; w < n_split; ++w) a += (s_m[w] == -INFINITY) ? 0.0f : s_acc[(size_t) w * DV + d] * expf(s_m[w] - m_g);
        out[d] = a * Sinv;
    }
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
    float params[3]; memcpy(params, dst->op_params, sizeof(params));   // kq_scale, max_bias, logit_softcap

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
    // split-K: when there are few query blocks (decode), split the key loop across warps so the GPU stays
    // busy; prefill already has enough (head x query) blocks, so it stays at 1 warp (no reduction overhead).
    const long blocks = (long) n_head * n_q * n_stream;
    int n_split = 1;
    while (n_split < 8 && blocks * (n_split * 2) <= 2048 && (n_split * 2) <= n_kv) n_split *= 2;
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
    if      (head_dim == DV && head_dim ==  64) kpc_flash_decode_kernel< 64, 64><<<grid, nthreads, smem, ctx.stream()>>>(KPC_FA_ARGS);
    else if (head_dim == DV && head_dim == 128) kpc_flash_decode_kernel<128,128><<<grid, nthreads, smem, ctx.stream()>>>(KPC_FA_ARGS);
    else if (head_dim == DV && head_dim == 256) kpc_flash_decode_kernel<256,256><<<grid, nthreads, smem, ctx.stream()>>>(KPC_FA_ARGS);
    else                                        kpc_flash_decode_kernel<  0,  0><<<grid, nthreads, smem, ctx.stream()>>>(KPC_FA_ARGS);
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
        int n_seqps, int band_size, int64_t sz_nb1, int64_t sz_nb2, int64_t k_nb1) {
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
            float v = (newmask & (1u << w)) ? k_cur[(size_t) tok_of[w] * C + c]
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
            float v0 = isnew ? k_cur[(size_t) tok_of[w] * C + c0] : __half2float(rsd[(size_t) w * C + c0]);
            float v1 = isnew ? k_cur[(size_t) tok_of[w] * C + c1] : __half2float(rsd[(size_t) w * C + c1]);
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
        int C, int nt, int slot_shift, int L) {
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
        for (int c = threadIdx.x; c < C; c += blockDim.x) rsd[(size_t) w * C + c] = __float2half(k_cur[(size_t) tok_of[w] * C + c]);
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
        n_seqps, band_size, scalezp->nb[1], scalezp->nb[2], dst->nb[1]);
    // SECOND launch: persist the staging window. Serialized after the seal on the same stream, so the seal's
    // reads of the OLD staging are done before this writes the NEW staging -> no inter-block staging race.
    kpc_write_stage_kernel<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) k_cur->data, (const int32_t *) kseq->data, (const int32_t *) kpos->data,
        (const int64_t *) k_idxs->data, (half *) k_resid->data,
        (int32_t *) rslots->data, (int32_t *) sgrp->data, (int32_t *) smask->data,
        C, nt, GGML_KPC_SLOT_SHIFT, L);
}

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
            int q0 = (int)((rc[2*b]   - (zmn + qz[2*b]  *sz)) / (qs[2*b]  *ss) + 0.5f);
            int q1 = (int)((rc[2*b+1] - (zmn + qz[2*b+1]*sz)) / (qs[2*b+1]*ss) + 0.5f);
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
