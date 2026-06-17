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
static __global__ void kpc_flash_decode_kernel(
        const float * Q, const uint8_t * Kp, const uint8_t * sz0, const uint8_t * V, const half * mask,
        const float * sinks, float kqscale, float max_bias, float logit_softcap, float * dst,
        int head_dim, int DV, int C_full, int n_kv, int n_head, int n_q, int n_kvh, int n_vh, int n_stream,
        int rk2, int rv2, int rk3, int rv3, bool v_q41,
        int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
        int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
        int64_t sz_nb1, int64_t sz_nb2,
        int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
        int64_t m_nb1, int64_t m_nb2, int64_t m_nb3, int m_ne2, int m_ne3,
        int64_t dst_nb1, int dst_ne1, int dst_ne2) {
    const int iq2 = blockIdx.x;   // query head
    const int iq1 = blockIdx.y;   // query position
    const int iq3 = blockIdx.z;   // stream
    if (iq2 >= n_head || iq1 >= n_q || iq3 >= n_stream || threadIdx.x != 0) return;

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

    float m = -INFINITY, l = 0.0f;
    extern __shared__ float acc[];                            // DV
    for (int d = 0; d < DV; ++d) acc[d] = 0.0f;

    for (int t = 0; t < n_kv; ++t) {
        const float mv = mp ? __half2float(mp[t]) : 0.0f;
        if (mp && mv == -INFINITY) continue;
        const uint8_t * s = sz0 + (size_t) ik3 * sz_nb2 + (size_t)(t / GGML_KPC_GROUP) * sz_nb1;
        float ss, zmn, szc; kpc_slab_super(s, &ss, &zmn, &szc);
        const uint8_t * qs = s + 6; const uint8_t * qz = s + 6 + C_full;
        const uint8_t * krow = (const uint8_t *) Kp + (size_t) t * k_nb1 + (size_t) ik2 * k_nb2 + (size_t) ik3 * k_nb3;
        float score = 0.0f;
        for (int c = 0; c < head_dim; ++c) {                 // dot over this kv-head's channel slice
            const int fc = cbase + c;
            score += qrow[c] * kpc_deq_nib(krow[c >> 1], c, qs[fc] * ss, zmn + qz[fc] * szc);
        }
        score *= kqscale;
        if (logit_softcap != 0.0f) score = logit_softcap * tanhf(score / logit_softcap);
        score += slope * mv;

        float mn = fmaxf(m, score), a = expf(m - mn), p = expf(score - mn);
        l = l * a + p;
        const uint8_t * vrow = (const uint8_t *) V + (size_t) t * v_nb1 + (size_t) iv2 * v_nb2 + (size_t) iv3 * v_nb3;
        if (v_q41) {
            for (int d = 0; d < DV; ++d) {
                const uint8_t * blk = vrow + (d / 32) * 20;          // q4_1 block: [f16 d][f16 m][16B nibbles]
                const float vd = __half2float(*(const half *)(blk + 0));
                const float vm = __half2float(*(const half *)(blk + 2));
                const int j = d % 32;
                const uint8_t b = blk[4 + (j % 16)];
                const int qv = (j < 16) ? (b & 0x0F) : (b >> 4);
                acc[d] = acc[d] * a + p * (qv * vd + vm);
            }
        } else {                                                    // f16 V
            const half * vh = (const half *) vrow;
            for (int d = 0; d < DV; ++d) acc[d] = acc[d] * a + p * __half2float(vh[d]);
        }
        m = mn;
    }
    if (sinks) {                                                    // attention sink: fold sink logit into denom
        const float sk = sinks[iq2];
        float mn = fmaxf(m, sk), a = expf(m - mn), p = expf(sk - mn);
        l = l * a + p;
        for (int d = 0; d < DV; ++d) acc[d] = acc[d] * a;
        m = mn;
    }
    const float Sinv = (l == 0.0f) ? 0.0f : 1.0f / l;
    float * out = (float *)((char *) dst + (size_t)((size_t) iq3 * dst_ne2 * dst_ne1 + iq2 + (size_t) iq1 * dst_ne1) * dst_nb1);
    for (int d = 0; d < DV; ++d) out[d] = acc[d] * Sinv;
}

// ---- host wrappers (thin launchers; exercised once M3 wires the GPU-native tensors) ----

void ggml_cuda_kpc_flash_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * Kp    = dst->src[1];
    const ggml_tensor * sz    = dst->src[2];
    const ggml_tensor * V     = dst->src[3];
    const ggml_tensor * mask  = dst->src[4];
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

    const size_t smem = DV * sizeof(float);
    dim3 grid(n_head, n_q, n_stream);
    kpc_flash_decode_kernel<<<grid, 1, smem, ctx.stream()>>>(
        (const float *) Q->data, (const uint8_t *) Kp->data, (const uint8_t *) sz->data,
        (const uint8_t *) V->data, mask ? (const half *) mask->data : nullptr,
        sinks ? (const float *) sinks->data : nullptr, params[0], params[1], params[2], (float *) dst->data,
        head_dim, DV, C_full, n_kv, n_head, n_q, n_kvh, n_vh, n_stream,
        rk2, rv2, rk3, rv3, v_q41,
        Q->nb[1], Q->nb[2], Q->nb[3],
        Kp->nb[1], Kp->nb[2], Kp->nb[3],
        sz->nb[1], sz->nb[2],
        V->nb[1], V->nb[2], V->nb[3],
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        mask ? (int) mask->ne[2] : 1, mask ? (int) mask->ne[3] : 1,
        dst->nb[1], (int) dst->ne[1], (int) dst->ne[2]);
}

// device write: ONE block, serial over the ubatch tokens (no cross-group races). Each token is staged into
// k_resid[w] (remembering its GLOBAL cell in k_resid_slots) and its open group is re-encoded into K from all
// staged members, so the read stays positional (K holds valid int4 for every written cell). Multi-stream:
// stream s = seq/n_seqps, scalezp pool = (seq%n_seqps)*band_size + group%band_size (mirrors the CPU pool_of);
// K cell is the GLOBAL slot from k_idxs (= s*kv_size + local). No rescue/virtualization (GPU-native design).
static __global__ void kpc_write_kernel(
        const float * k_cur, const int32_t * kpc_seq, const int32_t * kpc_pos, const int64_t * k_idxs,
        half * k_resid, uint8_t * scalezp, int32_t * resid_slots, int32_t * staged_group, int32_t * staged_mask,
        uint8_t * K, int C, int nt, int slot_shift, int seq_mask, int L,
        int n_seqps, int band_size, int64_t sz_nb1, int64_t sz_nb2, int64_t k_nb1) {
    if (blockIdx.x != 0) return;
    extern __shared__ float smem[];              // scale[C], zp[C]
    float * scale = smem; float * zp = smem + C;
    const int krow = C / 2;
    for (int i = 0; i < nt; ++i) {
        const int sb   = kpc_seq[i] & seq_mask;
        const int slot = (int) ((uint32_t) kpc_seq[i] >> slot_shift);
        if (slot >= L) continue;                 // spill sentinel: no staging (rare; skip for first cut)
        const int pos  = kpc_pos[i];
        const int g = pos / GGML_KPC_GROUP, w = pos % GGML_KPC_GROUP;
        const int st   = sb / n_seqps;                                   // physical stream
        const int pool = (sb % n_seqps) * band_size + (g % band_size);   // scalezp pool (CPU pool_of)
        half * rsd = k_resid + (size_t) slot * C * GGML_KPC_GROUP;
        if (threadIdx.x == 0 && staged_group[slot] != g) { staged_group[slot] = g; staged_mask[slot] = 0; }
        __syncthreads();
        for (int c = threadIdx.x; c < C; c += blockDim.x) rsd[(size_t) w * C + c] = __float2half(k_cur[(size_t) i * C + c]);
        __syncthreads();
        if (threadIdx.x == 0) { staged_mask[slot] |= (1u << w); resid_slots[slot * GGML_KPC_GROUP + w] = (int32_t) k_idxs[i]; }
        __syncthreads();
        const uint32_t members = (uint32_t) staged_mask[slot];
        // per-channel scale/zp over staged members
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float mn = INFINITY, mx = -INFINITY;
            for (int m = 0; m < GGML_KPC_GROUP; ++m) {
                if (!(members & (1u << m))) continue;
                float v = __half2float(rsd[(size_t) m * C + c]);
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
        // pack every staged member's int4 into its GLOBAL K cell (from k_resid_slots).
        // Thread owns a whole BYTE (channels 2b low + 2b+1 high): packing per-channel would race on the
        // shared nibble byte (read-modify-write), corrupting ~half the cache non-deterministically.
        for (int m = 0; m < GGML_KPC_GROUP; ++m) {
            if (!(members & (1u << m))) continue;
            const int64_t cell = resid_slots[slot * GGML_KPC_GROUP + m];   // global slot = st*kv_size + local
            uint8_t    * row = K + (size_t) cell * k_nb1;
            const half * rv  = rsd + (size_t) m * C;
            for (int b = threadIdx.x; b < krow; b += blockDim.x) {
                const int c0 = 2 * b, c1 = 2 * b + 1;
                int q0 = (int)((__half2float(rv[c0]) - (zmn + qz[c0]*sz)) / (qs[c0]*ss) + 0.5f);
                int q1 = (int)((__half2float(rv[c1]) - (zmn + qz[c1]*sz)) / (qs[c1]*ss) + 0.5f);
                q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
                q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
                row[b] = (uint8_t)(q0 | (q1 << 4));
            }
        }
        __syncthreads();
    }
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
    kpc_write_kernel<<<1, 256, smem, ctx.stream()>>>(
        (const float *) k_cur->data, (const int32_t *) kseq->data, (const int32_t *) kpos->data,
        (const int64_t *) k_idxs->data, (half *) k_resid->data, (uint8_t *) scalezp->data,
        (int32_t *) rslots->data, (int32_t *) sgrp->data, (int32_t *) smask->data, (uint8_t *) dst->data,
        C, nt, GGML_KPC_SLOT_SHIFT, GGML_KPC_SEQ_MASK, L,
        n_seqps, band_size, scalezp->nb[1], scalezp->nb[2], dst->nb[1]);
}

void ggml_cuda_kpc_requant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("KPC CUDA requant (RoPE context-shift) not implemented yet (M4)");
}

bool ggml_cuda_kpc_supported(const ggml_tensor * op) {
    // M3 first run (single-stream): READ + WRITE on the GPU (K/scalezp/k_resid/staging device-resident,
    // mutated in place by the write kernel). REQUANT (RoPE shift) stays on CPU until its kernel lands (M4).
    if (getenv("KPC_CUDA_OFF")) return false;   // DEBUG: force KPC onto CPU for A/B comparison
    return op->op == GGML_OP_KPC_FLASH_ATTN || op->op == GGML_OP_KPC_WRITE;
}
