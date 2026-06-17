#pragma once

#include "common.cuh"

// CUDA port of the KPC per-channel int4 K-cache ops (branch kpc-cuda-kv).
// GPU-native redesign vs the CPU scheme: sealed 32-token groups with a POSITIONAL int8-double-quant
// scale/zp slab (group = token>>5; no group_index pool, no virtualization, no rescue) + an fp16
// residual window for the open <32-token tail. Kernel math validated standalone on the P40
// (/tmp/kpc-cuda/{seal_test,decode_test}.cu: round-trip NMSE 3.5e-4, attn-read NMSE 3.2e-5).

// GGML_KPC_GROUP and GGML_KPC_SZ_GROUP_BYTES come from ggml.h.

void ggml_cuda_kpc_flash_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_kpc_write     (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_kpc_requant   (ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// supports_op gate. Returns false until the kv-cache GPU-native integration (M3) lands, so the
// scheduler keeps KPC on the CPU backend and nothing regresses.
bool ggml_cuda_kpc_supported(const ggml_tensor * op);
