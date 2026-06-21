#pragma once

#include "common.cuh"

// CUDA port of the KPC per-channel int4 K-cache ops.
// Unlike the CPU scheme, groups are sealed positionally (group = token>>5, no group_index pool,
// virtualization or rescue) with an int8 double-quant scale/zp slab, plus an fp16 residual window
// for the open <32-token tail.

// GGML_KPC_GROUP and GGML_KPC_SZ_GROUP_BYTES come from ggml.h.

void ggml_cuda_kpc_flash_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_kpc_write     (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_kpc_dequant   (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_kpc_requant   (ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// supports_op gate for the four GGML_OP_KPC_* ops.
bool ggml_cuda_kpc_supported(const ggml_tensor * op);
