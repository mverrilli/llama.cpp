#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-kpc.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat   (ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((2u*(1 + n_stream) + 6u)*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    if (type_k == GGML_TYPE_KPC4_1) {
        // each of the n_seqps sequences sharing a stream needs at least one scalezp pool in its band
        const uint32_t ng_max  = (kv_size + KPC_GROUP - 1) / KPC_GROUP;
        const uint32_t n_seqps = n_seq_max / n_stream;
        if (ng_max < n_seqps) {
            const uint32_t kv_min = n_seqps * KPC_GROUP;
            throw std::runtime_error(format(
                "KPC4_1 K cache needs kv_size/%d >= n_seq_max/n_stream (%u pools < %u seqs per stream). "
                "Raise the context size to >= %u, or %s", KPC_GROUP, ng_max, n_seqps, kv_min,
                n_stream == 1 ? "drop --kv-unified (it pins n_seq_max to LLAMA_MAX_SEQ)" : "reduce n_parallel"));
        }
    }

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (n_embd_head_v_all == 0) {
            n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
        } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
            n_embd_head_v_all = -1;
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        ggml_tensor * k_scalezp     = nullptr;
        ggml_tensor * k_resid       = nullptr;
        ggml_tensor * group_index   = nullptr;
        ggml_tensor * k_resid_slots = nullptr;
        ggml_tensor * staged_group  = nullptr;
        ggml_tensor * staged_mask   = nullptr;
        if (has_k && type_k == GGML_TYPE_KPC4_1) {
            const int64_t ng_max = (kv_size + KPC_GROUP - 1) / KPC_GROUP;
            // scalezp/k_resid/staging share K's buffer so the device write kernel can mutate them in place.
            // group_index stays on a host buffer: the device path pools positionally and only the host touches it.
            ggml_context * ctx_m = ctx_for_buft(ggml_backend_cpu_buffer_type());
            // scalezp indexed per stream (ng_max pools); staging sized by the staging-slot count
            k_scalezp = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, KPC_SZ_GROUP_BYTES(n_embd_k_gqa), ng_max, n_stream);
            ggml_format_name(k_scalezp, "cache_k_scalezp_l%d", il);
            k_resid = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd_k_gqa, KPC_GROUP, n_seq_max);
            ggml_format_name(k_resid, "cache_k_resid_l%d", il);
            group_index = ggml_new_tensor_2d(ctx_m, GGML_TYPE_I32, kv_size, n_stream);
            ggml_format_name(group_index, "cache_k_gidx_l%d", il);
            k_resid_slots = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KPC_GROUP, n_seq_max);
            ggml_format_name(k_resid_slots, "cache_k_resid_slots_l%d", il);
            // row 0 = the open group per staging slot (device-owned, written by the stage kernel); rows 1..KPC_GROUP =
            // committed survivor cell indices for the CUDA pool-re-encode rescue (host-owned, filled in set_input_kpc),
            // -1-terminated. Disjoint rows/timing from row 0, so no shared-write.
            staged_group = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1 + KPC_GROUP, n_seq_max);
            ggml_format_name(staged_group, "cache_k_staged_group_l%d", il);
            staged_mask = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seq_max);
            ggml_format_name(staged_mask, "cache_k_staged_mask_l%d", il);
        }

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, k_scalezp, k_resid, group_index, k_resid_slots, staged_group, staged_mask });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);

        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    kpc_reset_state();   // group_index starts unmapped (-1)

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            type_k != GGML_TYPE_KPC4_1 &&   // rotation breaks the per-channel layout
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek V3.2 DSA lightning indexer
        if (model.arch == LLM_ARCH_DEEPSEEK32 && hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    kpc_reset_state();
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                kpc_free_cell(seq_to_stream[seq_id], i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }

        // drop staged members in the removed range so a later write can't fold stale residuals
        if (kpc_enabled()) {
            if (cells.seq_pos_max(seq_id) < 0) {
                kpc_retire_seq(seq_id);
            } else {
                kpc_trim_staging(seq_id, p0, p1);
            }
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);
                kpc_free_cell(s, i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }

        if (kpc_enabled()) {
            for (uint32_t sq = 0; sq < n_seq_max; ++sq) {
                if (v_cells[seq_to_stream[sq]].seq_pos_max(sq) < 0) {
                    kpc_retire_seq(sq);
                } else {
                    kpc_trim_staging(sq, p0, p1);
                }
            }
        }
    }

    return true;
}

int32_t llama_kv_cache::kpc_staged_mask(uint32_t il, int32_t slot) const {
    if (slot < 0) {
        return 0;
    }
    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return 0;
    }
    const auto & layer = layers[it->second];
    if (!layer.staged_mask) {
        return 0;
    }
    int32_t mask = 0;   // staged_mask may be device-resident, so read via the backend
    ggml_backend_tensor_get(layer.staged_mask, &mask, (size_t) slot*sizeof(int32_t), sizeof(int32_t));
    return mask;
}

int32_t llama_kv_cache::kpc_slot_of(llama_seq_id seq_id) const {
    return (int32_t) seq_id;
}

void llama_kv_cache::kpc_clear_staging_slot(int32_t slot) const {
    if (slot < 0) {
        return;
    }
    const int32_t zero = 0;               // staged tensors may be device-resident, so write via the backend
    for (const auto & layer : layers) {   // no-op for non-KPC layers (staged tensors are null)
        if (layer.staged_mask)  ggml_backend_tensor_set(layer.staged_mask,  &zero, (size_t) slot*sizeof(int32_t), sizeof(int32_t));
        if (layer.staged_group) ggml_backend_tensor_set(layer.staged_group, &zero, (size_t) slot*(1+KPC_GROUP)*sizeof(int32_t), sizeof(int32_t));   // row 0 (group); survivor rows are transient
    }
}

void llama_kv_cache::kpc_retire_seq(llama_seq_id seq_id) const {
    if (!kpc_enabled()) {
        return;
    }
    kpc_clear_staging_slot(kpc_slot_of(seq_id));
}

void llama_kv_cache::kpc_trim_staging(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    if (!kpc_enabled()) {
        return;
    }
    const int32_t slot = kpc_slot_of(seq_id);
    if (slot < 0) {
        return;
    }
    for (const auto & layer : layers) {
        if (!layer.staged_mask) {
            continue;
        }
        int32_t mask = 0;   // staged tensors may be device-resident, so read/modify/write via the backend
        ggml_backend_tensor_get(layer.staged_mask, &mask, (size_t) slot*sizeof(int32_t), sizeof(int32_t));
        if (mask == 0) {
            continue;
        }
        int32_t grp = 0;
        ggml_backend_tensor_get(layer.staged_group, &grp, (size_t) slot*(1+KPC_GROUP)*sizeof(int32_t), sizeof(int32_t));   // row 0 = group
        const int32_t orig = mask;
        for (int w = 0; w < KPC_GROUP; ++w) {
            const llama_pos pos = (llama_pos) grp*KPC_GROUP + w;
            if ((mask & (1 << w)) && pos >= p0 && pos < p1) {
                mask &= ~(1 << w);
            }
        }
        if (mask != orig) {
            ggml_backend_tensor_set(layer.staged_mask, &mask, (size_t) slot*sizeof(int32_t), sizeof(int32_t));
        }
    }
}

void llama_kv_cache::kpc_free_cell(uint32_t strm, uint32_t i) const {
    if (!kpc_enabled()) {
        return;
    }
    for (const auto & layer : layers) {
        if (layer.group_index) {
            ((int32_t *) layer.group_index->data)[(size_t) strm*layer.group_index->ne[0] + i] = -1;
        }
    }
}

void llama_kv_cache::kpc_reset_state() const {
    if (!kpc_enabled()) {
        return;
    }
    for (const auto & layer : layers) {
        if (!layer.k_scalezp || !layer.group_index->data) {
            continue;
        }
        memset(layer.group_index->data, 0xFF, ggml_nbytes(layer.group_index));   // -1: all cells unmapped (host)
        // staging lives on the (possibly device) K buffer, so memset via the backend
        ggml_backend_tensor_memset(layer.staged_mask,  0, 0, ggml_nbytes(layer.staged_mask));
        ggml_backend_tensor_memset(layer.staged_group, 0, 0, ggml_nbytes(layer.staged_group));
    }
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        // retire dst's stale staging; the shared cells stay in src's pool band and a later re-encode rescues them
        kpc_retire_seq(seq_id_dst);

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            kpc_free_cell(seq_to_stream[seq_id], i);

            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }

    // every other sequence is gone; retire their KPC staging so reused seq ids start clean
    if (kpc_enabled()) {
        for (uint32_t sq = 0; sq < n_seq_max; ++sq) {
            if ((llama_seq_id) sq != seq_id) {
                kpc_retire_seq(sq);
            }
        }
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {   // cell dropped out of range and was removed
                kpc_free_cell(seq_to_stream[seq_id], i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }
    // surviving cells were renumbered; the KPC regroup happens in the K-shift pass has_shift triggers next update

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }

                if (layer.k_scalezp) {  // KPC: copy scale/zp + residual + staging state
                    auto copy_slab = [&](ggml_tensor * t) {
                        const size_t slab = ggml_nbytes(t) / n_stream;   // last dim is n_stream for all KPC state
                        std::vector<uint8_t> tmp(slab);
                        ggml_backend_tensor_get(t, tmp.data(), ssrc*slab, slab);
                        ggml_backend_tensor_set(t, tmp.data(), sdst*slab, slab);
                    };
                    copy_slab(layer.k_scalezp);
                    copy_slab(layer.k_resid);
                    copy_slab(layer.group_index);
                    copy_slab(layer.k_resid_slots);
                    copy_slab(layer.staged_group);
                    copy_slab(layer.staged_mask);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
                kpc_free_cell(sinfo.strm[s], idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    // KPC int4 K shifts via build_rope_shift's dequant/rope/requant path, so no n_stream restriction
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (k->type == GGML_TYPE_KPC4_1 && v_trans) {
        // flash-attn off (v_trans set): dequant packed K -> F16 for the mul_mat path, one slab per stream
        const int64_t C  = n_embd_k_gqa;
        const int64_t ng_max = layers[ikv].k_scalezp->ne[1];   // full scalezp pool (group_index points into it)
        ggml_tensor * sz_cache = layers[ikv].k_scalezp;
        ggml_tensor * gi_cache = layers[ikv].group_index;
        ggml_tensor * packed_v = ggml_view_3d(ctx, k, C, n_kv, ns, k->nb[1], k->nb[2], (int64_t)sinfo.s0*k->nb[2]);
        ggml_tensor * sz_v     = ggml_view_3d(ctx, sz_cache, KPC_SZ_GROUP_BYTES(C), ng_max, ns, sz_cache->nb[1], sz_cache->nb[2], (int64_t)sinfo.s0*sz_cache->nb[2]);
        ggml_tensor * gi_v     = ggml_view_2d(ctx, gi_cache, n_kv, ns, gi_cache->nb[1], (int64_t)sinfo.s0*gi_cache->nb[1]);
        ggml_tensor * kf16     = ggml_kpc_dequant(ctx, packed_v, sz_v, gi_v);
        return ggml_reshape_4d(ctx, kf16, hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns);
    }

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_k_scalezp(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    auto * k  = layers[ikv].k;
    auto * sz = layers[ikv].k_scalezp;
    const int64_t C  = k->ne[0];
    // group_index holds absolute (per-seq banded) pool indices, so return the full ng_max pool, not an n_kv slice
    GGML_UNUSED(n_kv);
    const int64_t ng = sz->ne[1];
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    return ggml_view_3d(ctx, sz, KPC_SZ_GROUP_BYTES(C), ng, ns, sz->nb[1], sz->nb[2], (int64_t)sinfo.s0*sz->nb[2]);
}

ggml_tensor * llama_kv_cache::get_k_groupidx(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    auto * gi = layers[ikv].group_index;
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    return ggml_view_2d(ctx, gi, n_kv, ns, gi->nb[1], (int64_t)sinfo.s0*gi->nb[1]);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, ggml_tensor * kpc_seq, ggml_tensor * kpc_pos, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (k->type == GGML_TYPE_KPC4_1) {
        // per-channel int4 K write: group by logical pos/32, requant full groups from f16 originals (k_resid);
        // kpc_seq/kpc_pos select the (seq,group) scalezp pool; placement scatters via k_idxs (SWA/defrag)
        GGML_ASSERT(kpc_seq && kpc_pos && "KPC write needs kpc_seq/kpc_pos inputs");
        return ggml_kpc_write(ctx, k, layers[ikv].k_scalezp, layers[ikv].k_resid, layers[ikv].group_index,
                                   layers[ikv].k_resid_slots, layers[ikv].staged_group, layers[ikv].staged_mask,
                                   k_cur, k_idxs, kpc_seq, kpc_pos, (int32_t) n_seq_max);
    }

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_kpc_seq(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ubatch.n_tokens);
    ggml_set_input(t);
    return t;
}

ggml_tensor * llama_kv_cache::build_input_kpc_pos(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ubatch.n_tokens);
    ggml_set_input(t);
    return t;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_kpc(ggml_tensor * seq, ggml_tensor * pos, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(seq->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pos->buffer));
    int32_t * seq_data = (int32_t *) seq->data;
    int32_t * pos_data = (int32_t *) pos->data;

    // same token ordering as set_input_k_idxs (ti = s*size + i) so kpc_seq / kpc_pos / k_idxs index-align
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            const uint32_t ti = s*sinfo.size() + i;
            GGML_ASSERT(ubatch->n_seq_id[ti] >= 1);
            const llama_seq_id sb = ubatch->seq_id[ti][0];   // primary seq for the (seq,group) pool
            GGML_ASSERT(sb >= 0 && sb < (int) n_seq_max);

            seq_data[ti] = sb;
            pos_data[ti] = ubatch->pos[ti];
        }
    }

    // the device write kernel pools positionally and never touches group_index, so rebuild it from the committed
    // cell positions here for the GPU read/dequant/requant.
    kpc_rebuild_group_index();

    // CUDA survivor-rescue (host half): fill staged_group rows 1.. with the committed cells mapped to each open
    // group's pool (gid==pool), -1 terminated. The CUDA write kernel skips the ones that are members of the current
    // write and re-encodes the rest against the new slab (matching the CPU rescue's resc cells). group_index is the
    // lifecycle-correct cell->pool map just rebuilt above and is identical across layers, so scan it once and write
    // every layer's rows 1.. via a partial backend-set; row 0 stays device-owned (the stage kernel writes it). A pool
    // can hold up to KPC_GROUP committed gid==pool cells (a sealed group re-targeted by seq_rm+continue), so fill up to
    // KPC_GROUP (members are dropped in-kernel); the kernel reads i<KPC_GROUP so -1 is needed only when the row isn't full.
    if (!layers.empty() && layers[0].staged_group && layers[0].group_index && layers[0].group_index->data) {
        const uint32_t kvs       = get_size();
        const uint32_t n_seqps   = n_seq_max / n_stream;
        const uint32_t band_size = (uint32_t) layers[0].k_scalezp->ne[1] / n_seqps;
        const int32_t  SG        = 1 + KPC_GROUP;                          // staged_group column stride (= ne[0])
        const int32_t * gi       = (const int32_t *) layers[0].group_index->data;

        // Per-seq open group = the LOWEST logical group written this ubatch (the group being extended, which may
        // hold prior committed cells that this write's seal must rescue). We key the survivor gather off this
        // rather than the device-owned staged_group row 0: a rejected speculative-draft batch that spilled into the
        // next group seals the lower group as "complete" and moves row 0 to the (then seq_rm-emptied) upper group,
        // while real writes keep landing in the lower group. Its committed cells then end up neither staged (omask)
        // nor gathered here -> silently re-quantized against a members-only slab (catastrophic; spec-decode). The
        // minimum written position can never be a to-be-rejected draft tail, so it is rejection-proof. omask still
        // keys off staged_group row 0 in-kernel; only this survivor-rescue list changes. seqs not written this
        // ubatch keep the prior staged_group/mask scan.
        std::vector<int32_t> open_grp(n_seq_max, -1);
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                const uint32_t     ti = s*sinfo.size() + i;
                const llama_seq_id sb = ubatch->seq_id[ti][0];
                const int32_t      g  = pos_data[ti] / KPC_GROUP;
                if (open_grp[sb] < 0 || g < open_grp[sb]) open_grp[sb] = g;
            }
        }

        int32_t surv[KPC_GROUP];                                           // <= KPC_GROUP-1 cell indices + the -1 terminator
        for (uint32_t slot = 0; slot < n_seq_max; ++slot) {
            int32_t g = open_grp[slot];                                    // open group from this ubatch's min position
            if (g < 0) {                                                   // seq not written this ubatch: prior behavior
                int32_t mask = 0;
                ggml_backend_tensor_get(layers[0].staged_mask, &mask, (size_t) slot*sizeof(int32_t), sizeof(int32_t));
                if (mask != 0) {
                    ggml_backend_tensor_get(layers[0].staged_group, &g, (size_t) slot*SG*sizeof(int32_t), sizeof(int32_t)); // row 0
                }
            }
            int n = 0;
            if (g >= 0) {
                const int32_t  pool = (int32_t)((slot % n_seqps)*band_size + ((uint32_t) g % band_size));
                const uint32_t s    = slot / n_seqps;                      // physical stream
                for (uint32_t cell = 0; cell < kvs && n < KPC_GROUP; ++cell) {   // cap KPC_GROUP: a re-targeted sealed
                    if (gi[(size_t) s*kvs + cell] == pool) surv[n++] = (int32_t) cell;   // group can have 32 gid==pool cells
                }
            }
            int wn = n;                                                   // entries to write (rows 1..)
            if (n < KPC_GROUP) { surv[n] = -1; wn = n + 1; }              // -1 terminator only when the row isn't full;
            for (const auto & layer : layers) {                          // n==KPC_GROUP fills all 32 rows, kernel reads i<32
                if (layer.staged_group) {
                    ggml_backend_tensor_set(layer.staged_group, surv,
                        (size_t)(slot*SG + 1)*sizeof(int32_t), (size_t) wn*sizeof(int32_t));
                }
            }
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

// Rebuild group_index (cell -> scalezp pool) from the current cell positions, for every layer.
// pool = seq_band + logical_group % band_size; empty cells map to -1. Used for the CUDA path (the device
// write pools positionally) and after a RoPE shift, where the physical cell no longer matches its position.
void llama_kv_cache::kpc_rebuild_group_index() const {
    const uint32_t kvs       = get_size();
    const uint32_t n_seqps   = n_seq_max / n_stream;
    const uint32_t ng_max    = (uint32_t) layers[0].k_scalezp->ne[1];
    const uint32_t band_size = ng_max / n_seqps;

    // a cell's int4 rows are packed under its current pool's scale, so when this maps a cell to a different pool
    // reconcile the scale slabs: a single source is a bit-exact slab copy; aliased or mixed dests re-encode the
    // union (dequant each vs its old slab, common min/max, repack int4).
    std::vector<int32_t> gid_old((size_t) kvs * n_stream);   // snapshot before the in-place rebuild below
    memcpy(gid_old.data(), layers[0].group_index->data, gid_old.size() * sizeof(int32_t));

    bool any_reloc = false;
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < kvs; ++i) {
            int32_t pool = -1;
            if (!cells.is_empty(i)) {
                llama_seq_id sq = -1;
                if (cells.seq_count(i) == 1) {
                    sq = cells.seq_get(i);
                } else {
                    for (uint32_t cand = 0; cand < n_seq_max; ++cand) {   // shared cell: first owner's band
                        if (cells.seq_has(i, cand)) { sq = (llama_seq_id) cand; break; }
                    }
                }
                const uint32_t lg   = (uint32_t) cells.pos_get(i) / KPC_GROUP;
                const uint32_t band = ((uint32_t) sq % n_seqps) * band_size;
                pool = (int32_t) (band + lg % band_size);
            }
            const int32_t old_pool = gid_old[(size_t) s*kvs + i];
            if (pool >= 0 && old_pool >= 0 && pool != old_pool) { any_reloc = true; }
            for (const auto & layer : layers) {
                ((int32_t *) layer.group_index->data)[(size_t) s*kvs + i] = pool;
            }
        }
    }
    if (!any_reloc) {
        return;   // steady decode: no relocation, no slab/K traffic
    }
    const int32_t * gidN = (const int32_t *) layers[0].group_index->data;   // post-rebuild cell->pool map

    const int    Cc     = (int)    layers[0].k->ne[0];           // channels (n_embd_k_gqa)
    const size_t slabsz = (size_t) layers[0].k_scalezp->nb[1];   // 2*C + 6 bytes
    const size_t krow   = (size_t) layers[0].k->nb[1];           // C/2 packed bytes per cell

    // mirror the (static, non-linkable) ggml-cpu/kpc.cpp slab helpers; fp16 round-trip via the public ggml API.
    auto slab_decode = [&](const uint8_t * slab, std::vector<float> & sca, std::vector<float> & zpc) {
        ggml_fp16_t h; float ss, zmn, sz;
        memcpy(&h, slab + 0, 2); ss  = ggml_fp16_to_fp32(h);
        memcpy(&h, slab + 2, 2); zmn = ggml_fp16_to_fp32(h);
        memcpy(&h, slab + 4, 2); sz  = ggml_fp16_to_fp32(h);
        for (int c = 0; c < Cc; ++c) { sca[c] = slab[6 + c] * ss; zpc[c] = zmn + slab[6 + Cc + c] * sz; }
    };
    auto slab_encode = [&](const std::vector<float> & sca, const std::vector<float> & zp, uint8_t * slab) {
        float smax = 0.0f; for (int c = 0; c < Cc; ++c) if (sca[c] > smax) smax = sca[c];
        float ss = smax / 255.0f; if (ss == 0.0f) ss = 1.0f;
        float zmn = INFINITY, zmx = -INFINITY; for (int c = 0; c < Cc; ++c) { if (zp[c] < zmn) zmn = zp[c]; if (zp[c] > zmx) zmx = zp[c]; }
        float sz = (zmx - zmn) / 255.0f; if (sz == 0.0f) sz = 1.0f;
        ggml_fp16_t h;
        h = ggml_fp32_to_fp16(ss);  memcpy(slab + 0, &h, 2);
        h = ggml_fp32_to_fp16(zmn); memcpy(slab + 2, &h, 2);
        h = ggml_fp32_to_fp16(sz);  memcpy(slab + 4, &h, 2);
        uint8_t * qs = slab + 6, * qz = slab + 6 + Cc;
        for (int c = 0; c < Cc; ++c) {
            int a = (int)(sca[c] / ss + 0.5f);        a = a<0?0:(a>255?255:a);
            int b = (int)((zp[c] - zmn) / sz + 0.5f); b = b<0?0:(b>255?255:b);
            qs[c] = (uint8_t) a; qz[c] = (uint8_t) b;
        }
    };

    // snapshot all old slabs up front so a merge reads pre-rebuild scales (a prior dest may have rewritten its
    // slab). each cell belongs to exactly one dest pool, so the int4 K needs no snapshot.
    std::vector<std::vector<uint8_t>> sz_old(layers.size());
    for (size_t l = 0; l < layers.size(); ++l) {
        sz_old[l].resize((size_t) layers[l].k_scalezp->nb[2] * n_stream);
        ggml_backend_tensor_get(layers[l].k_scalezp, sz_old[l].data(), 0, sz_old[l].size());
    }

    // distinct dest pools (stream, pool) that received >= 1 relocation
    std::vector<std::array<int32_t,2>> dests;
    for (uint32_t s = 0; s < n_stream; ++s) {
        for (uint32_t i = 0; i < kvs; ++i) {
            const int32_t np = gidN[(size_t) s*kvs + i], op = gid_old[(size_t) s*kvs + i];
            if (np >= 0 && op >= 0 && np != op) {
                bool seen = false; for (auto & d : dests) if (d[0]==(int32_t)s && d[1]==np) { seen=true; break; }
                if (!seen) dests.push_back({ (int32_t) s, np });
            }
        }
    }

    std::vector<float>   sca(Cc), zpc(Cc), mn(Cc), mx(Cc);
    std::vector<uint8_t> slab_new(slabsz);
    std::vector<int32_t> ucells;
    for (const auto & d : dests) {
        const uint32_t s = (uint32_t) d[0];
        const int32_t  X = d[1];
        const size_t   szoff = (size_t) s*layers[0].k_scalezp->nb[2] + (size_t) X*slabsz;   // X's slab byte offset (per layer)

        ucells.clear();
        int32_t single_src = -2;   // -2 unset, -1 multi-source, else the one source pool
        for (uint32_t i = 0; i < kvs; ++i) {
            if (gidN[(size_t) s*kvs + i] != X) continue;
            ucells.push_back((int32_t) i);
            const int32_t op = gid_old[(size_t) s*kvs + i];
            if      (single_src == -2) single_src = op;
            else if (single_src != op) single_src = -1;
        }
        if (ucells.empty()) continue;

        if (single_src >= 0) {
            // pure single-source relocation: copy the source slab to dest, bit-exact (int4 untouched)
            for (size_t l = 0; l < layers.size(); ++l) {
                const uint8_t * srcslab = sz_old[l].data() + (size_t) s*layers[l].k_scalezp->nb[2] + (size_t) single_src*slabsz;
                ggml_backend_tensor_set(layers[l].k_scalezp, srcslab, szoff, slabsz);
            }
            continue;
        }

        // merge: union has >1 source, or includes resident cells, so re-encode over the union per layer
        std::vector<uint8_t> krd(krow);
        std::vector<std::vector<float>> uf32(ucells.size(), std::vector<float>(Cc));   // dequantized union (reused per layer)
        for (size_t l = 0; l < layers.size(); ++l) {
            const uint8_t * szbase = sz_old[l].data() + (size_t) s*layers[l].k_scalezp->nb[2];
            for (int c = 0; c < Cc; ++c) { mn[c] = INFINITY; mx[c] = -INFINITY; }
            // dequant each union cell against its old slab, accumulate per-channel min/max
            for (size_t u = 0; u < ucells.size(); ++u) {
                const int32_t i  = ucells[u];
                const int32_t op = gid_old[(size_t) s*kvs + i];
                slab_decode(szbase + (size_t) op*slabsz, sca, zpc);
                ggml_backend_tensor_get(layers[l].k, krd.data(), (size_t) s*layers[l].k->nb[2] + (size_t) i*krow, krow);
                for (int c = 0; c < Cc; ++c) {
                    const int    q = (c & 1) ? (krd[c/2] >> 4) : (krd[c/2] & 0x0F);
                    const float  v = q * sca[c] + zpc[c];
                    uf32[u][c] = v;
                    if (v < mn[c]) mn[c] = v; if (v > mx[c]) mx[c] = v;
                }
            }
            for (int c = 0; c < Cc; ++c) { float sc = (mx[c] - mn[c]) / 15.0f; if (sc == 0.0f) sc = 1.0f; sca[c] = sc; zpc[c] = mn[c]; }
            slab_encode(sca, zpc, slab_new.data());
            ggml_backend_tensor_set(layers[l].k_scalezp, slab_new.data(), szoff, slabsz);
            // read back the int8-rounded effective scale/zp, then repack each union cell under the merged slab
            slab_decode(slab_new.data(), sca, zpc);
            for (int c = 0; c < Cc; ++c) if (sca[c] == 0.0f) sca[c] = 1.0f;
            for (size_t u = 0; u < ucells.size(); ++u) {
                const int32_t i = ucells[u];
                for (int b = 0; b < (int) krow; ++b) {
                    const int c0 = 2*b, c1 = 2*b + 1;
                    int q0 = (int)((uf32[u][c0] - zpc[c0]) / sca[c0] + 0.5f); q0 = q0<0?0:(q0>15?15:q0);
                    int q1 = (int)((uf32[u][c1] - zpc[c1]) / sca[c1] + 0.5f); q1 = q1<0?0:(q1>15?15:q1);
                    krd[b] = (uint8_t)(q0 | (q1 << 4));
                }
                ggml_backend_tensor_set(layers[l].k, krd.data(), (size_t) s*layers[l].k->nb[2] + (size_t) i*krow, krow);
            }
        }
    }
}

void llama_kv_cache::set_input_kpc_shift(ggml_tensor * gi_old) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(gi_old->buffer));
    GGML_ASSERT(kpc_enabled());

    const uint32_t kvs = get_size();

    // all layers carry the same cell->pool map; snapshot it for the in-graph dequant
    memcpy(gi_old->data, layers[0].group_index->data, (size_t) kvs*n_stream*sizeof(int32_t));

    // rebuild group_index from the post-shift positions; the in-graph requant re-encodes every referenced
    // pool. cursors reset and staging drops, so the next write rescues the open group.
    kpc_rebuild_group_index();

    for (uint32_t slot = 0; slot < n_seq_max; ++slot) {
        kpc_clear_staging_slot((int32_t) slot);
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
        // KPC side tensors are part of the K-cache footprint
        for (ggml_tensor * t : { layer.k_resid, layer.k_scalezp, layer.group_index,
                                 layer.k_resid_slots, layer.staged_group, layer.staged_mask }) {
            if (t) {
                size_k_bytes += ggml_nbytes(t);
            }
        }
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il,
                ggml_tensor * kpc_gi_old) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (type_k() == GGML_TYPE_KPC4_1) {
        // KPC K-shift: dequant int4 K -> rope -> requant in place, streams flattened into the rope token dim.
        // dequant reads the pre-shift pool map (kpc_gi_old), requant groups by the rebuilt post-shift group_index.
        GGML_ASSERT(rot == nullptr && "KPC K-shift does not support attn_rot_k");
        GGML_ASSERT(kpc_gi_old != nullptr && "KPC K-shift needs the group_index snapshot input");
        const int32_t ikv = map_layer_ids.at(il);
        ggml_tensor * pk  = layers[ikv].k;            // [C, kv_size, n_stream] packed int4
        ggml_tensor * szc = layers[ikv].k_scalezp;    // [KPC_SZ_GROUP_BYTES(C), ng_max, n_stream]
        ggml_tensor * gic = layers[ikv].group_index;  // [kv_size, n_stream]
        const int64_t C         = pk->ne[0];
        const int64_t kv        = pk->ne[1];
        const int64_t ns        = pk->ne[2];
        const int64_t head_dim  = hparams.n_embd_head_k(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        ggml_tensor * kf32 = ggml_cast(ctx, ggml_kpc_dequant(ctx, pk, szc, kpc_gi_old), GGML_TYPE_F32);  // [C, kv, ns]
        ggml_tensor * k3   = ggml_reshape_3d(ctx, kf32, head_dim, n_head_kv, kv*ns);                     // flatten streams
        ggml_tensor * roped = ggml_rope_ext(ctx, k3,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
        ggml_tensor * roped3 = ggml_reshape_3d(ctx, roped, C, kv, ns);
        return ggml_kpc_requant(ctx, pk, szc, gic, roped3);
    }

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    // KPC: pre-regroup cell->pool snapshot for the dequant pass; I32 [kv_size, n_stream]
    ggml_tensor * kpc_gi_old = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }

    if (kpc_gi_old) {
        kv_self->set_input_kpc_shift(kpc_gi_old);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    if (kpc_enabled()) {
        inp->kpc_gi_old = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, get_size(), n_stream);
        ggml_set_input(inp->kpc_gi_old);
    }

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        // KPC: pass the raw cache tensor; build_rope_shift's KPC branch ignores the n_rot view
        ggml_tensor * k = (layer.k->type == GGML_TYPE_KPC4_1)
            ? layer.k
            : ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il, inp->kpc_gi_old);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // KPC4_1 side tensors aren't transferred by the on-device seq io path; require the host path (flags=0)
    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && kpc_enabled()) {
        throw std::runtime_error("KPC4_1 KV cache does not support on-device sequence state; use the host path (flags=0)");
    }
    // per-sequence KPC4_1 save: K is emitted as dequantized f16 in state_write_data; the shared scalezp pools
    // and group_index aren't serialized, since restore re-quantizes inline (see state_read_data).

    io.write(&n_stream, sizeof(n_stream));

    // KPC slab framing and pool-band layout depend on this; a mismatched reader must refuse
    if (kpc_enabled()) {
        io.write(&n_seq_max, sizeof(n_seq_max));
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr, seq_id);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // see state_write: on-device seq state unsupported for KPC4_1 (side tensors not transferred)
    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && kpc_enabled()) {
        throw std::runtime_error("KPC4_1 KV cache does not support on-device sequence state; use the host path (flags=0)");
    }
    // per-sequence KPC4_1 restore (host path) is supported: state_read_data re-quantizes the f16 K inline.

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    // the stored group_index pool ids and staging slabs were laid out under this; refuse a mismatch
    // instead of silently decoding old cells against wrong pool bands
    if (kpc_enabled()) {
        uint32_t n_seq_max_cur = 0;
        io.read(&n_seq_max_cur, sizeof(n_seq_max_cur));
        if (n_seq_max_cur != n_seq_max) {
            throw std::runtime_error(format(
                "KPC4_1 state was saved with n_seq_max=%u but this context has %u; "
                "restore requires a matching configuration", n_seq_max_cur, n_seq_max));
        }
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        res = res && state_read_data(io, strm, cell_count, sinfo, seq_id);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

// Inline KPC quant math for per-sequence state save/restore, which runs in the I/O layer with no compute graph.
// Mirrors the wire format of ggml-cpu/kpc.cpp via the same IEEE fp16 + nibble layout the live attention path reads.
namespace {
    inline float kpc_inl_deq(uint8_t nib_byte, int c, float s, float z) {
        const int q = (c & 1) ? (nib_byte >> 4) : (nib_byte & 0x0F);
        return q * s + z;
    }
    inline void kpc_inl_pack(uint8_t * row, int64_t c, float v, float s, float z) {
        int qv = (int)((v - z)/s + 0.5f); if (qv < 0) qv = 0; if (qv > 15) qv = 15;
        uint8_t * b = &row[c/2];
        *b = (c & 1) ? ((*b & 0x0F) | (uint8_t)(qv << 4)) : ((*b & 0xF0) | (uint8_t) qv);
    }
    inline void kpc_inl_super_read(const uint8_t * slab, float * ss, float * zmin, float * sz) {
        ggml_fp16_t h;
        memcpy(&h, slab + 0, 2); *ss   = ggml_fp16_to_fp32(h);
        memcpy(&h, slab + 2, 2); *zmin = ggml_fp16_to_fp32(h);
        memcpy(&h, slab + 4, 2); *sz   = ggml_fp16_to_fp32(h);
    }
    inline void kpc_inl_dec1(const uint8_t * slab, int64_t C, int64_t c, float ss, float zmin, float sz, float * scale, float * zp) {
        *scale = slab[6 + c] * ss;
        *zp    = zmin + slab[6 + C + c] * sz;
    }
    // encode per-channel float scale[C] (>=0) and zp[C] into one group's int8 slab (mirror kpc_sz_encode)
    inline void kpc_inl_encode(const float * scale, const float * zp, int64_t C, uint8_t * slab) {
        const int QMAX = 255;
        float smax = 0.0f; for (int64_t c = 0; c < C; ++c) if (scale[c] > smax) smax = scale[c];
        float ss = smax / (float) QMAX; if (ss == 0.0f) ss = 1.0f;
        float zmn = INFINITY, zmx = -INFINITY;
        for (int64_t c = 0; c < C; ++c) { if (zp[c] < zmn) zmn = zp[c]; if (zp[c] > zmx) zmx = zp[c]; }
        float sz = (zmx - zmn) / (float) QMAX; if (sz == 0.0f) sz = 1.0f;
        ggml_fp16_t h;
        h = ggml_fp32_to_fp16(ss);  memcpy(slab + 0, &h, 2);
        h = ggml_fp32_to_fp16(zmn); memcpy(slab + 2, &h, 2);
        h = ggml_fp32_to_fp16(sz);  memcpy(slab + 4, &h, 2);
        uint8_t * qs = slab + 6, * qz = slab + 6 + C;
        for (int64_t c = 0; c < C; ++c) {
            int s = (int)(scale[c] / ss + 0.5f);     if (s < 0) s = 0; if (s > QMAX) s = QMAX;
            int z = (int)((zp[c] - zmn) / sz + 0.5f); if (z < 0) z = 0; if (z > QMAX) z = QMAX;
            qs[c] = (uint8_t) s; qz[c] = (uint8_t) z;
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // per-sequence KPC4_1 save: emit K as dequantized f16 (no scalezp side block); restore re-quantizes inline
    const bool kpc_seq = !layers.empty() && layers[0].k_scalezp && seq_id >= 0;

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        if (kpc_seq) {
            const int64_t C = n_embd_k_gqa;
            ggml_tensor * sz = layer.k_scalezp;
            ggml_tensor * gi = layer.group_index;
            const size_t  krow      = ggml_row_size(k->type, C);     // packed int4 row = C/2 bytes
            const size_t  sz_stream = ggml_nbytes(sz) / sz->ne[2];   // per-stream slab bytes
            const int64_t ng_max    = sz->ne[1];
            const int64_t szgb      = GGML_KPC_SZ_GROUP_BYTES(C);
            // K and scalezp may be device-resident; stage them into host buffers so the per-cell reads below
            // work for both host and offloaded caches. group_index is always host.
            std::vector<uint8_t> k_host, sz_host;
            const uint8_t * kd;
            const uint8_t * szd;
            if (ggml_backend_buffer_is_host(k->buffer)) {
                kd  = (const uint8_t *) k->data;
                szd = (const uint8_t *) sz->data + (size_t) cr.strm * sz_stream;
            } else {
                k_host.resize(ggml_nbytes(k));
                ggml_backend_tensor_get(k, k_host.data(), 0, ggml_nbytes(k));
                kd = k_host.data();
                sz_host.resize(sz_stream);
                ggml_backend_tensor_get(sz, sz_host.data(), (size_t) cr.strm * sz_stream, sz_stream);
                szd = sz_host.data();
            }
            const int32_t * gid = (const int32_t *) gi->data + (size_t) cr.strm * gi->ne[0];

            // cells in token order (cr.data is sorted by cell index = token order for a contiguous seq)
            std::vector<uint32_t> ord;
            for (const auto & range : cr.data) for (uint32_t p = range.first; p < range.second; ++p) ord.push_back(p);
            const int64_t G = GGML_KPC_GROUP;
            const int64_t n_groups = ((int64_t) ord.size() + G - 1) / G;

            // RAW save stores int4-K + scalezp verbatim so per-seq restore is bit-exact; the f16 dequant path
            // (kmode 0) stays only as the automatic fallback on a misaligned restore and for older saved blobs.
            const uint8_t kmode = 1;   // 0 = dequant f16, 1 = raw int4+slab
            io.write(&kmode, sizeof(kmode));

            if (kmode == 1) {
                const uint64_t krow_u = krow, szgb_u = (uint64_t) szgb;
                io.write(&krow_u, sizeof(krow_u));
                io.write(&szgb_u, sizeof(szgb_u));
                for (int64_t lg = 0; lg < n_groups; ++lg) {
                    const int64_t g_lo = lg*G, g_hi = std::min<int64_t>(g_lo + G, (int64_t) ord.size());
                    int32_t pool = gid[ord[g_lo]]; if (pool < 0 || pool >= ng_max) pool = 0;
                    for (int64_t i = g_lo; i < g_hi; ++i) io.write(kd + (size_t) ord[i]*krow, krow);  // int4 rows
                    io.write(szd + (size_t) pool*szgb, szgb);                                          // scalezp slab
                }
            } else {
                const int32_t  f16_type = (int32_t) GGML_TYPE_F16;
                const uint64_t f16_row  = (uint64_t) C * sizeof(ggml_fp16_t);
                io.write(&f16_type, sizeof(f16_type));
                io.write(&f16_row,  sizeof(f16_row));
                std::vector<float> scale(C), zp(C);
                std::vector<ggml_fp16_t> frow(C);
                for (uint32_t p : ord) {
                    const int32_t pool = gid[p];
                    const uint8_t * row = kd + (size_t) p * krow;
                    if (pool >= 0 && pool < ng_max) {
                        const uint8_t * slab = szd + (size_t) pool * szgb;
                        float ss, zmin, szc; kpc_inl_super_read(slab, &ss, &zmin, &szc);
                        for (int64_t c = 0; c < C; ++c) kpc_inl_dec1(slab, C, c, ss, zmin, szc, &scale[c], &zp[c]);
                        for (int64_t c = 0; c < C; ++c) frow[c] = ggml_fp32_to_fp16(kpc_inl_deq(row[c/2], (int) c, scale[c], zp[c]));
                    } else {
                        for (int64_t c = 0; c < C; ++c) frow[c] = ggml_fp32_to_fp16(0.0f);   // unwritten cell
                    }
                    io.write(frow.data(), C * sizeof(ggml_fp16_t));
                }
            }
            // save this sequence's open-group staging (staged_group/mask + the pre-quant f16 residual window) so
            // restore can re-seal the partial last group bit-faithfully. The whole-cache serialize below skips kpc_seq.
            {
                const int slot = (int) seq_id;
                const int64_t GxC = (int64_t) GGML_KPC_GROUP * C;
                int32_t sgv = -1, smv = 0;
                std::vector<ggml_fp16_t> resid((size_t) GxC, ggml_fp32_to_fp16(0.0f));
                if (slot >= 0) {
                    ggml_backend_tensor_get(layer.staged_group, &sgv, (size_t) slot * (1+KPC_GROUP) * sizeof(int32_t), sizeof(int32_t));   // row 0 = group
                    ggml_backend_tensor_get(layer.staged_mask,  &smv, (size_t) slot * sizeof(int32_t), sizeof(int32_t));
                    ggml_backend_tensor_get(layer.k_resid, resid.data(), (size_t) slot * GxC * sizeof(ggml_fp16_t), (size_t) GxC * sizeof(ggml_fp16_t));
                }
                io.write(&sgv, sizeof(sgv));
                io.write(&smv, sizeof(smv));
                io.write(resid.data(), (size_t) GxC * sizeof(ggml_fp16_t));
            }
            continue;
        }

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }

    // KPC4_1: serialize scale/zp + group_index (per stream) and the open-group staging (per sequence).
    // Skipped for per-sequence save (kpc_seq): the K was emitted as f16 and restore re-quantizes inline.
    for (const auto & layer : layers) {
        if (!layer.k_scalezp || kpc_seq) {
            continue;
        }
        ggml_tensor * sz  = layer.k_scalezp;     // [2C, ng_max, n_stream]
        ggml_tensor * rs  = layer.k_resid;       // [C, GROUP, n_seq_max]
        ggml_tensor * gi  = layer.group_index;   // [kv_size, n_stream]
        ggml_tensor * rsl = layer.k_resid_slots; // [GROUP, n_seq_max]
        ggml_tensor * sgp = layer.staged_group;  // [n_seq_max]
        ggml_tensor * smk = layer.staged_mask;   // [n_seq_max]
        const size_t sz_slab  = ggml_nbytes(sz)  / sz->ne[2];   // per-stream
        const size_t gi_slab  = ggml_nbytes(gi)  / gi->ne[1];   // per-stream
        const size_t rs_slab  = ggml_nbytes(rs)  / rs->ne[2];   // per-sequence
        const size_t rsl_slab = ggml_nbytes(rsl) / rsl->ne[1];  // per-sequence
        const size_t st_el    = ggml_type_size(GGML_TYPE_I32);  // one int32 per sequence
        io.write_tensor(sz, cr.strm * sz_slab, sz_slab);
        io.write_tensor(gi, cr.strm * gi_slab, gi_slab);
        // staging indexed by seq_id: per-seq save writes one seq, full save writes all n_seq_max (unified holds
        // all seqs in stream 0, so cr.strm alone would miss them)
        const uint32_t seq_off = seq_id >= 0 ? (uint32_t) seq_id : (n_stream == 1 ? 0 : cr.strm);
        const uint32_t seq_cnt = seq_id >= 0 ? 1                 : (n_stream == 1 ? n_seq_max : 1);
        // prefix the slab count so the reader knows exactly what was written
        io.write(&seq_cnt, sizeof(seq_cnt));
        io.write_tensor(rs,  seq_off * rs_slab,  seq_cnt * rs_slab);
        io.write_tensor(rsl, seq_off * rsl_slab, seq_cnt * rsl_slab);
        io.write_tensor(sgp, seq_off * st_el,    seq_cnt * st_el);
        io.write_tensor(smk, seq_off * st_el,    seq_cnt * st_el);
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo, llama_seq_id dest_seq_id) {
    // per-sequence KPC4_1 restore: K arrives as f16 and is re-quantized inline into int4 + scalezp + group_index
    const bool kpc_seq = !layers.empty() && layers[0].k_scalezp && dest_seq_id >= 0;
    if (!layers.empty() && layers[0].k_scalezp && !kpc_seq) {
        // whole-cache: KPC slabs map back only for contiguous restore from a group boundary at head 0; refuse otherwise
        if (!sinfo.is_contiguous() || sinfo.head() != 0) {
            LLAMA_LOG_ERROR("%s: KPC4_1 state restore requires contiguous placement at head 0\n", __func__);
            return false;
        }
    }

    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        if (kpc_seq) {
            // re-quantize each logical 32-token group into int4 + scalezp + group_index, scattering into the dst
            // cells (sinfo.idxs[0]). Pool ids mirror the kernel's pool_of(sb=dest_seq_id, lg) banding so a later
            // decode continues the sequence correctly. kmode 0 = dequant f16, 1 = raw int4+slab.
            const int64_t C = n_embd_k_gqa;
            ggml_tensor * sz = layer.k_scalezp;
            ggml_tensor * gi = layer.group_index;
            const size_t  krow      = ggml_row_size(k->type, C);
            const size_t  sz_stream = ggml_nbytes(sz) / sz->ne[2];
            const int64_t ng_max    = sz->ne[1];
            const int64_t szgb      = GGML_KPC_SZ_GROUP_BYTES(C);
            const int64_t n_seqps   = (int64_t) n_seq_max / (int64_t) n_stream;
            const int64_t band_size = ng_max / n_seqps;
            const int64_t G         = GGML_KPC_GROUP;
            // K and scalezp may be device-resident; stage into host buffers, re-quantize there, and write the
            // whole stream slabs back to the device at the end. group_index is always host.
            const bool kdev = !ggml_backend_buffer_is_host(k->buffer);
            std::vector<uint8_t> k_host, sz_host;
            uint8_t * kd;
            uint8_t * szd;
            if (kdev) {
                k_host.resize(ggml_nbytes(k));
                ggml_backend_tensor_get(k, k_host.data(), 0, ggml_nbytes(k));
                kd = k_host.data();
                sz_host.resize(sz_stream);
                ggml_backend_tensor_get(sz, sz_host.data(), (size_t) strm * sz_stream, sz_stream);
                szd = sz_host.data();
            } else {
                kd  = (uint8_t *) k->data;
                szd = (uint8_t *) sz->data + (size_t) strm * sz_stream;
            }
            int32_t * gid = (int32_t *) gi->data + (size_t) strm * gi->ne[0];
            const int64_t n_groups = ((int64_t) cell_count + G - 1) / G;

            std::vector<float> scale(C), zp(C), nsc(C), nzp(C);
            // re-quantize a group's f32 K [(g_hi-g_lo)*C] into int4 + scalezp + group_index at the dst cells
            auto requant_group = [&](int64_t lg, int64_t g_lo, int64_t g_hi, const float * kf32) -> bool {
                for (int64_t c = 0; c < C; ++c) {
                    float mn = INFINITY, mx = -INFINITY;
                    for (int64_t i = g_lo; i < g_hi; ++i) { const float v = kf32[(i - g_lo)*C + c]; if (v < mn) mn = v; if (v > mx) mx = v; }
                    float s = (mx - mn) / 15.0f; if (s == 0.0f) s = 1.0f;
                    scale[c] = s; zp[c] = mn;
                }
                const int64_t pool = ((int64_t) dest_seq_id % n_seqps) * band_size + (lg % band_size);
                if (pool < 0 || pool >= ng_max) return false;
                uint8_t * slab = szd + (size_t) pool * szgb;
                kpc_inl_encode(scale.data(), zp.data(), C, slab);
                float ss, zmin, szc; kpc_inl_super_read(slab, &ss, &zmin, &szc);
                for (int64_t c = 0; c < C; ++c) { kpc_inl_dec1(slab, C, c, ss, zmin, szc, &nsc[c], &nzp[c]); if (nsc[c] == 0.0f) nsc[c] = 1.0f; }
                for (int64_t i = g_lo; i < g_hi; ++i) {
                    const uint32_t idx = sinfo.idxs[0][i];
                    uint8_t * row = kd + (size_t) idx * krow;
                    for (int64_t c = 0; c < C; ++c) kpc_inl_pack(row, c, kf32[(i - g_lo)*C + c], nsc[c], nzp[c]);
                    gid[idx] = (int32_t) pool;
                }
                return true;
            };

            uint8_t kmode = 0; io.read(&kmode, sizeof(kmode));
            if (kmode == 1) {
                uint64_t krow_ref = 0, szgb_ref = 0;
                io.read(&krow_ref, sizeof(krow_ref));
                io.read(&szgb_ref, sizeof(szgb_ref));
                if (krow_ref != (uint64_t) krow || szgb_ref != (uint64_t) szgb) {
                    LLAMA_LOG_ERROR("%s: KPC4_1 per-seq RAW record mismatch (layer %d)\n", __func__, il);
                    return false;
                }
                // raw verbatim copy: scatter each int4 row to its dst cell and the per-group slab to the dst pool.
                // a saved sequence has contiguous positions, so every 32-token group maps to one pool and the
                // byte-copy is bit-exact regardless of cell alignment.
                const bool aligned = true;
                std::vector<uint8_t> rows; std::vector<uint8_t> slab(szgb); std::vector<float> kf32;
                for (int64_t lg = 0; lg < n_groups; ++lg) {
                    const int64_t g_lo = lg * G, g_hi = std::min<int64_t>(g_lo + G, cell_count);
                    const int64_t n = g_hi - g_lo;
                    rows.resize((size_t) n * krow);
                    io.read(rows.data(), (size_t) n * krow);
                    io.read(slab.data(), szgb);
                    const int64_t pool = ((int64_t) dest_seq_id % n_seqps) * band_size + (lg % band_size);
                    if (pool < 0 || pool >= ng_max) { LLAMA_LOG_ERROR("%s: KPC4_1 per-seq pool oob (layer %d)\n", __func__, il); return false; }
                    if (aligned) {
                        memcpy(szd + (size_t) pool*szgb, slab.data(), szgb);
                        for (int64_t i = g_lo; i < g_hi; ++i) {
                            const uint32_t idx = sinfo.idxs[0][i];
                            memcpy(kd + (size_t) idx*krow, rows.data() + (size_t)(i - g_lo)*krow, krow);
                            gid[idx] = (int32_t) pool;
                        }
                    } else {
                        float ss, zmin, szc; kpc_inl_super_read(slab.data(), &ss, &zmin, &szc);
                        for (int64_t c = 0; c < C; ++c) kpc_inl_dec1(slab.data(), C, c, ss, zmin, szc, &scale[c], &zp[c]);
                        kf32.resize((size_t) n * C);
                        for (int64_t i = 0; i < n; ++i) { const uint8_t * row = rows.data() + (size_t) i*krow;
                            for (int64_t c = 0; c < C; ++c) kf32[(size_t) i*C + c] = kpc_inl_deq(row[c/2], (int) c, scale[c], zp[c]); }
                        if (!requant_group(lg, g_lo, g_hi, kf32.data())) { LLAMA_LOG_ERROR("%s: KPC4_1 per-seq pool oob (layer %d)\n", __func__, il); return false; }
                    }
                }
            } else {
                int32_t  ftype_ref = 0; io.read(&ftype_ref, sizeof(ftype_ref));
                uint64_t frow_ref  = 0; io.read(&frow_ref,  sizeof(frow_ref));
                if (ftype_ref != (int32_t) GGML_TYPE_F16 || frow_ref != (uint64_t) C * sizeof(ggml_fp16_t)) {
                    LLAMA_LOG_ERROR("%s: KPC4_1 per-seq K record mismatch (layer %d)\n", __func__, il);
                    return false;
                }
                std::vector<ggml_fp16_t> kf16((size_t) cell_count * C);
                io.read(kf16.data(), (size_t) cell_count * C * sizeof(ggml_fp16_t));
                std::vector<float> kf32(C * G);
                for (int64_t lg = 0; lg < n_groups; ++lg) {
                    const int64_t g_lo = lg * G, g_hi = std::min<int64_t>(g_lo + G, cell_count);
                    for (int64_t i = g_lo; i < g_hi; ++i)
                        for (int64_t c = 0; c < C; ++c) kf32[(i - g_lo)*C + c] = ggml_fp16_to_fp32(kf16[(size_t) i*C + c]);
                    if (!requant_group(lg, g_lo, g_hi, kf32.data())) {
                        LLAMA_LOG_ERROR("%s: KPC4_1 per-seq pool out of range (layer %d)\n", __func__, il);
                        return false;
                    }
                }
            }
            if (kdev) {   // scatter only this sequence's cells + scalezp back, leaving other sequences' device cells untouched
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t off = (size_t) sinfo.idxs[0][i] * krow;
                    ggml_backend_tensor_set(k, k_host.data() + off, off, krow);
                }
                ggml_backend_tensor_set(sz, sz_host.data(), (size_t) strm * sz_stream, sz_stream);
            }
            // restore the open-group staging the save emitted (pre-quant f16 residual + staged_group/mask) so a
            // continuing decode re-seals the partial last group bit-faithfully. resid_slots are remapped to the
            // dst cells, since restore may land the sequence at different cells than it was saved from.
            {
                int32_t sgv = -1, smv = 0;
                const int64_t GxC = (int64_t) G * C;
                std::vector<ggml_fp16_t> resid((size_t) GxC, ggml_fp32_to_fp16(0.0f));
                io.read(&sgv, sizeof(sgv));
                io.read(&smv, sizeof(smv));
                io.read(resid.data(), (size_t) GxC * sizeof(ggml_fp16_t));
                const int slot = (int) dest_seq_id;
                if (slot >= 0) {
                    std::vector<int32_t> rslots(G, 0);
                    for (int64_t i = 0; i < cell_count; ++i) {     // map each staged within-group slot to its dst cell
                        const uint32_t idx = sinfo.idxs[0][i];
                        const int32_t  pos = cells.pos_get(idx);
                        if (pos / (int32_t) G != sgv) continue;
                        const int32_t w = pos % (int32_t) G;
                        // the CUDA seal kernel addresses cells globally (stream*kv_size + local), so resid_slots
                        // carries the stream offset on the device cache; the host cache uses stream-local indices.
                        if (smv & (1 << w)) rslots[w] = (int32_t) (kdev ? ((size_t) strm * get_size() + idx) : (size_t) idx);
                    }
                    ggml_backend_tensor_set(layer.k_resid,       resid.data(),  (size_t) slot * GxC * sizeof(ggml_fp16_t), (size_t) GxC * sizeof(ggml_fp16_t));
                    ggml_backend_tensor_set(layer.k_resid_slots, rslots.data(), (size_t) slot * G * sizeof(int32_t),       (size_t) G * sizeof(int32_t));
                    ggml_backend_tensor_set(layer.staged_group,  &sgv,          (size_t) slot * (1+KPC_GROUP) * sizeof(int32_t),           sizeof(int32_t));   // row 0 = group
                    ggml_backend_tensor_set(layer.staged_mask,   &smv,          (size_t) slot * sizeof(int32_t),           sizeof(int32_t));
                }
            }
            continue;
        }

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    // KPC4_1: restore scale/zp + group_index (per stream) and staging (per sequence); mirrors state_write_data.
    // Skipped for per-sequence restore (kpc_seq): scalezp + group_index were rebuilt by the inline requant above.
    for (const auto & layer : layers) {
        if (!layer.k_scalezp || kpc_seq) {
            continue;
        }
        ggml_tensor * sz  = layer.k_scalezp;
        ggml_tensor * rs  = layer.k_resid;
        ggml_tensor * gi  = layer.group_index;
        ggml_tensor * rsl = layer.k_resid_slots;
        ggml_tensor * sgp = layer.staged_group;
        ggml_tensor * smk = layer.staged_mask;
        const size_t sz_slab  = ggml_nbytes(sz)  / sz->ne[2];   // per-stream
        const size_t gi_slab  = ggml_nbytes(gi)  / gi->ne[1];   // per-stream
        const size_t rs_slab  = ggml_nbytes(rs)  / rs->ne[2];   // per-sequence
        const size_t rsl_slab = ggml_nbytes(rsl) / rsl->ne[1];  // per-sequence
        const size_t st_el    = ggml_type_size(GGML_TYPE_I32);  // one int32 per sequence
        io.read_tensor(sz, strm * sz_slab, sz_slab);
        io.read_tensor(gi, strm * gi_slab, gi_slab);
        const uint32_t seq_off = dest_seq_id >= 0 ? (uint32_t) dest_seq_id : (n_stream == 1 ? 0 : strm);
        // read the recorded slab count, place what fits, discard overflow to keep the stream framed
        uint32_t n_stage = 0;
        io.read(&n_stage, sizeof(n_stage));
        const uint32_t cap  = seq_off < n_seq_max ? n_seq_max - seq_off : 0;
        const uint32_t fit  = std::min(n_stage, cap);
        const uint32_t over = n_stage - fit;
        std::vector<char> scratch;
        auto read_block = [&](ggml_tensor * t, size_t slab) {
            if (fit)  io.read_tensor(t, seq_off * slab, fit * slab);
            if (over) { scratch.resize(over * slab); io.read(scratch.data(), over * slab); }
        };
        read_block(rs,  rs_slab);
        read_block(rsl, rsl_slab);
        read_block(sgp, st_el);
        read_block(smk, st_el);
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_k_scalezp(ggml_context * ctx, int32_t il) const {
    return kv->get_k_scalezp(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_k_groupidx(ggml_context * ctx, int32_t il) const {
    return kv->get_k_groupidx(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, ggml_tensor * kpc_seq, ggml_tensor * kpc_pos, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, kpc_seq, kpc_pos, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_kpc_seq(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_kpc_seq(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_kpc_pos(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_kpc_pos(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kpc(ggml_tensor * seq, ggml_tensor * pos, const llama_ubatch * ubatch) const {
    kv->set_input_kpc(seq, pos, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}
