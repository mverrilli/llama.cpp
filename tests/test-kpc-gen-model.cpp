// generate a tiny head_dim=128 llama model and save it to argv[1]; kpc4_1 requires
// n_embd_head_k % 32 == 0, which the stock tiny test models do not satisfy.
// weights are seeded-random; the kpc tests only compare logits across runs

#include "llama.h"
#include "gguf.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) tmp[i] = dis(gen);
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) tmp[i] = ggml_fp32_to_fp16(dis(gen));
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected tensor type");
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    gguf_context * gguf = gguf_init_empty();
    {
        llama_model_saver ms(LLM_ARCH_LLAMA, gguf);
        const uint32_t n_ctx = 128, n_vocab = 128, n_embd = 256, n_head = 2, n_ff = 384, n_layer = 2;
        ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,         llm_arch_name(LLM_ARCH_LLAMA));
        ms.add_kv(LLM_KV_VOCAB_SIZE,                   n_vocab);
        ms.add_kv(LLM_KV_CONTEXT_LENGTH,               n_ctx);
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH,             n_embd);
        ms.add_kv(LLM_KV_BLOCK_COUNT,                  n_layer);
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,          n_ff);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,         n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,      n_head);          // n_embd_head_k = 256/2 = 128
        ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  1.0e-5f);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,         n_embd / n_head); // 128
        ms.add_kv(LLM_KV_TOKENIZER_MODEL,              "no_vocab");
    }

    llama_model_params mparams = llama_model_default_params();
    size_t seed = 1234;
    llama_model * model = llama_model_init_from_user(gguf, set_tensor_data, &seed, mparams);
    gguf_free(gguf);
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to build model\n", __func__);
        return 1;
    }

    llama_model_saver out(model);
    out.add_kv_from_model();
    out.add_tensors_from_model();
    out.save(std::string(argv[1]));

    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "%s: wrote head_dim=128 llama model to %s\n", __func__, argv[1]);
    return 0;
}
