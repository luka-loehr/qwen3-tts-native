#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace qwen3_tts {

cudaError_t launch_rms_norm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int width,
    float epsilon,
    cudaStream_t stream
);

cudaError_t launch_rms_norm_rows(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int width,
    float epsilon,
    cudaStream_t stream
);

cudaError_t launch_head_rms_norm(
    __nv_bfloat16* values,
    const __nv_bfloat16* weight,
    int heads,
    int head_dimension,
    float epsilon,
    cudaStream_t stream
);

cudaError_t launch_head_rms_norm_rows(
    __nv_bfloat16* values,
    const __nv_bfloat16* weight,
    int rows,
    int heads,
    int head_dimension,
    float epsilon,
    cudaStream_t stream
);

cudaError_t launch_rope(
    __nv_bfloat16* values,
    int heads,
    int head_dimension,
    int position,
    float theta,
    cudaStream_t stream
);

cudaError_t launch_rope_rows(
    __nv_bfloat16* values,
    int rows,
    int heads,
    int head_dimension,
    int first_position,
    float theta,
    cudaStream_t stream
);

cudaError_t launch_causal_gqa_attention(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    int query_heads,
    int key_value_heads,
    int head_dimension,
    int sequence_length,
    cudaStream_t stream
);

cudaError_t launch_prefill_causal_gqa_attention(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    int rows,
    int query_heads,
    int key_value_heads,
    int head_dimension,
    cudaStream_t stream
);

cudaError_t launch_pack_gqa_heads(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    __nv_bfloat16* packed_query,
    __nv_bfloat16* packed_key,
    __nv_bfloat16* packed_value,
    int rows,
    int query_heads,
    int key_value_heads,
    int head_dimension,
    cudaStream_t stream
);

cudaError_t launch_causal_softmax(
    __nv_bfloat16* scores,
    int rows,
    int heads,
    int head_dimension,
    cudaStream_t stream
);

cudaError_t launch_unpack_heads(
    const __nv_bfloat16* packed,
    __nv_bfloat16* output,
    int rows,
    int heads,
    int head_dimension,
    cudaStream_t stream
);

cudaError_t launch_add_in_place(
    __nv_bfloat16* destination,
    const __nv_bfloat16* source,
    int width,
    cudaStream_t stream
);

cudaError_t launch_copy_add(
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    __nv_bfloat16* output,
    int width,
    cudaStream_t stream
);

cudaError_t launch_bias_activation(
    __nv_bfloat16* values,
    const __nv_bfloat16* bias,
    int width,
    bool silu,
    cudaStream_t stream
);

cudaError_t launch_silu_gate(
    __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int width,
    cudaStream_t stream
);

cudaError_t launch_fill_zero(
    __nv_bfloat16* values,
    int width,
    cudaStream_t stream
);

cudaError_t launch_sample_logits(
    const __nv_bfloat16* logits,
    int vocabulary,
    bool suppress_talker_reserved,
    int codec_eos_token,
    const int* semantic_history,
    int semantic_history_count,
    int do_sample,
    int top_k,
    float top_p,
    float temperature,
    float repetition_penalty,
    uint64_t* random_state,
    int* selected_token,
    cudaStream_t stream
);

cudaError_t launch_store_token(
    int* tokens,
    int index,
    int value,
    cudaStream_t stream
);

cudaError_t launch_store_sampled_token(
    int* tokens,
    int index,
    const int* sampled_token,
    cudaStream_t stream
);

cudaError_t launch_store_sampled_token_at(
    int* tokens,
    const int* index,
    const int* sampled_token,
    cudaStream_t stream
);

cudaError_t launch_sample_logits_at(
    const __nv_bfloat16* logits,
    int vocabulary,
    bool suppress_talker_reserved,
    int codec_eos_token,
    const int* semantic_history,
    const int* semantic_history_count,
    int do_sample,
    int top_k,
    float top_p,
    float temperature,
    float repetition_penalty,
    uint64_t* random_state,
    int* selected_token,
    cudaStream_t stream
);

cudaError_t launch_pack_frame_codes(
    const int* tokens,
    uint16_t* codes,
    cudaStream_t stream
);

cudaError_t launch_gather_embedding(
    const __nv_bfloat16* table,
    int vocabulary,
    int width,
    const int* token,
    __nv_bfloat16* output,
    cudaStream_t stream
);

cudaError_t launch_add_embedding(
    __nv_bfloat16* destination,
    const __nv_bfloat16* table,
    int vocabulary,
    int width,
    const int* token,
    cudaStream_t stream
);

cudaError_t launch_rope_rows_at(
    __nv_bfloat16* values,
    int rows,
    int heads,
    int head_dimension,
    const int* positions,
    float theta,
    cudaStream_t stream
);

cudaError_t launch_batch_causal_gqa_attention(
    const __nv_bfloat16* query,
    __nv_bfloat16* const* key_bases,
    __nv_bfloat16* const* value_bases,
    const int* positions,
    __nv_bfloat16* output,
    int rows,
    int query_heads,
    int key_value_heads,
    int head_dimension,
    int max_sequence_length,
    cudaStream_t stream
);

cudaError_t launch_kv_scatter_rows(
    const __nv_bfloat16* rows_data,
    __nv_bfloat16* const* cache_bases,
    const int* positions,
    int rows,
    int width,
    cudaStream_t stream
);

cudaError_t launch_bias_activation_rows(
    __nv_bfloat16* values,
    const __nv_bfloat16* bias,
    int rows,
    int width,
    bool silu,
    cudaStream_t stream
);

cudaError_t launch_gather_embedding_rows(
    const __nv_bfloat16* table,
    int vocabulary,
    int width,
    const int* const* tokens,
    int token_offset,
    __nv_bfloat16* output,
    int rows,
    cudaStream_t stream
);

cudaError_t launch_add_embedding_rows(
    const __nv_bfloat16* table,
    int vocabulary,
    int width,
    const int* const* tokens,
    int token_offset,
    __nv_bfloat16* output,
    int rows,
    cudaStream_t stream
);

cudaError_t launch_quantize_weight_rows(
    const __nv_bfloat16* weight,
    int8_t* quantized,
    float* scales,
    int in_features,
    int out_features,
    cudaStream_t stream
);

cudaError_t launch_int8_gemm_rows(
    const int8_t* weight,
    const float* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int in_features,
    int out_features,
    int rows,
    cudaStream_t stream
);

}  // namespace qwen3_tts
