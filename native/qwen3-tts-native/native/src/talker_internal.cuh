#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace qwen3_tts {

cudaError_t launch_rms_norm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
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

cudaError_t launch_rope(
    __nv_bfloat16* values,
    int heads,
    int head_dimension,
    int position,
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

}  // namespace qwen3_tts
