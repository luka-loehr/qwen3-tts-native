#include "talker_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace qwen3_tts {
namespace {

constexpr int kThreads = 256;

__device__ float block_sum(float value) {
    __shared__ float partial[kThreads];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return partial[0];
}

__global__ void rms_norm_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int width,
    float epsilon
) {
    float square_sum = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float value = __bfloat162float(input[index]);
        square_sum = fmaf(value, value, square_sum);
    }
    const float inverse_rms = rsqrtf(block_sum(square_sum) / static_cast<float>(width) + epsilon);
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float value = __bfloat162float(input[index]);
        const float scale = __bfloat162float(weight[index]);
        output[index] = __float2bfloat16(value * inverse_rms * scale);
    }
}

__global__ void head_rms_norm_kernel(
    __nv_bfloat16* values,
    const __nv_bfloat16* weight,
    int head_dimension,
    float epsilon
) {
    __nv_bfloat16* head = values + static_cast<size_t>(blockIdx.x) * head_dimension;
    float square_sum = 0.0f;
    for (int index = threadIdx.x; index < head_dimension; index += blockDim.x) {
        const float value = __bfloat162float(head[index]);
        square_sum = fmaf(value, value, square_sum);
    }
    const float inverse_rms =
        rsqrtf(block_sum(square_sum) / static_cast<float>(head_dimension) + epsilon);
    for (int index = threadIdx.x; index < head_dimension; index += blockDim.x) {
        const float value = __bfloat162float(head[index]);
        const float scale = __bfloat162float(weight[index]);
        head[index] = __float2bfloat16(value * inverse_rms * scale);
    }
}

__global__ void rope_kernel(
    __nv_bfloat16* values,
    int heads,
    int head_dimension,
    int position,
    float theta
) {
    const int half = head_dimension / 2;
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = heads * half;
    if (pair >= total_pairs) {
        return;
    }
    const int head = pair / half;
    const int dimension = pair % half;
    __nv_bfloat16* base = values + static_cast<size_t>(head) * head_dimension;
    const float exponent = static_cast<float>(2 * dimension) / static_cast<float>(head_dimension);
    const float angle = static_cast<float>(position) * powf(theta, -exponent);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float first = __bfloat162float(base[dimension]);
    const float second = __bfloat162float(base[dimension + half]);
    base[dimension] = __float2bfloat16(first * cosine - second * sine);
    base[dimension + half] = __float2bfloat16(second * cosine + first * sine);
}

__global__ void causal_gqa_attention_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    int query_heads,
    int key_value_heads,
    int head_dimension,
    int sequence_length
) {
    extern __shared__ float scores[];
    const int query_head = blockIdx.x;
    const int groups = query_heads / key_value_heads;
    const int key_value_head = query_head / groups;
    const __nv_bfloat16* query_values =
        query + static_cast<size_t>(query_head) * head_dimension;
    const float scale = rsqrtf(static_cast<float>(head_dimension));

    for (int position = 0; position < sequence_length; ++position) {
        const __nv_bfloat16* key = key_cache
            + (static_cast<size_t>(position) * key_value_heads + key_value_head)
                * head_dimension;
        float dot = 0.0f;
        for (int dimension = threadIdx.x; dimension < head_dimension; dimension += blockDim.x) {
            dot = fmaf(
                __bfloat162float(query_values[dimension]),
                __bfloat162float(key[dimension]),
                dot
            );
        }
        scores[position] = block_sum(dot) * scale;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float maximum = -__int_as_float(0x7f800000);
        for (int position = 0; position < sequence_length; ++position) {
            maximum = fmaxf(maximum, scores[position]);
        }
        float denominator = 0.0f;
        for (int position = 0; position < sequence_length; ++position) {
            scores[position] = expf(scores[position] - maximum);
            denominator += scores[position];
        }
        const float inverse = 1.0f / denominator;
        for (int position = 0; position < sequence_length; ++position) {
            scores[position] *= inverse;
        }
    }
    __syncthreads();

    __nv_bfloat16* head_output =
        output + static_cast<size_t>(query_head) * head_dimension;
    for (int dimension = threadIdx.x; dimension < head_dimension; dimension += blockDim.x) {
        float accumulated = 0.0f;
        for (int position = 0; position < sequence_length; ++position) {
            const __nv_bfloat16* value = value_cache
                + (static_cast<size_t>(position) * key_value_heads + key_value_head)
                    * head_dimension;
            accumulated = fmaf(
                scores[position],
                __bfloat162float(value[dimension]),
                accumulated
            );
        }
        head_output[dimension] = __float2bfloat16(accumulated);
    }
}

__global__ void add_in_place_kernel(
    __nv_bfloat16* destination,
    const __nv_bfloat16* source,
    int width
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < width;
         index += blockDim.x * gridDim.x) {
        destination[index] = __float2bfloat16(
            __bfloat162float(destination[index]) + __bfloat162float(source[index])
        );
    }
}

__global__ void copy_add_kernel(
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    __nv_bfloat16* output,
    int width
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < width;
         index += blockDim.x * gridDim.x) {
        output[index] = __float2bfloat16(
            __bfloat162float(left[index]) + __bfloat162float(right[index])
        );
    }
}

__global__ void bias_activation_kernel(
    __nv_bfloat16* values,
    const __nv_bfloat16* bias,
    int width,
    bool silu
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < width;
         index += blockDim.x * gridDim.x) {
        float value = __bfloat162float(values[index]) + __bfloat162float(bias[index]);
        if (silu) {
            value *= 1.0f / (1.0f + expf(-value));
        }
        values[index] = __float2bfloat16(value);
    }
}

__global__ void silu_gate_kernel(
    __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int width
) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < width;
         index += blockDim.x * gridDim.x) {
        const float gate_value = __bfloat162float(gate[index]);
        const float activated = gate_value / (1.0f + expf(-gate_value));
        gate[index] = __float2bfloat16(activated * __bfloat162float(up[index]));
    }
}

int blocks_for(int width) {
    return (width + kThreads - 1) / kThreads;
}

}  // namespace

cudaError_t launch_rms_norm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int width,
    float epsilon,
    cudaStream_t stream
) {
    rms_norm_kernel<<<1, kThreads, 0, stream>>>(input, weight, output, width, epsilon);
    return cudaGetLastError();
}

cudaError_t launch_head_rms_norm(
    __nv_bfloat16* values,
    const __nv_bfloat16* weight,
    int heads,
    int head_dimension,
    float epsilon,
    cudaStream_t stream
) {
    head_rms_norm_kernel<<<heads, kThreads, 0, stream>>>(
        values,
        weight,
        head_dimension,
        epsilon
    );
    return cudaGetLastError();
}

cudaError_t launch_rope(
    __nv_bfloat16* values,
    int heads,
    int head_dimension,
    int position,
    float theta,
    cudaStream_t stream
) {
    const int pairs = heads * head_dimension / 2;
    rope_kernel<<<blocks_for(pairs), kThreads, 0, stream>>>(
        values,
        heads,
        head_dimension,
        position,
        theta
    );
    return cudaGetLastError();
}

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
) {
    causal_gqa_attention_kernel<<<
        query_heads,
        kThreads,
        static_cast<size_t>(sequence_length) * sizeof(float),
        stream
    >>>(
        query,
        key_cache,
        value_cache,
        output,
        query_heads,
        key_value_heads,
        head_dimension,
        sequence_length
    );
    return cudaGetLastError();
}

cudaError_t launch_add_in_place(
    __nv_bfloat16* destination,
    const __nv_bfloat16* source,
    int width,
    cudaStream_t stream
) {
    add_in_place_kernel<<<blocks_for(width), kThreads, 0, stream>>>(
        destination,
        source,
        width
    );
    return cudaGetLastError();
}

cudaError_t launch_copy_add(
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    __nv_bfloat16* output,
    int width,
    cudaStream_t stream
) {
    copy_add_kernel<<<blocks_for(width), kThreads, 0, stream>>>(left, right, output, width);
    return cudaGetLastError();
}

cudaError_t launch_bias_activation(
    __nv_bfloat16* values,
    const __nv_bfloat16* bias,
    int width,
    bool silu,
    cudaStream_t stream
) {
    bias_activation_kernel<<<blocks_for(width), kThreads, 0, stream>>>(
        values,
        bias,
        width,
        silu
    );
    return cudaGetLastError();
}

cudaError_t launch_silu_gate(
    __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int width,
    cudaStream_t stream
) {
    silu_gate_kernel<<<blocks_for(width), kThreads, 0, stream>>>(gate, up, width);
    return cudaGetLastError();
}

cudaError_t launch_fill_zero(
    __nv_bfloat16* values,
    int width,
    cudaStream_t stream
) {
    return cudaMemsetAsync(values, 0, static_cast<size_t>(width) * sizeof(*values), stream);
}

}  // namespace qwen3_tts
