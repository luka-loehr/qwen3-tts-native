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
    const __nv_bfloat16* row_input = input + static_cast<size_t>(blockIdx.x) * width;
    __nv_bfloat16* row_output = output + static_cast<size_t>(blockIdx.x) * width;
    float square_sum = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float value = __bfloat162float(row_input[index]);
        square_sum = fmaf(value, value, square_sum);
    }
    const float inverse_rms = rsqrtf(block_sum(square_sum) / static_cast<float>(width) + epsilon);
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        const float value = __bfloat162float(row_input[index]);
        const float scale = __bfloat162float(weight[index]);
        const __nv_bfloat16 normalized = __float2bfloat16(value * inverse_rms);
        row_output[index] = __float2bfloat16(__bfloat162float(normalized) * scale);
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
        const __nv_bfloat16 normalized = __float2bfloat16(value * inverse_rms);
        head[index] = __float2bfloat16(__bfloat162float(normalized) * scale);
    }
}

__global__ void rope_kernel(
    __nv_bfloat16* values,
    int rows,
    int heads,
    int head_dimension,
    int first_position,
    float theta
) {
    const int half = head_dimension / 2;
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_pairs = heads * half;
    const int total_pairs = rows * row_pairs;
    if (pair >= total_pairs) {
        return;
    }
    const int row = pair / row_pairs;
    const int row_pair = pair % row_pairs;
    const int head = row_pair / half;
    const int dimension = row_pair % half;
    const int position = first_position + row;
    __nv_bfloat16* base = values
        + (static_cast<size_t>(row) * heads + head) * head_dimension;
    const float exponent = static_cast<float>(2 * dimension) / static_cast<float>(head_dimension);
    const float angle = static_cast<float>(position) * powf(theta, -exponent);
    const __nv_bfloat16 cosine = __float2bfloat16(cosf(angle));
    const __nv_bfloat16 sine = __float2bfloat16(sinf(angle));
    const float first = __bfloat162float(base[dimension]);
    const float second = __bfloat162float(base[dimension + half]);
    const __nv_bfloat16 first_cosine =
        __float2bfloat16(first * __bfloat162float(cosine));
    const __nv_bfloat16 second_sine =
        __float2bfloat16(second * __bfloat162float(sine));
    const __nv_bfloat16 second_cosine =
        __float2bfloat16(second * __bfloat162float(cosine));
    const __nv_bfloat16 first_sine =
        __float2bfloat16(first * __bfloat162float(sine));
    base[dimension] = __float2bfloat16(
        __bfloat162float(first_cosine) - __bfloat162float(second_sine)
    );
    base[dimension + half] = __float2bfloat16(
        __bfloat162float(second_cosine) + __bfloat162float(first_sine)
    );
}

__global__ void prefill_causal_gqa_attention_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    __nv_bfloat16* output,
    int rows,
    int query_heads,
    int key_value_heads,
    int head_dimension
) {
    extern __shared__ float scores[];
    const int query_row = blockIdx.x / query_heads;
    const int query_head = blockIdx.x % query_heads;
    if (query_row >= rows) {
        return;
    }
    const int groups = query_heads / key_value_heads;
    const int key_value_head = query_head / groups;
    const __nv_bfloat16* query_values = query
        + (static_cast<size_t>(query_row) * query_heads + query_head) * head_dimension;
    const float scale = rsqrtf(static_cast<float>(head_dimension));
    const int sequence_length = query_row + 1;

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
        const __nv_bfloat16 dot_product = __float2bfloat16(block_sum(dot));
        scores[position] = __bfloat162float(__float2bfloat16(
            __bfloat162float(dot_product) * scale
        ));
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
            scores[position] = __bfloat162float(__float2bfloat16(scores[position] * inverse));
        }
    }
    __syncthreads();

    __nv_bfloat16* head_output = output
        + (static_cast<size_t>(query_row) * query_heads + query_head) * head_dimension;
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

__global__ void pack_query_heads_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int rows,
    int heads,
    int head_dimension
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * heads * head_dimension;
    if (index >= total) {
        return;
    }
    const int dimension = index % head_dimension;
    const int head = (index / head_dimension) % heads;
    const int row = index / (heads * head_dimension);
    output[(static_cast<size_t>(head) * rows + row) * head_dimension + dimension]
        = input[index];
}

__global__ void pack_repeated_key_value_heads_kernel(
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    __nv_bfloat16* packed_key,
    __nv_bfloat16* packed_value,
    int rows,
    int query_heads,
    int key_value_heads,
    int head_dimension
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * query_heads * head_dimension;
    if (index >= total) {
        return;
    }
    const int dimension = index % head_dimension;
    const int query_head = (index / head_dimension) % query_heads;
    const int row = index / (query_heads * head_dimension);
    const int groups = query_heads / key_value_heads;
    const int key_value_head = query_head / groups;
    const size_t source = (static_cast<size_t>(row) * key_value_heads + key_value_head)
        * head_dimension + dimension;
    const size_t destination = (static_cast<size_t>(query_head) * rows + row)
        * head_dimension + dimension;
    packed_key[destination] = key[source];
    packed_value[destination] = value[source];
}

__global__ void causal_softmax_kernel(
    __nv_bfloat16* scores,
    int rows,
    int head_dimension
) {
    const int query = blockIdx.x % rows;
    __nv_bfloat16* row = scores + static_cast<size_t>(blockIdx.x) * rows;
    const float scale = rsqrtf(static_cast<float>(head_dimension));
    float maximum = -__int_as_float(0x7f800000);
    for (int key = 0; key <= query; ++key) {
        const __nv_bfloat16 scaled = __float2bfloat16(__bfloat162float(row[key]) * scale);
        row[key] = scaled;
        maximum = fmaxf(maximum, __bfloat162float(scaled));
    }
    float denominator = 0.0f;
    for (int key = 0; key <= query; ++key) {
        denominator += expf(__bfloat162float(row[key]) - maximum);
    }
    const float inverse = 1.0f / denominator;
    for (int key = 0; key <= query; ++key) {
        row[key] = __float2bfloat16(
            expf(__bfloat162float(row[key]) - maximum) * inverse
        );
    }
    for (int key = query + 1; key < rows; ++key) {
        row[key] = __float2bfloat16(0.0f);
    }
}

__global__ void unpack_heads_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int rows,
    int heads,
    int head_dimension
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * heads * head_dimension;
    if (index >= total) {
        return;
    }
    const int dimension = index % head_dimension;
    const int head = (index / head_dimension) % heads;
    const int row = index / (heads * head_dimension);
    output[index] = input[(static_cast<size_t>(head) * rows + row) * head_dimension + dimension];
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
        const __nv_bfloat16 dot_product = __float2bfloat16(block_sum(dot));
        scores[position] = __bfloat162float(__float2bfloat16(
            __bfloat162float(dot_product) * scale
        ));
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
            scores[position] = __bfloat162float(__float2bfloat16(scores[position] * inverse));
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
        const __nv_bfloat16 activated = __float2bfloat16(
            gate_value / (1.0f + expf(-gate_value))
        );
        gate[index] = __float2bfloat16(
            __bfloat162float(activated) * __bfloat162float(up[index])
        );
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
    return launch_rms_norm_rows(input, weight, output, 1, width, epsilon, stream);
}

cudaError_t launch_rms_norm_rows(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int width,
    float epsilon,
    cudaStream_t stream
) {
    rms_norm_kernel<<<rows, kThreads, 0, stream>>>(input, weight, output, width, epsilon);
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
    return launch_head_rms_norm_rows(
        values,
        weight,
        1,
        heads,
        head_dimension,
        epsilon,
        stream
    );
}

cudaError_t launch_head_rms_norm_rows(
    __nv_bfloat16* values,
    const __nv_bfloat16* weight,
    int rows,
    int heads,
    int head_dimension,
    float epsilon,
    cudaStream_t stream
) {
    head_rms_norm_kernel<<<rows * heads, kThreads, 0, stream>>>(
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
    return launch_rope_rows(
        values,
        1,
        heads,
        head_dimension,
        position,
        theta,
        stream
    );
}

cudaError_t launch_rope_rows(
    __nv_bfloat16* values,
    int rows,
    int heads,
    int head_dimension,
    int first_position,
    float theta,
    cudaStream_t stream
) {
    const int pairs = rows * heads * head_dimension / 2;
    rope_kernel<<<blocks_for(pairs), kThreads, 0, stream>>>(
        values,
        rows,
        heads,
        head_dimension,
        first_position,
        theta
    );
    return cudaGetLastError();
}

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
) {
    prefill_causal_gqa_attention_kernel<<<
        rows * query_heads,
        kThreads,
        static_cast<size_t>(rows) * sizeof(float),
        stream
    >>>(
        query,
        key_cache,
        value_cache,
        output,
        rows,
        query_heads,
        key_value_heads,
        head_dimension
    );
    return cudaGetLastError();
}

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
) {
    const int elements = rows * query_heads * head_dimension;
    pack_query_heads_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
        query,
        packed_query,
        rows,
        query_heads,
        head_dimension
    );
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    pack_repeated_key_value_heads_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
        key,
        value,
        packed_key,
        packed_value,
        rows,
        query_heads,
        key_value_heads,
        head_dimension
    );
    return cudaGetLastError();
}

cudaError_t launch_causal_softmax(
    __nv_bfloat16* scores,
    int rows,
    int heads,
    int head_dimension,
    cudaStream_t stream
) {
    causal_softmax_kernel<<<rows * heads, 1, 0, stream>>>(scores, rows, head_dimension);
    return cudaGetLastError();
}

cudaError_t launch_unpack_heads(
    const __nv_bfloat16* packed,
    __nv_bfloat16* output,
    int rows,
    int heads,
    int head_dimension,
    cudaStream_t stream
) {
    const int elements = rows * heads * head_dimension;
    unpack_heads_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
        packed,
        output,
        rows,
        heads,
        head_dimension
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
