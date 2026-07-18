#include "talker_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace qwen3_tts {
namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kInt8VectorBytes = 16;

/* Per-output-channel symmetric quantization of one BF16 weight row (the
 * contiguous [in_features] column of the OP_T GEMV). One block per output
 * channel: block-reduce the absolute maximum, derive the FP32 scale, then
 * quantize round-to-nearest into int8. */
__global__ void quantize_weight_rows_kernel(
    const __nv_bfloat16* weight,
    int8_t* quantized,
    float* scales,
    int in_features
) {
    __shared__ float partial[kThreads];
    const size_t row_offset = static_cast<size_t>(blockIdx.x) * in_features;
    const __nv_bfloat16* row = weight + row_offset;
    float maximum = 0.0f;
    for (int index = threadIdx.x; index < in_features; index += blockDim.x) {
        maximum = fmaxf(maximum, fabsf(__bfloat162float(row[index])));
    }
    partial[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = partial[0] > 0.0f ? partial[0] / 127.0f : 1.0f;
    const float inverse_scale = 1.0f / scale;
    if (threadIdx.x == 0) {
        scales[blockIdx.x] = scale;
    }
    int8_t* output_row = quantized + row_offset;
    for (int index = threadIdx.x; index < in_features; index += blockDim.x) {
        const float value = __bfloat162float(row[index]) * inverse_scale;
        const float clamped = fminf(fmaxf(value, -127.0f), 127.0f);
        output_row[index] = static_cast<int8_t>(lrintf(clamped));
    }
}

/* Weight-only INT8 dequantizing GEMV/GEMM for the decode phase. One warp per
 * output channel, vectorized 16-byte int8 loads, BF16 activations, FP32
 * accumulation, per-output-channel FP32 scale, BF16 output. Qualified on GB10
 * at 89-97 percent of the achievable DRAM read bandwidth. */
template <int kRows>
__global__ void int8_dequant_gemm_kernel(
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int in_features,
    int out_features
) {
    const int lane = threadIdx.x % kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int global_warp = blockIdx.x * warps_per_block + threadIdx.x / kWarpSize;
    const int total_warps = gridDim.x * warps_per_block;
    const int vectors = in_features / kInt8VectorBytes;

    for (int channel = global_warp; channel < out_features; channel += total_warps) {
        const int8_t* column = weight + static_cast<size_t>(channel) * in_features;
        float accumulators[kRows];
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
            accumulators[row] = 0.0f;
        }
        for (int vector = lane; vector < vectors; vector += kWarpSize) {
            const int4 raw = __ldg(reinterpret_cast<const int4*>(column) + vector);
            const int8_t* weights = reinterpret_cast<const int8_t*>(&raw);
            const int element = vector * kInt8VectorBytes;
#pragma unroll
            for (int row = 0; row < kRows; ++row) {
                const int4* activation_vectors = reinterpret_cast<const int4*>(
                    input + static_cast<size_t>(row) * in_features + element
                );
                const int4 first_raw = __ldg(activation_vectors);
                const int4 second_raw = __ldg(activation_vectors + 1);
                const __nv_bfloat16* first =
                    reinterpret_cast<const __nv_bfloat16*>(&first_raw);
                const __nv_bfloat16* second =
                    reinterpret_cast<const __nv_bfloat16*>(&second_raw);
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sum = fmaf(
                        static_cast<float>(weights[j]),
                        __bfloat162float(first[j]),
                        sum
                    );
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sum = fmaf(
                        static_cast<float>(weights[8 + j]),
                        __bfloat162float(second[j]),
                        sum
                    );
                }
                accumulators[row] += sum;
            }
        }
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
            float value = accumulators[row];
#pragma unroll
            for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
                value += __shfl_down_sync(0xffffffff, value, offset);
            }
            if (lane == 0) {
                output[static_cast<size_t>(row) * out_features + channel] =
                    __float2bfloat16(value * scales[channel]);
            }
        }
    }
}

/* Same dequantizing GEMM but the output row is ADDED into the destination
 * (residual epilogue), removing the separate add pass and its graph node. */
template <int kRows>
__global__ void int8_dequant_gemm_add_kernel(
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int in_features,
    int out_features
) {
    const int lane = threadIdx.x % kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int global_warp = blockIdx.x * warps_per_block + threadIdx.x / kWarpSize;
    const int total_warps = gridDim.x * warps_per_block;
    const int vectors = in_features / kInt8VectorBytes;

    for (int channel = global_warp; channel < out_features; channel += total_warps) {
        const int8_t* column = weight + static_cast<size_t>(channel) * in_features;
        float accumulators[kRows];
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
            accumulators[row] = 0.0f;
        }
        for (int vector = lane; vector < vectors; vector += kWarpSize) {
            const int4 raw = __ldg(reinterpret_cast<const int4*>(column) + vector);
            const int8_t* weights = reinterpret_cast<const int8_t*>(&raw);
            const int element = vector * kInt8VectorBytes;
#pragma unroll
            for (int row = 0; row < kRows; ++row) {
                const int4* activation_vectors = reinterpret_cast<const int4*>(
                    input + static_cast<size_t>(row) * in_features + element
                );
                const int4 first_raw = __ldg(activation_vectors);
                const int4 second_raw = __ldg(activation_vectors + 1);
                const __nv_bfloat16* first =
                    reinterpret_cast<const __nv_bfloat16*>(&first_raw);
                const __nv_bfloat16* second =
                    reinterpret_cast<const __nv_bfloat16*>(&second_raw);
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sum = fmaf(static_cast<float>(weights[j]),
                               __bfloat162float(first[j]), sum);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sum = fmaf(static_cast<float>(weights[8 + j]),
                               __bfloat162float(second[j]), sum);
                }
                accumulators[row] += sum;
            }
        }
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
            float value = accumulators[row];
#pragma unroll
            for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
                value += __shfl_down_sync(0xffffffff, value, offset);
            }
            if (lane == 0) {
                const size_t index =
                    static_cast<size_t>(row) * out_features + channel;
                output[index] = __float2bfloat16(
                    __bfloat162float(output[index]) + value * scales[channel]
                );
            }
        }
    }
}

/* Fused gate/up projection with SiLU epilogue: both weight columns for one
 * intermediate channel are read by the same warp and the kernel writes
 * silu(gate) * up directly, replacing two GEMMs plus the separate SiLU-gate
 * pass. The two matrices are stored as one [2*out, in] quantized tensor with
 * gate rows first. */
template <int kRows>
__global__ void int8_dequant_gate_up_silu_kernel(
    const int8_t* __restrict__ weight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int in_features,
    int out_features
) {
    const int lane = threadIdx.x % kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int global_warp = blockIdx.x * warps_per_block + threadIdx.x / kWarpSize;
    const int total_warps = gridDim.x * warps_per_block;
    const int vectors = in_features / kInt8VectorBytes;

    for (int channel = global_warp; channel < out_features; channel += total_warps) {
        const int8_t* gate_column =
            weight + static_cast<size_t>(channel) * in_features;
        const int8_t* up_column = weight
            + (static_cast<size_t>(out_features) + channel) * in_features;
        float gate_accumulators[kRows];
        float up_accumulators[kRows];
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
            gate_accumulators[row] = 0.0f;
            up_accumulators[row] = 0.0f;
        }
        for (int vector = lane; vector < vectors; vector += kWarpSize) {
            const int4 gate_raw =
                __ldg(reinterpret_cast<const int4*>(gate_column) + vector);
            const int4 up_raw =
                __ldg(reinterpret_cast<const int4*>(up_column) + vector);
            const int8_t* gate_weights =
                reinterpret_cast<const int8_t*>(&gate_raw);
            const int8_t* up_weights = reinterpret_cast<const int8_t*>(&up_raw);
            const int element = vector * kInt8VectorBytes;
#pragma unroll
            for (int row = 0; row < kRows; ++row) {
                const int4* activation_vectors = reinterpret_cast<const int4*>(
                    input + static_cast<size_t>(row) * in_features + element
                );
                const int4 first_raw = __ldg(activation_vectors);
                const int4 second_raw = __ldg(activation_vectors + 1);
                const __nv_bfloat16* first =
                    reinterpret_cast<const __nv_bfloat16*>(&first_raw);
                const __nv_bfloat16* second =
                    reinterpret_cast<const __nv_bfloat16*>(&second_raw);
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float activation = __bfloat162float(first[j]);
                    gate_sum = fmaf(static_cast<float>(gate_weights[j]),
                                    activation, gate_sum);
                    up_sum = fmaf(static_cast<float>(up_weights[j]),
                                  activation, up_sum);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float activation = __bfloat162float(second[j]);
                    gate_sum = fmaf(static_cast<float>(gate_weights[8 + j]),
                                    activation, gate_sum);
                    up_sum = fmaf(static_cast<float>(up_weights[8 + j]),
                                  activation, up_sum);
                }
                gate_accumulators[row] += gate_sum;
                up_accumulators[row] += up_sum;
            }
        }
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
            float gate_value = gate_accumulators[row];
            float up_value = up_accumulators[row];
#pragma unroll
            for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
                gate_value += __shfl_down_sync(0xffffffff, gate_value, offset);
                up_value += __shfl_down_sync(0xffffffff, up_value, offset);
            }
            if (lane == 0) {
                const float gate = gate_value * scales[channel];
                const float up = up_value * scales[out_features + channel];
                const float activated = gate / (1.0f + expf(-gate));
                output[static_cast<size_t>(row) * out_features + channel] =
                    __float2bfloat16(activated * up);
            }
        }
    }
}

int blocks_for_channels(int out_features) {
    const int warps_per_block = kThreads / kWarpSize;
    return (out_features + warps_per_block - 1) / warps_per_block;
}

}  // namespace

cudaError_t launch_quantize_weight_rows(
    const __nv_bfloat16* weight,
    int8_t* quantized,
    float* scales,
    int in_features,
    int out_features,
    cudaStream_t stream
) {
    if (in_features % kInt8VectorBytes != 0) {
        return cudaErrorInvalidValue;
    }
    quantize_weight_rows_kernel<<<out_features, kThreads, 0, stream>>>(
        weight, quantized, scales, in_features
    );
    return cudaGetLastError();
}

cudaError_t launch_int8_gemm_rows(
    const int8_t* weight,
    const float* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int in_features,
    int out_features,
    int rows,
    cudaStream_t stream
) {
    if (in_features % kInt8VectorBytes != 0 || rows < 1 || rows > 8) {
        return cudaErrorInvalidValue;
    }
    const int grid = blocks_for_channels(out_features);
    switch (rows) {
        case 1:
            int8_dequant_gemm_kernel<1><<<grid, kThreads, 0, stream>>>(
                weight, scales, input, output, in_features, out_features);
            break;
        case 2:
            int8_dequant_gemm_kernel<2><<<grid, kThreads, 0, stream>>>(
                weight, scales, input, output, in_features, out_features);
            break;
        case 3:
            int8_dequant_gemm_kernel<3><<<grid, kThreads, 0, stream>>>(
                weight, scales, input, output, in_features, out_features);
            break;
        case 4:
            int8_dequant_gemm_kernel<4><<<grid, kThreads, 0, stream>>>(
                weight, scales, input, output, in_features, out_features);
            break;
        case 5:
            int8_dequant_gemm_kernel<5><<<grid, kThreads, 0, stream>>>(
                weight, scales, input, output, in_features, out_features);
            break;
        case 6:
            int8_dequant_gemm_kernel<6><<<grid, kThreads, 0, stream>>>(
                weight, scales, input, output, in_features, out_features);
            break;
        case 7:
            int8_dequant_gemm_kernel<7><<<grid, kThreads, 0, stream>>>(
                weight, scales, input, output, in_features, out_features);
            break;
        default:
            int8_dequant_gemm_kernel<8><<<grid, kThreads, 0, stream>>>(
                weight, scales, input, output, in_features, out_features);
            break;
    }
    return cudaGetLastError();
}

cudaError_t launch_int8_gemm_rows_add(
    const int8_t* weight,
    const float* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int in_features,
    int out_features,
    int rows,
    cudaStream_t stream
) {
    if (in_features % kInt8VectorBytes != 0 || rows < 1 || rows > 8) {
        return cudaErrorInvalidValue;
    }
    const int grid = blocks_for_channels(out_features);
    switch (rows) {
        case 1: int8_dequant_gemm_add_kernel<1><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 2: int8_dequant_gemm_add_kernel<2><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 3: int8_dequant_gemm_add_kernel<3><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 4: int8_dequant_gemm_add_kernel<4><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 5: int8_dequant_gemm_add_kernel<5><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 6: int8_dequant_gemm_add_kernel<6><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 7: int8_dequant_gemm_add_kernel<7><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        default: int8_dequant_gemm_add_kernel<8><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
    }
    return cudaGetLastError();
}

cudaError_t launch_int8_gate_up_silu_rows(
    const int8_t* weight,
    const float* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int in_features,
    int out_features,
    int rows,
    cudaStream_t stream
) {
    if (in_features % kInt8VectorBytes != 0 || rows < 1 || rows > 8) {
        return cudaErrorInvalidValue;
    }
    const int grid = blocks_for_channels(out_features);
    switch (rows) {
        case 1: int8_dequant_gate_up_silu_kernel<1><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 2: int8_dequant_gate_up_silu_kernel<2><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 3: int8_dequant_gate_up_silu_kernel<3><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 4: int8_dequant_gate_up_silu_kernel<4><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 5: int8_dequant_gate_up_silu_kernel<5><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 6: int8_dequant_gate_up_silu_kernel<6><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        case 7: int8_dequant_gate_up_silu_kernel<7><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
        default: int8_dequant_gate_up_silu_kernel<8><<<grid, kThreads, 0, stream>>>(weight, scales, input, output, in_features, out_features); break;
    }
    return cudaGetLastError();
}

}  // namespace qwen3_tts
