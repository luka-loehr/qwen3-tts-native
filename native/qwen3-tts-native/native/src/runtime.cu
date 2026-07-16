#include "qwen3_tts_native.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarmupIterations = 100;

void write_error(char* destination, size_t capacity, const char* message) {
    if (destination == nullptr || capacity == 0) {
        return;
    }
    std::snprintf(destination, capacity, "%s", message);
}

int32_t cuda_failure(
    cudaError_t status,
    const char* operation,
    char* error,
    size_t error_capacity
) {
    char message[512];
    std::snprintf(
        message,
        sizeof(message),
        "%s failed: %s",
        operation,
        cudaGetErrorString(status)
    );
    write_error(error, error_capacity, message);
    return static_cast<int32_t>(status);
}

__global__ void bf16_argmax_kernel(
    const __nv_bfloat16* logits,
    int vocabulary_size,
    int* selected_token
) {
    float local_value = -__int_as_float(0x7f800000);
    int local_index = -1;

    for (int index = threadIdx.x; index < vocabulary_size; index += blockDim.x) {
        const float value = __bfloat162float(logits[index]);
        if (value > local_value || (value == local_value && index < local_index)) {
            local_value = value;
            local_index = index;
        }
    }

    __shared__ float values[kThreads];
    __shared__ int indices[kThreads];
    values[threadIdx.x] = local_value;
    indices[threadIdx.x] = local_index;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float candidate_value = values[threadIdx.x + stride];
            const int candidate_index = indices[threadIdx.x + stride];
            if (candidate_value > values[threadIdx.x] ||
                (candidate_value == values[threadIdx.x] &&
                 candidate_index < indices[threadIdx.x])) {
                values[threadIdx.x] = candidate_value;
                indices[threadIdx.x] = candidate_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *selected_token = indices[0];
    }
}

}  // namespace

extern "C" QWEN3_TTS_API int32_t qwen3_tts_probe_device(
    int32_t device_index,
    Qwen3TtsDeviceInfo* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "output device-info pointer is null");
        return -1;
    }

    cudaError_t status = cudaSetDevice(device_index);
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaSetDevice", error, error_capacity);
    }

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device_index);
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaGetDeviceProperties", error, error_capacity);
    }

    size_t free_memory = 0;
    size_t total_memory = 0;
    status = cudaMemGetInfo(&free_memory, &total_memory);
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaMemGetInfo", error, error_capacity);
    }

    std::memset(output, 0, sizeof(*output));
    output->device_index = device_index;
    output->compute_major = properties.major;
    output->compute_minor = properties.minor;
    output->total_global_memory_bytes = properties.totalGlobalMem;
    output->runtime_free_memory_bytes = free_memory;
    output->runtime_total_memory_bytes = total_memory;
    std::snprintf(output->device_name, sizeof(output->device_name), "%s", properties.name);
    write_error(error, error_capacity, "");
    return 0;
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_benchmark_bf16_argmax(
    int32_t device_index,
    int32_t vocabulary_size,
    int32_t iterations,
    Qwen3TtsArgmaxBenchmark* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "output benchmark pointer is null");
        return -1;
    }
    if (vocabulary_size <= 0 || vocabulary_size > 65'536) {
        write_error(error, error_capacity, "vocabulary size must be in [1, 65536]");
        return -2;
    }
    if (iterations <= 0) {
        write_error(error, error_capacity, "iterations must be positive");
        return -3;
    }

    cudaError_t status = cudaSetDevice(device_index);
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaSetDevice", error, error_capacity);
    }

    const int expected_token = std::min(1'729, vocabulary_size - 1);
    std::vector<__nv_bfloat16> host_logits(static_cast<size_t>(vocabulary_size));
    for (int index = 0; index < vocabulary_size; ++index) {
        const float value = std::sin(static_cast<float>(index) * 0.017f) * 2.0f;
        host_logits[static_cast<size_t>(index)] = __float2bfloat16(value);
    }
    host_logits[static_cast<size_t>(expected_token)] = __float2bfloat16(19.0f);

    __nv_bfloat16* device_logits = nullptr;
    int* device_token = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    status = cudaMalloc(
        reinterpret_cast<void**>(&device_logits),
        host_logits.size() * sizeof(__nv_bfloat16)
    );
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaMalloc(logits)", error, error_capacity);
    }
    status = cudaMalloc(reinterpret_cast<void**>(&device_token), sizeof(int));
    if (status != cudaSuccess) {
        cudaFree(device_logits);
        return cuda_failure(status, "cudaMalloc(token)", error, error_capacity);
    }
    status = cudaMemcpy(
        device_logits,
        host_logits.data(),
        host_logits.size() * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        cudaFree(device_token);
        cudaFree(device_logits);
        return cuda_failure(status, "cudaMemcpy(logits)", error, error_capacity);
    }
    if ((status = cudaEventCreate(&start)) != cudaSuccess ||
        (status = cudaEventCreate(&stop)) != cudaSuccess) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        cudaFree(device_token);
        cudaFree(device_logits);
        return cuda_failure(status, "cudaEventCreate", error, error_capacity);
    }

    cudaEventRecord(start);
    bf16_argmax_kernel<<<1, kThreads>>>(device_logits, vocabulary_size, device_token);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cold_milliseconds = 0.0f;
    cudaEventElapsedTime(&cold_milliseconds, start, stop);

    for (int index = 0; index < kWarmupIterations; ++index) {
        bf16_argmax_kernel<<<1, kThreads>>>(device_logits, vocabulary_size, device_token);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int index = 0; index < iterations; ++index) {
        bf16_argmax_kernel<<<1, kThreads>>>(device_logits, vocabulary_size, device_token);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float measured_milliseconds = 0.0f;
    cudaEventElapsedTime(&measured_milliseconds, start, stop);
    int selected_token = -1;
    status = cudaMemcpy(
        &selected_token,
        device_token,
        sizeof(selected_token),
        cudaMemcpyDeviceToHost
    );
    const cudaError_t launch_status = cudaGetLastError();

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_token);
    cudaFree(device_logits);

    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaMemcpy(token)", error, error_capacity);
    }
    if (launch_status != cudaSuccess) {
        return cuda_failure(launch_status, "bf16_argmax_kernel", error, error_capacity);
    }
    if (selected_token != expected_token) {
        write_error(error, error_capacity, "argmax kernel selected the wrong token");
        return -4;
    }

    std::memset(output, 0, sizeof(*output));
    output->vocabulary_size = vocabulary_size;
    output->iterations = iterations;
    output->selected_token = selected_token;
    output->expected_token = expected_token;
    output->cold_launch_microseconds = cold_milliseconds * 1'000.0f;
    output->mean_launch_microseconds =
        measured_milliseconds * 1'000.0f / static_cast<float>(iterations);
    write_error(error, error_capacity, "");
    return 0;
}
