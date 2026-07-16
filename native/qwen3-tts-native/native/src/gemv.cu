#include "qwen3_tts_native.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <limits>

namespace {

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

int32_t cublas_failure(
    cublasStatus_t status,
    const char* operation,
    char* error,
    size_t error_capacity
) {
    char message[512];
    std::snprintf(
        message,
        sizeof(message),
        "%s failed with cuBLAS status %d",
        operation,
        static_cast<int>(status)
    );
    write_error(error, error_capacity, message);
    return -1000 - static_cast<int32_t>(status);
}

void release(
    cublasHandle_t handle,
    cudaEvent_t start,
    cudaEvent_t stop,
    __nv_bfloat16* weights,
    __nv_bfloat16* input,
    __nv_bfloat16* output
) {
    if (stop != nullptr) {
        cudaEventDestroy(stop);
    }
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (handle != nullptr) {
        cublasDestroy(handle);
    }
    if (output != nullptr) {
        cudaFree(output);
    }
    if (input != nullptr) {
        cudaFree(input);
    }
    if (weights != nullptr) {
        cudaFree(weights);
    }
}

}  // namespace

extern "C" QWEN3_TTS_API int32_t qwen3_tts_benchmark_bf16_gemv(
    int32_t device_index,
    int32_t input_features,
    int32_t output_features,
    int32_t iterations,
    Qwen3TtsGemvBenchmark* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "output benchmark pointer is null");
        return -1;
    }
    if (input_features <= 0 || input_features > 65'536 ||
        output_features <= 0 || output_features > 65'536) {
        write_error(error, error_capacity, "feature sizes must be in [1, 65536]");
        return -2;
    }
    if (iterations <= 0) {
        write_error(error, error_capacity, "iterations must be positive");
        return -3;
    }

    const uint64_t weight_elements =
        static_cast<uint64_t>(input_features) * static_cast<uint64_t>(output_features);
    if (weight_elements > std::numeric_limits<size_t>::max() / sizeof(__nv_bfloat16)) {
        write_error(error, error_capacity, "weight allocation size overflow");
        return -4;
    }
    const size_t weight_bytes =
        static_cast<size_t>(weight_elements) * sizeof(__nv_bfloat16);
    const size_t input_bytes =
        static_cast<size_t>(input_features) * sizeof(__nv_bfloat16);
    const size_t output_bytes =
        static_cast<size_t>(output_features) * sizeof(__nv_bfloat16);

    cudaError_t cuda_status = cudaSetDevice(device_index);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(cuda_status, "cudaSetDevice", error, error_capacity);
    }

    __nv_bfloat16* weights = nullptr;
    __nv_bfloat16* input = nullptr;
    __nv_bfloat16* device_output = nullptr;
    cublasHandle_t handle = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cuda_status = cudaMalloc(reinterpret_cast<void**>(&weights), weight_bytes);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(cuda_status, "cudaMalloc(weights)", error, error_capacity);
    }
    cuda_status = cudaMalloc(reinterpret_cast<void**>(&input), input_bytes);
    if (cuda_status != cudaSuccess) {
        release(handle, start, stop, weights, input, device_output);
        return cuda_failure(cuda_status, "cudaMalloc(input)", error, error_capacity);
    }
    cuda_status = cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes);
    if (cuda_status != cudaSuccess) {
        release(handle, start, stop, weights, input, device_output);
        return cuda_failure(cuda_status, "cudaMalloc(output)", error, error_capacity);
    }

    cudaMemset(weights, 0, weight_bytes);
    cudaMemset(input, 0, input_bytes);
    cudaMemset(device_output, 0xff, output_bytes);

    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        release(handle, start, stop, weights, input, device_output);
        return cublas_failure(cublas_status, "cublasCreate", error, error_capacity);
    }
    cublas_status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        release(handle, start, stop, weights, input, device_output);
        return cublas_failure(cublas_status, "cublasSetMathMode", error, error_capacity);
    }

    cuda_status = cudaEventCreate(&start);
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaEventCreate(&stop);
    }
    if (cuda_status != cudaSuccess) {
        release(handle, start, stop, weights, input, device_output);
        return cuda_failure(cuda_status, "cudaEventCreate", error, error_capacity);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch = [&]() {
        return cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            output_features,
            1,
            input_features,
            &alpha,
            weights,
            CUDA_R_16BF,
            input_features,
            input,
            CUDA_R_16BF,
            input_features,
            &beta,
            device_output,
            CUDA_R_16BF,
            output_features,
            CUBLAS_COMPUTE_32F_FAST_16BF,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    };

    cudaEventRecord(start);
    cublas_status = launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        release(handle, start, stop, weights, input, device_output);
        return cublas_failure(cublas_status, "cublasGemmEx(cold)", error, error_capacity);
    }
    float cold_milliseconds = 0.0f;
    cudaEventElapsedTime(&cold_milliseconds, start, stop);

    for (int index = 0; index < kWarmupIterations; ++index) {
        cublas_status = launch();
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            release(handle, start, stop, weights, input, device_output);
            return cublas_failure(cublas_status, "cublasGemmEx(warmup)", error, error_capacity);
        }
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int index = 0; index < iterations; ++index) {
        cublas_status = launch();
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            release(handle, start, stop, weights, input, device_output);
            return cublas_failure(cublas_status, "cublasGemmEx(measured)", error, error_capacity);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float measured_milliseconds = 0.0f;
    cudaEventElapsedTime(&measured_milliseconds, start, stop);
    __nv_bfloat16 first_output{};
    cuda_status = cudaMemcpy(
        &first_output,
        device_output,
        sizeof(first_output),
        cudaMemcpyDeviceToHost
    );
    const float first_output_value = __bfloat162float(first_output);
    release(handle, start, stop, weights, input, device_output);

    if (cuda_status != cudaSuccess) {
        return cuda_failure(cuda_status, "cudaMemcpy(output)", error, error_capacity);
    }
    if (first_output_value != 0.0f) {
        write_error(error, error_capacity, "zero GEMV correctness check failed");
        return -5;
    }

    const float mean_microseconds =
        measured_milliseconds * 1'000.0f / static_cast<float>(iterations);
    const double operations =
        2.0 * static_cast<double>(input_features) * static_cast<double>(output_features);
    const double tera_operations_per_second =
        operations / (static_cast<double>(mean_microseconds) * 1.0e6);

    std::memset(output, 0, sizeof(*output));
    output->input_features = input_features;
    output->output_features = output_features;
    output->iterations = iterations;
    output->weight_bytes = weight_bytes;
    output->cold_launch_microseconds = cold_milliseconds * 1'000.0f;
    output->mean_launch_microseconds = mean_microseconds;
    output->tera_operations_per_second =
        static_cast<float>(tera_operations_per_second);
    write_error(error, error_capacity, "");
    return 0;
}
