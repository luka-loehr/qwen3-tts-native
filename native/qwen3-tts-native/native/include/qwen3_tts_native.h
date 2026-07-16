#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define QWEN3_TTS_API __declspec(dllexport)
#else
#define QWEN3_TTS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Qwen3TtsDeviceInfo {
    int32_t device_index;
    int32_t compute_major;
    int32_t compute_minor;
    uint64_t total_global_memory_bytes;
    uint64_t runtime_free_memory_bytes;
    uint64_t runtime_total_memory_bytes;
    char device_name[256];
} Qwen3TtsDeviceInfo;

typedef struct Qwen3TtsArgmaxBenchmark {
    int32_t vocabulary_size;
    int32_t iterations;
    int32_t selected_token;
    int32_t expected_token;
    float cold_launch_microseconds;
    float mean_launch_microseconds;
} Qwen3TtsArgmaxBenchmark;

typedef struct Qwen3TtsGemvBenchmark {
    int32_t input_features;
    int32_t output_features;
    int32_t iterations;
    int32_t reserved;
    uint64_t weight_bytes;
    float cold_launch_microseconds;
    float mean_launch_microseconds;
    float tera_operations_per_second;
} Qwen3TtsGemvBenchmark;

QWEN3_TTS_API int32_t qwen3_tts_probe_device(
    int32_t device_index,
    Qwen3TtsDeviceInfo* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_benchmark_bf16_argmax(
    int32_t device_index,
    int32_t vocabulary_size,
    int32_t iterations,
    Qwen3TtsArgmaxBenchmark* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_benchmark_bf16_gemv(
    int32_t device_index,
    int32_t input_features,
    int32_t output_features,
    int32_t iterations,
    Qwen3TtsGemvBenchmark* output,
    char* error,
    size_t error_capacity
);

#ifdef __cplusplus
}
#endif
