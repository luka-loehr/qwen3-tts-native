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

typedef struct Qwen3TtsDeviceBuffer Qwen3TtsDeviceBuffer;

typedef struct Qwen3TtsWeightUploadMetrics {
    int32_t device_index;
    int32_t reserved;
    uint64_t allocation_bytes;
    uint64_t pinned_staging_bytes;
    uint64_t uploaded_bytes;
    uint64_t upload_calls;
    uint64_t free_before_bytes;
    uint64_t free_after_allocation_bytes;
    float allocation_microseconds;
    float upload_microseconds;
} Qwen3TtsWeightUploadMetrics;

typedef struct Qwen3TtsPrimitiveParity {
    float rms_norm_max_absolute_error;
    float rope_max_absolute_error;
    float attention_max_absolute_error;
    float silu_gate_max_absolute_error;
} Qwen3TtsPrimitiveParity;

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

QWEN3_TTS_API int32_t qwen3_tts_device_buffer_create(
    int32_t device_index,
    uint64_t capacity_bytes,
    uint64_t staging_bytes,
    Qwen3TtsDeviceBuffer** output,
    Qwen3TtsWeightUploadMetrics* metrics,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_validate_transformer_primitives(
    int32_t device_index,
    Qwen3TtsPrimitiveParity* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_device_buffer_upload(
    Qwen3TtsDeviceBuffer* buffer,
    uint64_t offset_bytes,
    const void* source,
    uint64_t bytes,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_device_buffer_finish(
    Qwen3TtsDeviceBuffer* buffer,
    Qwen3TtsWeightUploadMetrics* metrics,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_device_buffer_read(
    Qwen3TtsDeviceBuffer* buffer,
    uint64_t offset_bytes,
    void* destination,
    uint64_t bytes,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API const void* qwen3_tts_device_buffer_data(
    const Qwen3TtsDeviceBuffer* buffer
);

QWEN3_TTS_API void qwen3_tts_device_buffer_destroy(
    Qwen3TtsDeviceBuffer* buffer
);
#ifdef __cplusplus
}
#endif
