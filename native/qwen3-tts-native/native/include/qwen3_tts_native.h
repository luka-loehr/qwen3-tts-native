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

#define QWEN3_TTS_TALKER_ABI_VERSION 1U
#define QWEN3_TTS_CODEC_CODEBOOKS 16U

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

typedef struct Qwen3TtsSamplingConfig {
    int32_t do_sample;
    int32_t top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
} Qwen3TtsSamplingConfig;

typedef struct Qwen3TtsTalkerPrefillResult {
    uint16_t first_semantic_token;
    uint16_t reserved;
    uint32_t prompt_tokens;
    float talker_gpu_milliseconds;
} Qwen3TtsTalkerPrefillResult;

typedef struct Qwen3TtsCodecFrameResult {
    uint16_t codes[QWEN3_TTS_CODEC_CODEBOOKS];
    uint16_t next_semantic_token;
    uint16_t ended_by_eos;
    uint32_t talker_position;
    float predictor_gpu_milliseconds;
    float talker_gpu_milliseconds;
} Qwen3TtsCodecFrameResult;

typedef struct Qwen3TtsCodecFrameInfo {
    uint32_t talker_position;
    uint32_t ended_by_eos;
    float predictor_gpu_milliseconds;
    float talker_gpu_milliseconds;
} Qwen3TtsCodecFrameInfo;

typedef enum Qwen3TtsTalkerPhase {
    QWEN3_TTS_TALKER_CREATED = 0,
    QWEN3_TTS_TALKER_READY = 1,
    QWEN3_TTS_TALKER_PREFILLED = 2,
    QWEN3_TTS_TALKER_ENDED = 3,
} Qwen3TtsTalkerPhase;

typedef struct Qwen3TtsTalkerStateInfo {
    uint32_t abi_version;
    uint32_t phase;
    uint32_t talker_position;
    uint32_t semantic_history_count;
    uint64_t frames_generated;
    uint64_t device_sample_count;
    uint64_t host_sync_count;
} Qwen3TtsTalkerStateInfo;

typedef struct Qwen3TtsModelMemory {
    uint64_t shared_weight_bytes;
    uint32_t tensor_count;
    int32_t device_index;
} Qwen3TtsModelMemory;

typedef struct Qwen3TtsSessionMemory {
    uint64_t talker_kv_bytes;
    uint64_t predictor_kv_bytes;
    uint64_t workspace_bytes;
    uint32_t max_sequence_length;
    uint32_t reserved;
} Qwen3TtsSessionMemory;

typedef void* Qwen3TtsModelHandle;
typedef void* Qwen3TtsSessionHandle;

QWEN3_TTS_API uint32_t qwen3_tts_talker_abi_version(void);

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

QWEN3_TTS_API int32_t qwen3_tts_model_create(
    int32_t device_index,
    Qwen3TtsModelHandle* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API void qwen3_tts_model_destroy(Qwen3TtsModelHandle handle);

QWEN3_TTS_API int32_t qwen3_tts_model_upload_tensor(
    Qwen3TtsModelHandle handle,
    const char* name,
    const void* bf16_data,
    uint64_t byte_size,
    int32_t rank,
    const uint64_t* shape,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_model_finalize(
    Qwen3TtsModelHandle handle,
    Qwen3TtsModelMemory* memory,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_session_create(
    Qwen3TtsModelHandle model,
    int32_t max_sequence_length,
    uint64_t random_seed,
    Qwen3TtsSessionHandle* output,
    Qwen3TtsSessionMemory* memory,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API void qwen3_tts_session_destroy(Qwen3TtsSessionHandle handle);

QWEN3_TTS_API int32_t qwen3_tts_session_reset(
    Qwen3TtsSessionHandle handle,
    uint64_t random_seed,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_session_prefill(
    Qwen3TtsSessionHandle handle,
    const int32_t* text_token_ids,
    const int32_t* codec_token_ids,
    int32_t token_count,
    Qwen3TtsSamplingConfig sampling,
    Qwen3TtsTalkerPrefillResult* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_session_next_frame(
    Qwen3TtsSessionHandle handle,
    uint16_t semantic_token,
    int32_t trailing_text_token_id,
    Qwen3TtsSamplingConfig talker_sampling,
    Qwen3TtsSamplingConfig predictor_sampling,
    uint16_t* output_codes,
    size_t output_code_capacity,
    uint16_t* next_semantic_token,
    Qwen3TtsCodecFrameInfo* frame_info,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_API int32_t qwen3_tts_session_state_info(
    Qwen3TtsSessionHandle handle,
    Qwen3TtsTalkerStateInfo* output,
    char* error,
    size_t error_capacity
);

#ifdef __cplusplus
}
#endif
