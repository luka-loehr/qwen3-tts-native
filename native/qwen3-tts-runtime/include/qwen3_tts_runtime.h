#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define QWEN3_TTS_RUNTIME_API __declspec(dllexport)
#else
#define QWEN3_TTS_RUNTIME_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    QWEN3_TTS_RUNTIME_ABI_VERSION_V1 = 1,
    QWEN3_TTS_RUNTIME_SAMPLE_RATE = 24000,
    QWEN3_TTS_RUNTIME_CHANNELS = 1,
    QWEN3_TTS_RUNTIME_CODEBOOKS = 16,
    QWEN3_TTS_RUNTIME_SAMPLES_PER_CODEC_FRAME = 1920,
    QWEN3_TTS_RUNTIME_MAX_CONCURRENT_REQUESTS = 6,
    QWEN3_TTS_RUNTIME_MAX_PACKET_FRAMES = 4,
    QWEN3_TTS_RUNTIME_MAX_PCM_RING_SLOTS = 64,
    QWEN3_TTS_RUNTIME_MAX_CODEC_FRAMES = 8192,
    QWEN3_TTS_RUNTIME_MAX_TEXT_BYTES = 1048576,
    QWEN3_TTS_RUNTIME_MAX_INSTRUCT_BYTES = 262144,
};

typedef enum Qwen3TtsRuntimeStatusV1 {
    QWEN3_TTS_RUNTIME_OK = 0,
    QWEN3_TTS_RUNTIME_WOULD_BLOCK = 1,
    QWEN3_TTS_RUNTIME_END_OF_STREAM = 2,
    QWEN3_TTS_RUNTIME_INVALID_ARGUMENT = -1,
    QWEN3_TTS_RUNTIME_INVALID_UTF8 = -2,
    QWEN3_TTS_RUNTIME_UNSUPPORTED_LANGUAGE = -3,
    QWEN3_TTS_RUNTIME_MODEL = -4,
    QWEN3_TTS_RUNTIME_ALLOCATION = -5,
    QWEN3_TTS_RUNTIME_CUDA = -6,
    QWEN3_TTS_RUNTIME_STATE = -7,
    QWEN3_TTS_RUNTIME_CANCELLED = -8,
    QWEN3_TTS_RUNTIME_INTERNAL = -9,
} Qwen3TtsRuntimeStatusV1;

typedef enum Qwen3TtsLanguageV1 {
    QWEN3_TTS_LANGUAGE_AUTO = 0,
    QWEN3_TTS_LANGUAGE_CHINESE = 1,
    QWEN3_TTS_LANGUAGE_ENGLISH = 2,
    QWEN3_TTS_LANGUAGE_JAPANESE = 3,
    QWEN3_TTS_LANGUAGE_KOREAN = 4,
    QWEN3_TTS_LANGUAGE_GERMAN = 5,
    QWEN3_TTS_LANGUAGE_FRENCH = 6,
    QWEN3_TTS_LANGUAGE_RUSSIAN = 7,
    QWEN3_TTS_LANGUAGE_PORTUGUESE = 8,
    QWEN3_TTS_LANGUAGE_SPANISH = 9,
    QWEN3_TTS_LANGUAGE_ITALIAN = 10,
} Qwen3TtsLanguageV1;

typedef struct Qwen3TtsEngineV1 Qwen3TtsEngineV1;
typedef struct Qwen3TtsRequestV1 Qwen3TtsRequestV1;

typedef struct Qwen3TtsEngineConfigV1 {
    uint32_t struct_size;
    int32_t device_index;
    uint32_t max_concurrent_requests;
    uint32_t packet_frames;
    uint32_t pcm_ring_slots;
    uint32_t max_text_bytes;
    uint32_t max_instruct_bytes;
    uint32_t flags;
    uint64_t reserved[8];
} Qwen3TtsEngineConfigV1;

typedef struct Qwen3TtsGenerationConfigV1 {
    uint32_t struct_size;
    uint32_t max_codec_frames;
    uint64_t seed;
    float temperature;
    float top_p;
    float repetition_penalty;
    uint32_t top_k;
    uint32_t do_sample;
    float predictor_temperature;
    float predictor_top_p;
    uint32_t predictor_top_k;
    uint32_t predictor_do_sample;
    uint64_t reserved[8];
} Qwen3TtsGenerationConfigV1;

typedef struct Qwen3TtsRequestInputV1 {
    uint32_t struct_size;
    uint32_t language;
    const uint8_t* text_utf8;
    size_t text_bytes;
    const uint8_t* instruct_utf8;
    size_t instruct_bytes;
    Qwen3TtsGenerationConfigV1 generation;
} Qwen3TtsRequestInputV1;

typedef struct Qwen3TtsAudioPacketV1 {
    uint64_t request_id;
    uint64_t sequence;
    uint64_t first_codec_frame;
    uint64_t first_sample;
    uint32_t codec_frames;
    uint32_t sample_count;
    uint32_t sample_rate;
    uint32_t channels;
    uint32_t is_final;
    uint32_t reserved;
    float talker_gpu_microseconds;
    float codec_gpu_microseconds;
    float end_to_end_microseconds;
} Qwen3TtsAudioPacketV1;

typedef struct Qwen3TtsRequestMetricsV1 {
    uint64_t queue_microseconds;
    uint64_t prefill_microseconds;
    uint64_t first_codec_frame_microseconds;
    uint64_t first_audio_microseconds;
    uint64_t wall_microseconds;
    uint64_t generated_codec_frames;
    uint64_t emitted_samples;
    uint64_t emitted_packets;
    double talker_gpu_microseconds;
    double codec_gpu_microseconds;
    uint64_t peak_request_device_bytes;
    uint64_t peak_request_host_bytes;
} Qwen3TtsRequestMetricsV1;

QWEN3_TTS_RUNTIME_API uint32_t qwen3_tts_runtime_abi_version_v1(void);

/*
 * The engine/request functions below define the final ownership contract. The
 * implementation is connected only after real talker and codec parity pass.
 * Input UTF-8 is copied by request_start. poll copies s16 mono PCM into the
 * caller buffer and never exposes internal ring-buffer pointers.
 */
QWEN3_TTS_RUNTIME_API int32_t qwen3_tts_engine_create_v1(
    const uint8_t* model_root_utf8,
    size_t model_root_bytes,
    const Qwen3TtsEngineConfigV1* config,
    Qwen3TtsEngineV1** output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_RUNTIME_API int32_t qwen3_tts_engine_destroy_v1(
    Qwen3TtsEngineV1* engine,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_RUNTIME_API int32_t qwen3_tts_request_start_v1(
    Qwen3TtsEngineV1* engine,
    const Qwen3TtsRequestInputV1* input,
    Qwen3TtsRequestV1** output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_RUNTIME_API int32_t qwen3_tts_request_poll_v1(
    Qwen3TtsRequestV1* request,
    uint32_t timeout_milliseconds,
    int16_t* pcm_output,
    size_t pcm_capacity_samples,
    Qwen3TtsAudioPacketV1* packet,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_RUNTIME_API int32_t qwen3_tts_request_cancel_v1(
    Qwen3TtsRequestV1* request,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_RUNTIME_API int32_t qwen3_tts_request_metrics_v1(
    const Qwen3TtsRequestV1* request,
    Qwen3TtsRequestMetricsV1* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_RUNTIME_API int32_t qwen3_tts_request_destroy_v1(
    Qwen3TtsRequestV1* request,
    char* error,
    size_t error_capacity
);

#ifdef __cplusplus
}
#endif
