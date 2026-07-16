#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define QWEN3_TTS_CODEC_API __declspec(dllexport)
#else
#define QWEN3_TTS_CODEC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    QWEN3_TTS_CODEC_ABI_VERSION_V1 = 1,
    QWEN3_TTS_CODEC_CODEBOOKS = 16,
    QWEN3_TTS_CODEC_MAX_PACKET_FRAMES = 4,
    QWEN3_TTS_CODEC_SAMPLES_PER_FRAME = 1920,
    QWEN3_TTS_CODEC_MAX_PACKET_SAMPLES = 7680,
    QWEN3_TTS_CODEC_RING_SLOTS = 3,
    QWEN3_TTS_CODEC_TRANSFORMER_LAYERS = 8,
    QWEN3_TTS_CODEC_KV_HEADS = 16,
    QWEN3_TTS_CODEC_HEAD_DIM = 64,
    QWEN3_TTS_CODEC_KV_WINDOW = 72,
};

enum {
    QWEN3_TTS_CODEC_STATUS_OK = 0,
    QWEN3_TTS_CODEC_STATUS_INVALID_ARGUMENT = -1,
    QWEN3_TTS_CODEC_STATUS_CUDA = -2,
    QWEN3_TTS_CODEC_STATUS_STATE = -3,
    QWEN3_TTS_CODEC_STATUS_ALLOCATION = -4,
    QWEN3_TTS_CODEC_STATUS_MODEL = -5,
};

enum {
    QWEN3_TTS_CODEC_TENSOR_F32 = 1,
    QWEN3_TTS_CODEC_TENSOR_BF16 = 2,
    QWEN3_TTS_CODEC_MAX_TENSOR_RANK = 4,
};

typedef struct Qwen3TtsCodecContextV1 Qwen3TtsCodecContextV1;

typedef struct Qwen3TtsCodecConfigV1 {
    int32_t device_index;
    int32_t ring_slots;
    int32_t max_packet_frames;
    int32_t reserved;
} Qwen3TtsCodecConfigV1;

typedef struct Qwen3TtsCodecStateInfoV1 {
    uint64_t frame_position;
    uint64_t emitted_samples;
    uint64_t device_bytes;
    uint64_t host_pinned_bytes;
    uint64_t transformer_kv_bytes;
    uint64_t convolution_history_bytes;
    uint64_t codec_ring_bytes;
    uint64_t pcm_ring_bytes;
    uint32_t kv_ring_head;
    uint32_t next_ring_slot;
    uint32_t ring_slots;
    uint32_t max_packet_frames;
} Qwen3TtsCodecStateInfoV1;

typedef struct Qwen3TtsCodecPacketResultV1 {
    uint64_t first_frame_position;
    uint64_t first_sample_position;
    uint32_t frame_count;
    uint32_t sample_count;
    uint32_t ring_slot;
    uint32_t is_final;
    float gpu_microseconds;
    float end_to_end_microseconds;
} Qwen3TtsCodecPacketResultV1;

typedef struct Qwen3TtsCodecTensorViewV1 {
    const void* data;
    uint64_t byte_length;
    uint64_t shape[QWEN3_TTS_CODEC_MAX_TENSOR_RANK];
    uint32_t rank;
    uint32_t dtype;
} Qwen3TtsCodecTensorViewV1;

typedef int32_t (*Qwen3TtsCodecTensorAtV1)(
    void* user_data,
    uint64_t index,
    const char** name,
    Qwen3TtsCodecTensorViewV1* tensor,
    char* error,
    size_t error_capacity
);

typedef struct Qwen3TtsCodecWeightProviderV1 {
    uint32_t abi_version;
    uint32_t reserved;
    uint64_t tensor_count;
    void* user_data;
    Qwen3TtsCodecTensorAtV1 tensor_at;
} Qwen3TtsCodecWeightProviderV1;

typedef struct Qwen3TtsCodecModelInfoV1 {
    uint64_t source_bytes;
    uint64_t device_bytes;
    uint64_t parameter_count;
    uint32_t tensor_count;
    uint32_t source_dtype_f32_count;
    uint32_t source_dtype_bf16_count;
    uint32_t loaded;
} Qwen3TtsCodecModelInfoV1;

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_abi_version_v1(void);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_create_v1(
    const Qwen3TtsCodecConfigV1* config,
    Qwen3TtsCodecContextV1** output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_destroy_v1(
    Qwen3TtsCodecContextV1* context,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_reset_v1(
    Qwen3TtsCodecContextV1* context,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_state_info_v1(
    const Qwen3TtsCodecContextV1* context,
    Qwen3TtsCodecStateInfoV1* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_load_model_v1(
    Qwen3TtsCodecContextV1* context,
    const Qwen3TtsCodecWeightProviderV1* provider,
    Qwen3TtsCodecModelInfoV1* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_model_info_v1(
    const Qwen3TtsCodecContextV1* context,
    Qwen3TtsCodecModelInfoV1* output,
    char* error,
    size_t error_capacity
);

/* Research-only parity hook. Output layouts match the official checkpoints:
 * RVQ is [1, 512, frames], pre-convolution is [1, frames, 1024]. */
QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_debug_frontend_packet_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    float* rvq_output,
    size_t rvq_capacity,
    float* preconv_output,
    size_t preconv_capacity,
    char* error,
    size_t error_capacity
);

/* Research-only transformer parity hook. Output layout is [1, frames, 1024]. */
QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_debug_transformer_packet_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    float* transformer_output,
    size_t transformer_capacity,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_process_fixture_packet_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    int32_t is_final,
    int16_t* pcm_output,
    size_t pcm_capacity_samples,
    Qwen3TtsCodecPacketResultV1* result,
    char* error,
    size_t error_capacity
);

#ifdef __cplusplus
}
#endif
