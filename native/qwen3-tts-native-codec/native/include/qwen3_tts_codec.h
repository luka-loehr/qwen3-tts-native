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
    QWEN3_TTS_CODEC_MAX_BATCH_STREAMS = 6,
    QWEN3_TTS_CODEC_TRANSFORMER_LAYERS = 8,
    QWEN3_TTS_CODEC_KV_HEADS = 16,
    QWEN3_TTS_CODEC_HEAD_DIM = 64,
    QWEN3_TTS_CODEC_KV_WINDOW = 72,
};

enum {
    QWEN3_TTS_CODEC_CHECKPOINT_DECODER_PRECONV = 6,
    QWEN3_TTS_CODEC_CHECKPOINT_DECODER_BLOCK_1 = 7,
    QWEN3_TTS_CODEC_CHECKPOINT_DECODER_BLOCK_2 = 8,
    QWEN3_TTS_CODEC_CHECKPOINT_DECODER_BLOCK_3 = 9,
    QWEN3_TTS_CODEC_CHECKPOINT_DECODER_BLOCK_4 = 10,
    QWEN3_TTS_CODEC_CHECKPOINT_FINAL_SNAKE = 11,
    QWEN3_TTS_CODEC_CHECKPOINT_FINAL_PRECLAMP = 12,
    QWEN3_TTS_CODEC_CHECKPOINT_FINAL_CLAMP = 13,
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
typedef struct Qwen3TtsCodecModelV1 Qwen3TtsCodecModelV1;
typedef Qwen3TtsCodecContextV1 Qwen3TtsCodecSessionV1;

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

typedef struct Qwen3TtsCodecBatchItemV1 {
    Qwen3TtsCodecContextV1* context;
    const uint16_t* codec_frames;
    uint32_t frame_count;
    int32_t is_final;
    int16_t* pcm_output;
    size_t pcm_capacity_samples;
    Qwen3TtsCodecPacketResultV1* result;
} Qwen3TtsCodecBatchItemV1;

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

/* Shared immutable device-weight ownership. A loaded model may be used to
 * create independent sessions from multiple host threads. */
typedef struct Qwen3TtsCodecModelMemoryInfoV1 {
    uint64_t source_bytes;
    uint64_t shared_weight_device_bytes;
    uint64_t parameter_count;
    uint64_t transient_upload_device_bytes;
    uint32_t tensor_count;
    uint32_t warmup_completed;
    uint32_t active_session_count;
    uint32_t reserved;
} Qwen3TtsCodecModelMemoryInfoV1;

/* Per-session state only. Shared model weights are deliberately excluded. */
typedef struct Qwen3TtsCodecSessionMemoryInfoV1 {
    uint64_t device_bytes;
    uint64_t host_pinned_bytes;
    uint64_t transformer_kv_bytes;
    uint64_t convolution_history_bytes;
    uint64_t codec_ring_bytes;
    uint64_t pcm_ring_bytes;
    uint64_t workspace_device_bytes;
    uint64_t reserved;
} Qwen3TtsCodecSessionMemoryInfoV1;

typedef struct Qwen3TtsCodecSessionBatchItemV1 {
    Qwen3TtsCodecSessionV1* session;
    const uint16_t* codec_frames;
    uint32_t frame_count;
    int32_t is_final;
    int16_t* pcm_output;
    size_t pcm_capacity_samples;
    Qwen3TtsCodecPacketResultV1* result;
} Qwen3TtsCodecSessionBatchItemV1;

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

/* Initialize CUDA/cuBLAS execution paths before the first user packet. The
 * call is accepted only on a fresh state handle and restores that state. */
QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_warmup_v1(
    Qwen3TtsCodecContextV1* context,
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

/* Research-only latent upsampler parity hook. Outputs are channel-major
 * [1, 1024, frames*2] and [1, 1024, frames*4]. */
QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_debug_latent_packet_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    float* stage_one_output,
    size_t stage_one_capacity,
    float* stage_two_output,
    size_t stage_two_capacity,
    char* error,
    size_t error_capacity
);

/* Research-only waveform-decoder checkpoint hook. Output layout is
 * channel-major and its exact shape is defined by the official fixture. */
QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_debug_decoder_checkpoint_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    uint32_t checkpoint,
    float* checkpoint_output,
    size_t checkpoint_capacity,
    char* error,
    size_t error_capacity
);

/* Decode one real Qwen3-TTS speech-tokenizer packet. Input is frame-major
 * [frame_count, 16]. Output is mono 24 kHz signed 16-bit PCM and contains
 * exactly frame_count * 1920 samples. State persists until reset. */
QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_process_packet_v1(
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

/* Decode one packet for each independent state handle. The reference
 * implementation dispatches items in array order; it does not fuse streams. */
QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_process_batch_v1(
    Qwen3TtsCodecBatchItemV1* items,
    uint32_t item_count,
    char* error,
    size_t error_capacity
);

/* Deterministic state-machine fixture. This does not execute neural weights
 * and must never be used as a model-quality or model-latency measurement. */
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

/* Additive shared-model API. Model weights are uploaded and warmed once;
 * sessions retain the model and own all mutable CUDA execution state. */
QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_shared_model_create_v1(
    int32_t device_index,
    Qwen3TtsCodecModelV1** output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_shared_model_destroy_v1(
    Qwen3TtsCodecModelV1* model,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_shared_model_load_v1(
    Qwen3TtsCodecModelV1* model,
    const Qwen3TtsCodecWeightProviderV1* provider,
    Qwen3TtsCodecModelInfoV1* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_shared_model_warmup_v1(
    Qwen3TtsCodecModelV1* model,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_shared_model_info_v1(
    const Qwen3TtsCodecModelV1* model,
    Qwen3TtsCodecModelInfoV1* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_shared_model_memory_info_v1(
    const Qwen3TtsCodecModelV1* model,
    Qwen3TtsCodecModelMemoryInfoV1* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_session_create_v1(
    Qwen3TtsCodecModelV1* model,
    Qwen3TtsCodecSessionV1** output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_session_destroy_v1(
    Qwen3TtsCodecSessionV1* session,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_session_reset_v1(
    Qwen3TtsCodecSessionV1* session,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_session_cancel_v1(
    Qwen3TtsCodecSessionV1* session,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_session_state_info_v1(
    const Qwen3TtsCodecSessionV1* session,
    Qwen3TtsCodecStateInfoV1* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_session_memory_info_v1(
    const Qwen3TtsCodecSessionV1* session,
    Qwen3TtsCodecSessionMemoryInfoV1* output,
    char* error,
    size_t error_capacity
);

QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_session_process_packet_v1(
    Qwen3TtsCodecSessionV1* session,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    int32_t is_final,
    int16_t* pcm_output,
    size_t pcm_capacity_samples,
    Qwen3TtsCodecPacketResultV1* result,
    char* error,
    size_t error_capacity
);

/* Dispatches in array order for ABI convenience. True host-thread
 * concurrency is provided by independent session handles and Rust workers. */
QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_session_process_batch_v1(
    Qwen3TtsCodecSessionBatchItemV1* items,
    uint32_t item_count,
    char* error,
    size_t error_capacity
);

#ifdef __cplusplus
}
#endif
