#include "qwen3_tts_codec.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr size_t kBf16Bytes = sizeof(__nv_bfloat16);
constexpr size_t kTransformerKvElements =
    2ULL * QWEN3_TTS_CODEC_TRANSFORMER_LAYERS * QWEN3_TTS_CODEC_KV_HEADS *
    QWEN3_TTS_CODEC_KV_WINDOW * QWEN3_TTS_CODEC_HEAD_DIM;
constexpr size_t kTransformerKvBytes = kTransformerKvElements * kBf16Bytes;

// Exact persistent BF16 history layout for the 24 kHz tokenizer decoder.
constexpr size_t kPreConvOffset = 0;
constexpr size_t kPreConvElements = 512ULL * 2;
constexpr size_t kConvNextOffset = kPreConvOffset + kPreConvElements;
constexpr size_t kConvNextElements = 2ULL * 1024 * 6;
constexpr size_t kDecoderInputOffset = kConvNextOffset + kConvNextElements;
constexpr size_t kDecoderInputElements = 1024ULL * 6;
constexpr size_t kTransposeTailOffset = kDecoderInputOffset + kDecoderInputElements;
constexpr size_t kTransposeTailElements =
    768ULL * 8 + 384ULL * 5 + 192ULL * 4 + 96ULL * 3;
constexpr size_t kResidualOffset = kTransposeTailOffset + kTransposeTailElements;
constexpr size_t kResidualElements =
    (768ULL + 384ULL + 192ULL + 96ULL) * (6ULL + 18ULL + 54ULL);
constexpr size_t kFinalConvOffset = kResidualOffset + kResidualElements;
constexpr size_t kFinalConvElements = 96ULL * 6;
constexpr size_t kConvolutionHistoryElements =
    kFinalConvOffset + kFinalConvElements;
constexpr size_t kConvolutionHistoryBytes =
    kConvolutionHistoryElements * kBf16Bytes;

static_assert(kTransformerKvElements == 1'179'648);
static_assert(kTransformerKvBytes == 2'359'296);
static_assert(kConvolutionHistoryElements == 141'472);
static_assert(kConvolutionHistoryBytes == 282'944);

constexpr size_t kCodecRingElements =
    QWEN3_TTS_CODEC_RING_SLOTS * QWEN3_TTS_CODEC_MAX_PACKET_FRAMES *
    QWEN3_TTS_CODEC_CODEBOOKS;
constexpr size_t kCodecRingBytes = kCodecRingElements * sizeof(uint16_t);
constexpr size_t kPcmRingElements =
    QWEN3_TTS_CODEC_RING_SLOTS * QWEN3_TTS_CODEC_MAX_PACKET_SAMPLES;
constexpr size_t kPcmRingBytes = kPcmRingElements * sizeof(int16_t);
constexpr size_t kFixtureHistoryElements = 2 + QWEN3_TTS_CODEC_KV_WINDOW + 20;
constexpr size_t kFixtureHistoryBytes = kFixtureHistoryElements * sizeof(int32_t);

constexpr size_t kScratch0Elements = QWEN3_TTS_CODEC_MAX_PACKET_FRAMES;
constexpr size_t kScratch1Elements = kScratch0Elements * 2;
constexpr size_t kScratch2Elements = kScratch1Elements * 2;
constexpr size_t kScratch3Elements = kScratch2Elements * 8;
constexpr size_t kScratch4Elements = kScratch3Elements * 5;
constexpr size_t kScratch5Elements = kScratch4Elements * 4;
constexpr size_t kScratch6Elements = kScratch5Elements * 3;
constexpr size_t kScratchBytes =
    (kScratch0Elements + kScratch1Elements + kScratch2Elements +
     kScratch3Elements + kScratch4Elements + kScratch5Elements +
     kScratch6Elements) *
    sizeof(int32_t);
constexpr size_t kFrontendQuantizedElements =
    QWEN3_TTS_CODEC_MAX_PACKET_FRAMES * 512;
constexpr size_t kFrontendRvqElements =
    QWEN3_TTS_CODEC_MAX_PACKET_FRAMES * 512;
constexpr size_t kFrontendPreconvElements =
    QWEN3_TTS_CODEC_MAX_PACKET_FRAMES * 1024;
constexpr size_t kFrontendHistoryElements = 2 * 512;
constexpr size_t kFrontendDeviceBytes =
    (kFrontendQuantizedElements + kFrontendRvqElements +
     kFrontendPreconvElements + kFrontendHistoryElements) *
    sizeof(float);
constexpr size_t kNeuralKvElements =
    2ULL * QWEN3_TTS_CODEC_TRANSFORMER_LAYERS * QWEN3_TTS_CODEC_KV_HEADS *
    QWEN3_TTS_CODEC_KV_WINDOW * QWEN3_TTS_CODEC_HEAD_DIM;
constexpr size_t kTransformerPacketElements =
    QWEN3_TTS_CODEC_MAX_PACKET_FRAMES * 1024;
constexpr size_t kTransformerScratchElements = 8192;
constexpr size_t kTransformerDeviceBytes =
    (kNeuralKvElements + kTransformerPacketElements +
     kTransformerScratchElements) *
    sizeof(float);
constexpr size_t kLatentMaxPositions =
    QWEN3_TTS_CODEC_MAX_PACKET_FRAMES * 4;
constexpr size_t kLatentElements = kLatentMaxPositions * 1024;
constexpr size_t kLatentExpandedElements = kLatentMaxPositions * 4096;
constexpr size_t kLatentHistoryElements = 2 * 6 * 1024;
constexpr size_t kLatentDeviceBytes =
    (2 * kLatentElements + kLatentExpandedElements +
     kLatentHistoryElements) *
    sizeof(float);
constexpr size_t kDecoderMaxPositions =
    QWEN3_TTS_CODEC_MAX_PACKET_FRAMES * QWEN3_TTS_CODEC_SAMPLES_PER_FRAME;
constexpr size_t kDecoderMaxActivationElements = kDecoderMaxPositions * 96;
constexpr size_t kDecoderMaxIm2colElements = kDecoderMaxPositions * 96 * 7;
constexpr size_t kDecoderHistoryElements = 128160;
constexpr size_t kDecoderDeviceBytes =
    (2 * kDecoderMaxActivationElements + kDecoderMaxIm2colElements +
     kDecoderHistoryElements) *
    sizeof(float);
constexpr size_t kNeuralTransformerKvBytes =
    kNeuralKvElements * sizeof(float);
constexpr size_t kNeuralConvolutionHistoryBytes =
    (kFrontendHistoryElements + kLatentHistoryElements +
     kDecoderHistoryElements) *
    sizeof(float);
constexpr size_t kDecoderPreconvHistoryOffset = 0;
constexpr size_t kDecoderTransposeHistoryOffset = 6 * 1024;
constexpr size_t kDecoderResidualHistoryOffset =
    kDecoderTransposeHistoryOffset + (768 * 8 + 384 * 5 + 192 * 4 + 96 * 3);
constexpr size_t kDecoderFinalHistoryOffset =
    kDecoderResidualHistoryOffset + (768 + 384 + 192 + 96) * 78;
static_assert(kDecoderFinalHistoryOffset + 96 * 6 == kDecoderHistoryElements);

constexpr int32_t kStatusOk = QWEN3_TTS_CODEC_STATUS_OK;
constexpr int32_t kStatusInvalidArgument =
    QWEN3_TTS_CODEC_STATUS_INVALID_ARGUMENT;
constexpr int32_t kStatusCuda = QWEN3_TTS_CODEC_STATUS_CUDA;
constexpr int32_t kStatusState = QWEN3_TTS_CODEC_STATUS_STATE;
constexpr int32_t kStatusAllocation = QWEN3_TTS_CODEC_STATUS_ALLOCATION;
constexpr int32_t kStatusModel = QWEN3_TTS_CODEC_STATUS_MODEL;
constexpr uint32_t kExpectedDecoderTensors = 271;
constexpr size_t kBf16UploadStagingBytes = 8ULL * 1024 * 1024;

void clear_error(char* error, size_t capacity) noexcept {
    if (error != nullptr && capacity > 0) {
        error[0] = '\0';
    }
}

void write_error(char* error, size_t capacity, const char* message) noexcept {
    if (error != nullptr && capacity > 0) {
        std::snprintf(error, capacity, "%s", message);
        error[capacity - 1] = '\0';
    }
}

int32_t cuda_error(
    char* error,
    size_t capacity,
    const char* operation,
    cudaError_t status
) noexcept {
    if (error != nullptr && capacity > 0) {
        std::snprintf(
            error,
            capacity,
            "%s: %s",
            operation,
            cudaGetErrorString(status)
        );
        error[capacity - 1] = '\0';
    }
    return kStatusCuda;
}

int32_t cublas_error(
    char* error,
    size_t capacity,
    const char* operation,
    cublasStatus_t status
) noexcept {
    if (error != nullptr && capacity > 0) {
        std::snprintf(error, capacity, "%s: cuBLAS status %d", operation, status);
        error[capacity - 1] = '\0';
    }
    return kStatusCuda;
}

__global__ void ingest_fixture_kernel(
    const uint16_t* codec_frames,
    uint32_t frame_count,
    uint64_t frame_position,
    int32_t* preconv_history,
    int32_t* representative_kv_ring,
    int32_t* output
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int32_t previous_1 = preconv_history[0];
    int32_t previous_2 = preconv_history[1];
    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        int32_t weighted_sum = 0;
        for (uint32_t codebook = 0; codebook < QWEN3_TTS_CODEC_CODEBOOKS;
             ++codebook) {
            const int32_t code =
                codec_frames[frame * QWEN3_TTS_CODEC_CODEBOOKS + codebook] &
                2047;
            weighted_sum += code * static_cast<int32_t>(codebook + 1);
        }
        const int32_t centered = (weighted_sum & 4095) - 2048;
        const int32_t filtered =
            (3 * centered + 2 * previous_1 - previous_2) / 4;
        previous_2 = previous_1;
        previous_1 = centered;
        output[frame] = filtered;
        representative_kv_ring[(frame_position + frame) %
                               QWEN3_TTS_CODEC_KV_WINDOW] = filtered;
    }
    preconv_history[0] = previous_1;
    preconv_history[1] = previous_2;
}

__global__ void update_exact_kv_fixture_kernel(
    const int32_t* frame_values,
    uint32_t frame_count,
    uint64_t frame_position,
    __nv_bfloat16* kv
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr size_t kItemsPerFrame =
        2ULL * QWEN3_TTS_CODEC_TRANSFORMER_LAYERS *
        QWEN3_TTS_CODEC_KV_HEADS * QWEN3_TTS_CODEC_HEAD_DIM;
    const size_t total = static_cast<size_t>(frame_count) * kItemsPerFrame;
    if (item >= total) {
        return;
    }

    size_t local = item;
    const size_t dim = local % QWEN3_TTS_CODEC_HEAD_DIM;
    local /= QWEN3_TTS_CODEC_HEAD_DIM;
    const size_t head = local % QWEN3_TTS_CODEC_KV_HEADS;
    local /= QWEN3_TTS_CODEC_KV_HEADS;
    const size_t layer = local % QWEN3_TTS_CODEC_TRANSFORMER_LAYERS;
    local /= QWEN3_TTS_CODEC_TRANSFORMER_LAYERS;
    const size_t key_or_value = local % 2;
    const size_t frame = local / 2;
    const size_t slot =
        (frame_position + frame) % QWEN3_TTS_CODEC_KV_WINDOW;
    const size_t destination =
        (((key_or_value * QWEN3_TTS_CODEC_TRANSFORMER_LAYERS + layer) *
              QWEN3_TTS_CODEC_KV_HEADS +
          head) *
             QWEN3_TTS_CODEC_KV_WINDOW +
         slot) *
            QWEN3_TTS_CODEC_HEAD_DIM +
        dim;
    const float tag = static_cast<float>(
        key_or_value * 31 + layer * 7 + head * 3 + dim
    );
    const float value = static_cast<float>(frame_values[frame]) / 4096.0F +
                        tag / 8192.0F;
    kv[destination] = __float2bfloat16(value);
}

__global__ void repeat_kernel(
    const int32_t* input,
    size_t input_count,
    uint32_t repeat,
    int32_t* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = input_count * repeat;
    if (item < total) {
        output[item] = input[item / repeat];
    }
}

__global__ void transpose_overlap_kernel(
    const int32_t* input,
    size_t input_count,
    uint32_t stride,
    const int32_t* prior_tail,
    int32_t* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = input_count * stride;
    if (item >= total) {
        return;
    }
    const size_t input_index = item / stride;
    const size_t phase = item % stride;
    const int32_t previous =
        input_index == 0 ? prior_tail[phase] : input[input_index - 1];
    output[item] = (3 * input[input_index] + previous) / 4;
}

__global__ void update_tail_kernel(
    const int32_t* input,
    size_t input_count,
    uint32_t stride,
    int32_t* tail
) {
    const uint32_t phase = blockIdx.x * blockDim.x + threadIdx.x;
    if (phase < stride) {
        tail[phase] = input[input_count - 1];
    }
}

__global__ void convert_pcm_kernel(
    const int32_t* input,
    size_t sample_count,
    int16_t* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= sample_count) {
        return;
    }
    int32_t value = input[item] * 8;
    value = value > 32767 ? 32767 : value;
    value = value < -32768 ? -32768 : value;
    output[item] = static_cast<int16_t>(value);
}

__global__ void gather_codebook_kernel(
    const uint16_t* codes,
    uint32_t frame_count,
    uint32_t codebook,
    const float* embedding_sum,
    const float* cluster_usage,
    float* output,
    int32_t add
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(frame_count) * 256;
    if (item >= total) {
        return;
    }
    const size_t frame = item / 256;
    const size_t dimension = item % 256;
    const uint16_t code = codes[frame * QWEN3_TTS_CODEC_CODEBOOKS + codebook];
    const float usage = fmaxf(cluster_usage[code], 1.0e-5F);
    const float value = embedding_sum[static_cast<size_t>(code) * 256 + dimension] / usage;
    output[item] = add != 0 ? output[item] + value : value;
}

__global__ void add_rvq_projection_kernel(
    const float* acoustic,
    uint32_t frame_count,
    float* semantic
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(frame_count) * 512;
    if (item < total) {
        semantic[item] += acoustic[item];
    }
}

__global__ void causal_preconv_kernel(
    const float* input,
    const float* history,
    uint32_t frame_count,
    const float* weight,
    const float* bias,
    float* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(frame_count) * 1024;
    if (item >= total) {
        return;
    }
    const int32_t frame = static_cast<int32_t>(item / 1024);
    const size_t output_channel = item % 1024;
    float value = bias[output_channel];
    for (size_t input_channel = 0; input_channel < 512; ++input_channel) {
        for (int32_t kernel = 0; kernel < 3; ++kernel) {
            const int32_t source_frame = frame - (2 - kernel);
            const float source = source_frame >= 0
                                     ? input[static_cast<size_t>(source_frame) * 512 + input_channel]
                                     : history[static_cast<size_t>(2 + source_frame) * 512 + input_channel];
            const size_t weight_index =
                (output_channel * 512 + input_channel) * 3 + kernel;
            value = fmaf(weight[weight_index], source, value);
        }
    }
    output[static_cast<size_t>(frame) * 1024 + output_channel] = value;
}

__global__ void update_frontend_history_kernel(
    const float* input,
    uint32_t frame_count,
    float* history
) {
    const size_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= 512) {
        return;
    }
    if (frame_count == 1) {
        const float prior = history[512 + channel];
        history[channel] = prior;
        history[512 + channel] = input[channel];
    } else {
        history[channel] =
            input[(static_cast<size_t>(frame_count) - 2) * 512 + channel];
        history[512 + channel] =
            input[(static_cast<size_t>(frame_count) - 1) * 512 + channel];
    }
}

__global__ void add_bias_kernel(float* values, const float* bias, size_t count, size_t width) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < count) {
        values[item] += bias[item % width];
    }
}

__global__ void rms_norm_kernel(
    const float* input,
    const float* weight,
    size_t width,
    float epsilon,
    float* output
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float square_sum = 0.0F;
    for (size_t index = 0; index < width; ++index) {
        square_sum = fmaf(input[index], input[index], square_sum);
    }
    const float scale = rsqrtf(square_sum / static_cast<float>(width) + epsilon);
    for (size_t index = 0; index < width; ++index) {
        output[index] = input[index] * scale * weight[index];
    }
}

__global__ void apply_rope_kernel(
    float* query,
    float* key,
    uint64_t position
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= 16 * 64) {
        return;
    }
    const size_t head = item / 64;
    const size_t dimension = item % 64;
    if (dimension >= 32) {
        return;
    }
    const float exponent = static_cast<float>(2 * dimension) / 64.0F;
    const float frequency = 1.0F / powf(10000.0F, exponent);
    const float angle = static_cast<float>(position) * frequency;
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const size_t first = head * 64 + dimension;
    const size_t second = first + 32;
    const float q_first = query[first];
    const float q_second = query[second];
    const float k_first = key[first];
    const float k_second = key[second];
    query[first] = q_first * cosine - q_second * sine;
    query[second] = q_second * cosine + q_first * sine;
    key[first] = k_first * cosine - k_second * sine;
    key[second] = k_second * cosine + k_first * sine;
}

__global__ void sliding_attention_kernel(
    const float* query,
    const float* key,
    const float* value,
    uint32_t layer,
    uint64_t position,
    float* kv,
    float* output
) {
    const uint32_t head = threadIdx.x;
    if (blockIdx.x != 0 || head >= QWEN3_TTS_CODEC_KV_HEADS) {
        return;
    }
    const size_t slot = position % QWEN3_TTS_CODEC_KV_WINDOW;
    for (size_t dimension = 0; dimension < QWEN3_TTS_CODEC_HEAD_DIM; ++dimension) {
        const size_t cache_index =
            ((((static_cast<size_t>(layer) * QWEN3_TTS_CODEC_KV_HEADS + head) *
                QWEN3_TTS_CODEC_KV_WINDOW +
               slot) *
                  QWEN3_TTS_CODEC_HEAD_DIM) +
             dimension);
        kv[cache_index] = key[head * 64 + dimension];
        kv[kNeuralKvElements / 2 + cache_index] = value[head * 64 + dimension];
    }
    const uint64_t first_position =
        position + 1 > QWEN3_TTS_CODEC_KV_WINDOW
            ? position + 1 - QWEN3_TTS_CODEC_KV_WINDOW
            : 0;
    const uint32_t count = static_cast<uint32_t>(position - first_position + 1);
    float scores[QWEN3_TTS_CODEC_KV_WINDOW];
    float maximum = -3.402823466e+38F;
    for (uint32_t index = 0; index < count; ++index) {
        const size_t source_slot =
            (first_position + index) % QWEN3_TTS_CODEC_KV_WINDOW;
        float score = 0.0F;
        for (size_t dimension = 0; dimension < 64; ++dimension) {
            const size_t cache_index =
                ((((static_cast<size_t>(layer) * 16 + head) * 72 + source_slot) * 64) +
                 dimension);
            score = fmaf(
                query[head * 64 + dimension], kv[cache_index], score
            );
        }
        score *= 0.125F;
        scores[index] = score;
        maximum = fmaxf(maximum, score);
    }
    float denominator = 0.0F;
    for (uint32_t index = 0; index < count; ++index) {
        scores[index] = expf(scores[index] - maximum);
        denominator += scores[index];
    }
    for (size_t dimension = 0; dimension < 64; ++dimension) {
        float result = 0.0F;
        for (uint32_t index = 0; index < count; ++index) {
            const size_t source_slot =
                (first_position + index) % QWEN3_TTS_CODEC_KV_WINDOW;
            const size_t cache_index =
                ((((static_cast<size_t>(layer) * 16 + head) * 72 + source_slot) * 64) +
                 dimension);
            result = fmaf(
                scores[index] / denominator,
                kv[kNeuralKvElements / 2 + cache_index],
                result
            );
        }
        output[head * 64 + dimension] = result;
    }
}

__global__ void scaled_residual_kernel(
    const float* residual,
    const float* update,
    const float* scale,
    size_t width,
    float* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < width) {
        output[item] = residual[item] + update[item] * scale[item];
    }
}

__global__ void silu_product_kernel(
    const float* gate,
    const float* up,
    size_t width,
    float* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < width) {
        const float value = gate[item];
        output[item] = (value / (1.0F + expf(-value))) * up[item];
    }
}

__global__ void latent_transpose_kernel(
    const float* input,
    size_t input_positions,
    const float* weight,
    const float* bias,
    float* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t output_positions = input_positions * 2;
    const size_t total = output_positions * 1024;
    if (item >= total) {
        return;
    }
    const size_t output_position = item / 1024;
    const size_t output_channel = item % 1024;
    const size_t input_position = output_position / 2;
    const size_t phase = output_position % 2;
    float value = bias[output_channel];
    for (size_t input_channel = 0; input_channel < 1024; ++input_channel) {
        const size_t weight_index =
            (input_channel * 1024 + output_channel) * 2 + phase;
        value = fmaf(
            input[input_position * 1024 + input_channel],
            weight[weight_index],
            value
        );
    }
    output[item] = value;
}

__global__ void depthwise_causal_kernel(
    const float* input,
    size_t positions,
    size_t channels,
    const float* history,
    size_t history_positions,
    const float* weight,
    const float* bias,
    float* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = positions * channels;
    if (item >= total) {
        return;
    }
    const int32_t position = static_cast<int32_t>(item / channels);
    const size_t channel = item % channels;
    float result = bias[channel];
    for (int32_t kernel = 0; kernel < 7; ++kernel) {
        const int32_t source_position = position - (6 - kernel);
        const float source = source_position >= 0
                                 ? input[static_cast<size_t>(source_position) * channels + channel]
                                 : history[(history_positions + source_position) * channels + channel];
        result = fmaf(weight[channel * 7 + kernel], source, result);
    }
    output[item] = result;
}

__global__ void update_causal_history_kernel(
    const float* input,
    size_t positions,
    size_t channels,
    size_t history_positions,
    float* history
) {
    const size_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    for (size_t history_position = 0; history_position < history_positions; ++history_position) {
        const size_t combined_position = history_position + positions;
        history[history_position * channels + channel] =
            combined_position < history_positions
                ? history[combined_position * channels + channel]
                : input[(combined_position - history_positions) * channels + channel];
    }
}

__global__ void layer_norm_kernel(
    const float* input,
    size_t positions,
    size_t channels,
    const float* weight,
    const float* bias,
    float epsilon,
    float* output
) {
    const size_t position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= positions) {
        return;
    }
    const float* row = input + position * channels;
    float mean = 0.0F;
    for (size_t channel = 0; channel < channels; ++channel) {
        mean += row[channel];
    }
    mean /= static_cast<float>(channels);
    float variance = 0.0F;
    for (size_t channel = 0; channel < channels; ++channel) {
        const float centered = row[channel] - mean;
        variance = fmaf(centered, centered, variance);
    }
    const float scale = rsqrtf(variance / static_cast<float>(channels) + epsilon);
    for (size_t channel = 0; channel < channels; ++channel) {
        output[position * channels + channel] =
            (row[channel] - mean) * scale * weight[channel] + bias[channel];
    }
}

__global__ void gelu_kernel(float* values, size_t count) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < count) {
        const float value = values[item];
        values[item] = 0.5F * value * (1.0F + erff(value * 0.7071067811865475F));
    }
}

__global__ void gamma_residual_kernel(
    const float* residual,
    const float* update,
    const float* gamma,
    size_t count,
    size_t channels,
    float* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < count) {
        output[item] = residual[item] + update[item] * gamma[item % channels];
    }
}

__global__ void snake_beta_kernel(
    const float* input,
    const float* alpha_parameter,
    const float* beta_parameter,
    size_t count,
    size_t channels,
    float* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < count) {
        const size_t channel = item % channels;
        const float alpha = expf(alpha_parameter[channel]);
        const float beta = expf(beta_parameter[channel]);
        const float sine = sinf(input[item] * alpha);
        output[item] = input[item] + (sine * sine) / (beta + 1.0e-9F);
    }
}

__global__ void causal_im2col_kernel(
    const float* input,
    size_t positions,
    size_t channels,
    size_t kernel_size,
    size_t dilation,
    const float* history,
    size_t history_positions,
    float* columns
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t width = channels * kernel_size;
    const size_t total = positions * width;
    if (item >= total) {
        return;
    }
    const int32_t position = static_cast<int32_t>(item / width);
    const size_t column = item % width;
    const size_t channel = column / kernel_size;
    const size_t kernel = column % kernel_size;
    const int32_t source_position = position - static_cast<int32_t>(
        (kernel_size - 1 - kernel) * dilation
    );
    columns[item] = source_position >= 0
                        ? input[static_cast<size_t>(source_position) * channels + channel]
                        : history[static_cast<size_t>(
                              static_cast<int32_t>(history_positions) + source_position
                          ) *
                                      channels +
                                  channel];
}

__global__ void transpose_overlap_neural_kernel(
    const float* input,
    size_t input_positions,
    size_t input_channels,
    size_t output_channels,
    size_t stride,
    const float* weight,
    const float* bias,
    const float* prior_tail,
    float* output
) {
    const size_t output_positions = input_positions * stride;
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = output_positions * output_channels;
    if (item >= total) {
        return;
    }
    const size_t output_position = item / output_channels;
    const size_t output_channel = item % output_channels;
    const size_t input_position = output_position / stride;
    const size_t phase = output_position % stride;
    float result = bias[output_channel];
    if (input_position == 0) {
        result += prior_tail[phase * output_channels + output_channel];
    } else {
        for (size_t input_channel = 0; input_channel < input_channels; ++input_channel) {
            const size_t weight_index =
                (input_channel * output_channels + output_channel) * (2 * stride) +
                stride + phase;
            result = fmaf(
                input[(input_position - 1) * input_channels + input_channel],
                weight[weight_index],
                result
            );
        }
    }
    for (size_t input_channel = 0; input_channel < input_channels; ++input_channel) {
        const size_t weight_index =
            (input_channel * output_channels + output_channel) * (2 * stride) + phase;
        result = fmaf(
            input[input_position * input_channels + input_channel],
            weight[weight_index],
            result
        );
    }
    output[item] = result;
}

__global__ void update_transpose_tail_neural_kernel(
    const float* input,
    size_t input_positions,
    size_t input_channels,
    size_t output_channels,
    size_t stride,
    const float* weight,
    float* tail
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = stride * output_channels;
    if (item >= total) {
        return;
    }
    const size_t phase = item / output_channels;
    const size_t output_channel = item % output_channels;
    float result = 0.0F;
    for (size_t input_channel = 0; input_channel < input_channels; ++input_channel) {
        const size_t weight_index =
            (input_channel * output_channels + output_channel) * (2 * stride) +
            stride + phase;
        result = fmaf(
            input[(input_positions - 1) * input_channels + input_channel],
            weight[weight_index],
            result
        );
    }
    tail[item] = result;
}

__global__ void add_residual_kernel(
    const float* residual,
    float* values,
    size_t count
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < count) {
        values[item] += residual[item];
    }
}

__global__ void clamp_waveform_kernel(float* values, size_t count) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < count) {
        values[item] = fminf(1.0F, fmaxf(-1.0F, values[item]));
    }
}

__global__ void waveform_to_pcm_kernel(
    const float* waveform,
    size_t count,
    int16_t* pcm
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < count) {
        const float scaled = roundf(waveform[item] * 32767.0F);
        pcm[item] = static_cast<int16_t>(fminf(32767.0F, fmaxf(-32768.0F, scaled)));
    }
}

__global__ void bf16_to_f32_kernel(
    const __nv_bfloat16* input,
    size_t count,
    float* output
) {
    const size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < count) {
        output[item] = __bfloat162float(input[item]);
    }
}

template <typename T>
cudaError_t allocate_device(T** pointer, size_t elements) noexcept {
    return cudaMalloc(reinterpret_cast<void**>(pointer), elements * sizeof(T));
}

size_t blocks_for(size_t items) noexcept {
    return (items + 255) / 256;
}

}  // namespace

struct DeviceTensor {
    void* data = nullptr;
    uint64_t byte_length = 0;
    uint64_t shape[QWEN3_TTS_CODEC_MAX_TENSOR_RANK]{};
    uint32_t rank = 0;
    uint32_t dtype = 0;
};

struct Qwen3TtsCodecContextV1 {
    int32_t device_index = 0;
    uint32_t ring_slots = QWEN3_TTS_CODEC_RING_SLOTS;
    uint32_t max_packet_frames = QWEN3_TTS_CODEC_MAX_PACKET_FRAMES;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    uint16_t* codec_ring = nullptr;
    int16_t* pcm_ring = nullptr;
    int16_t* host_pcm_ring = nullptr;
    __nv_bfloat16* transformer_kv = nullptr;
    __nv_bfloat16* convolution_history = nullptr;
    int32_t* fixture_history = nullptr;
    int32_t* scratch0 = nullptr;
    int32_t* scratch1 = nullptr;
    int32_t* scratch2 = nullptr;
    int32_t* scratch3 = nullptr;
    int32_t* scratch4 = nullptr;
    int32_t* scratch5 = nullptr;
    int32_t* scratch6 = nullptr;
    float* frontend_quantized = nullptr;
    float* frontend_rvq = nullptr;
    float* frontend_preconv = nullptr;
    float* frontend_history = nullptr;
    float* neural_kv = nullptr;
    float* transformer_packet = nullptr;
    float* transformer_scratch = nullptr;
    float* latent_a = nullptr;
    float* latent_b = nullptr;
    float* latent_expanded = nullptr;
    float* latent_history = nullptr;
    float* decoder_a = nullptr;
    float* decoder_b = nullptr;
    float* decoder_im2col = nullptr;
    float* decoder_history = nullptr;
    uint64_t frame_position = 0;
    uint64_t emitted_samples = 0;
    uint64_t neural_frame_position = 0;
    uint32_t next_ring_slot = 0;
    uint32_t kv_ring_head = 0;
    bool finalized = false;
    std::unordered_map<std::string, DeviceTensor> weights;
    Qwen3TtsCodecModelInfoV1 model_info{};
};

namespace {

const DeviceTensor* find_weight(
    const Qwen3TtsCodecContextV1* context,
    const char* name
) noexcept {
    const auto position = context->weights.find(name);
    return position == context->weights.end() ? nullptr : &position->second;
}

const float* require_f32_weight(
    const Qwen3TtsCodecContextV1* context,
    const char* name,
    char* error,
    size_t error_capacity
) noexcept {
    const DeviceTensor* tensor = find_weight(context, name);
    if (tensor == nullptr) {
        write_error(error, error_capacity, "required decoder weight is missing");
        return nullptr;
    }
    if (tensor->dtype != QWEN3_TTS_CODEC_TENSOR_F32) {
        write_error(
            error,
            error_capacity,
            "neural kernels currently require F32 decoder weights"
        );
        return nullptr;
    }
    return static_cast<const float*>(tensor->data);
}

int32_t launch_linear_vector(
    Qwen3TtsCodecContextV1* context,
    const float* input,
    size_t input_width,
    const char* weight_name,
    const char* bias_name,
    size_t output_width,
    float* output,
    char* error,
    size_t error_capacity
) noexcept {
    const float* weight = require_f32_weight(
        context, weight_name, error, error_capacity
    );
    if (weight == nullptr) {
        return kStatusModel;
    }
    constexpr float kOne = 1.0F;
    constexpr float kZero = 0.0F;
    const cublasStatus_t cublas_status = cublasSgemv(
        context->cublas,
        CUBLAS_OP_T,
        static_cast<int>(input_width),
        static_cast<int>(output_width),
        &kOne,
        weight,
        static_cast<int>(input_width),
        input,
        1,
        &kZero,
        output,
        1
    );
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_error(error, error_capacity, "execute linear projection", cublas_status);
    }
    if (bias_name != nullptr) {
        const float* bias = require_f32_weight(
            context, bias_name, error, error_capacity
        );
        if (bias == nullptr) {
            return kStatusModel;
        }
        add_bias_kernel<<<blocks_for(output_width), 256, 0, context->stream>>>(
            output, bias, output_width, output_width
        );
        const cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch linear bias", status);
        }
    }
    return kStatusOk;
}

int32_t launch_linear_matrix(
    Qwen3TtsCodecContextV1* context,
    const float* input,
    size_t positions,
    size_t input_width,
    const char* weight_name,
    const char* bias_name,
    size_t output_width,
    float* output,
    char* error,
    size_t error_capacity
) noexcept {
    const float* weight = require_f32_weight(
        context, weight_name, error, error_capacity
    );
    if (weight == nullptr) {
        return kStatusModel;
    }
    constexpr float kOne = 1.0F;
    constexpr float kZero = 0.0F;
    const cublasStatus_t cublas_status = cublasSgemm(
        context->cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(output_width),
        static_cast<int>(positions),
        static_cast<int>(input_width),
        &kOne,
        weight,
        static_cast<int>(input_width),
        input,
        static_cast<int>(input_width),
        &kZero,
        output,
        static_cast<int>(output_width)
    );
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_error(error, error_capacity, "execute matrix projection", cublas_status);
    }
    if (bias_name != nullptr) {
        const float* bias = require_f32_weight(
            context, bias_name, error, error_capacity
        );
        if (bias == nullptr) {
            return kStatusModel;
        }
        const size_t count = positions * output_width;
        add_bias_kernel<<<blocks_for(count), 256, 0, context->stream>>>(
            output, bias, count, output_width
        );
        const cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch matrix bias", status);
        }
    }
    return kStatusOk;
}

int32_t run_convnext_stage(
    Qwen3TtsCodecContextV1* context,
    uint32_t stage,
    const float* input,
    size_t positions,
    float* output,
    char* error,
    size_t error_capacity
) noexcept {
    char name[128];
    std::snprintf(
        name,
        sizeof(name),
        "decoder.upsample.%u.1.dwconv.conv.weight",
        stage
    );
    const float* depthwise_weight = require_f32_weight(
        context, name, error, error_capacity
    );
    std::snprintf(
        name,
        sizeof(name),
        "decoder.upsample.%u.1.dwconv.conv.bias",
        stage
    );
    const float* depthwise_bias = require_f32_weight(
        context, name, error, error_capacity
    );
    if (depthwise_weight == nullptr || depthwise_bias == nullptr) {
        return kStatusModel;
    }
    float* history = context->latent_history + static_cast<size_t>(stage) * 6 * 1024;
    depthwise_causal_kernel<<<blocks_for(positions * 1024), 256, 0, context->stream>>>(
        input,
        positions,
        1024,
        history,
        6,
        depthwise_weight,
        depthwise_bias,
        output
    );
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
        update_causal_history_kernel<<<4, 256, 0, context->stream>>>(
            input, positions, 1024, 6, history
        );
        status = cudaGetLastError();
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch ConvNeXt depthwise convolution", status);
    }
    std::snprintf(
        name, sizeof(name), "decoder.upsample.%u.1.norm.weight", stage
    );
    const float* norm_weight = require_f32_weight(
        context, name, error, error_capacity
    );
    std::snprintf(
        name, sizeof(name), "decoder.upsample.%u.1.norm.bias", stage
    );
    const float* norm_bias = require_f32_weight(
        context, name, error, error_capacity
    );
    if (norm_weight == nullptr || norm_bias == nullptr) {
        return kStatusModel;
    }
    layer_norm_kernel<<<blocks_for(positions), 256, 0, context->stream>>>(
        output, positions, 1024, norm_weight, norm_bias, 1.0e-6F, output
    );
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch ConvNeXt layer norm", status);
    }
    char weight_name[128];
    char bias_name[128];
    std::snprintf(
        weight_name,
        sizeof(weight_name),
        "decoder.upsample.%u.1.pwconv1.weight",
        stage
    );
    std::snprintf(
        bias_name,
        sizeof(bias_name),
        "decoder.upsample.%u.1.pwconv1.bias",
        stage
    );
    int32_t result = launch_linear_matrix(
        context,
        output,
        positions,
        1024,
        weight_name,
        bias_name,
        4096,
        context->latent_expanded,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    gelu_kernel<<<blocks_for(positions * 4096), 256, 0, context->stream>>>(
        context->latent_expanded, positions * 4096
    );
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch ConvNeXt GELU", status);
    }
    std::snprintf(
        weight_name,
        sizeof(weight_name),
        "decoder.upsample.%u.1.pwconv2.weight",
        stage
    );
    std::snprintf(
        bias_name,
        sizeof(bias_name),
        "decoder.upsample.%u.1.pwconv2.bias",
        stage
    );
    result = launch_linear_matrix(
        context,
        context->latent_expanded,
        positions,
        4096,
        weight_name,
        bias_name,
        1024,
        output,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    std::snprintf(
        name, sizeof(name), "decoder.upsample.%u.1.gamma", stage
    );
    const float* gamma = require_f32_weight(
        context, name, error, error_capacity
    );
    if (gamma == nullptr) {
        return kStatusModel;
    }
    gamma_residual_kernel<<<blocks_for(positions * 1024), 256, 0, context->stream>>>(
        input, output, gamma, positions * 1024, 1024, output
    );
    status = cudaGetLastError();
    return status == cudaSuccess
               ? kStatusOk
               : cuda_error(error, error_capacity, "launch ConvNeXt residual", status);
}

int32_t run_latent_upsampling(
    Qwen3TtsCodecContextV1* context,
    const float* transformer,
    size_t frames,
    float* stage_one_host,
    float** output,
    size_t* output_positions,
    char* error,
    size_t error_capacity
) noexcept {
    const float* input = transformer;
    size_t positions = frames;
    for (uint32_t stage = 0; stage < 2; ++stage) {
        char weight_name[128];
        char bias_name[128];
        std::snprintf(
            weight_name,
            sizeof(weight_name),
            "decoder.upsample.%u.0.conv.weight",
            stage
        );
        std::snprintf(
            bias_name,
            sizeof(bias_name),
            "decoder.upsample.%u.0.conv.bias",
            stage
        );
        const float* weight = require_f32_weight(
            context, weight_name, error, error_capacity
        );
        const float* bias = require_f32_weight(
            context, bias_name, error, error_capacity
        );
        if (weight == nullptr || bias == nullptr) {
            return kStatusModel;
        }
        latent_transpose_kernel<<<
            blocks_for(positions * 2 * 1024),
            256,
            0,
            context->stream>>>(
            input, positions, weight, bias, context->latent_a
        );
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch latent transposed convolution", status);
        }
        positions *= 2;
        const int32_t result = run_convnext_stage(
            context,
            stage,
            context->latent_a,
            positions,
            context->latent_b,
            error,
            error_capacity
        );
        if (result != kStatusOk) {
            return result;
        }
        if (stage == 0 && stage_one_host != nullptr) {
            status = cudaMemcpyAsync(
                stage_one_host,
                context->latent_b,
                positions * 1024 * sizeof(float),
                cudaMemcpyDeviceToHost,
                context->stream
            );
            if (status == cudaSuccess) {
                status = cudaStreamSynchronize(context->stream);
            }
            if (status != cudaSuccess) {
                return cuda_error(error, error_capacity, "copy latent stage one", status);
            }
        }
        input = context->latent_b;
    }
    *output = context->latent_b;
    *output_positions = positions;
    return kStatusOk;
}

int32_t run_snake_beta(
    Qwen3TtsCodecContextV1* context,
    const float* input,
    size_t positions,
    size_t channels,
    const char* alpha_name,
    const char* beta_name,
    float* output,
    char* error,
    size_t error_capacity
) noexcept {
    const float* alpha = require_f32_weight(
        context, alpha_name, error, error_capacity
    );
    const float* beta = require_f32_weight(
        context, beta_name, error, error_capacity
    );
    if (alpha == nullptr || beta == nullptr) {
        return kStatusModel;
    }
    const size_t count = positions * channels;
    snake_beta_kernel<<<blocks_for(count), 256, 0, context->stream>>>(
        input, alpha, beta, count, channels, output
    );
    const cudaError_t status = cudaGetLastError();
    return status == cudaSuccess
               ? kStatusOk
               : cuda_error(error, error_capacity, "launch SnakeBeta", status);
}

int32_t run_causal_convolution(
    Qwen3TtsCodecContextV1* context,
    const float* input,
    size_t positions,
    size_t input_channels,
    size_t output_channels,
    size_t kernel_size,
    size_t dilation,
    float* history,
    const char* weight_name,
    const char* bias_name,
    float* output,
    char* error,
    size_t error_capacity
) noexcept {
    const size_t history_positions = (kernel_size - 1) * dilation;
    const size_t column_width = input_channels * kernel_size;
    const size_t column_count = positions * column_width;
    causal_im2col_kernel<<<blocks_for(column_count), 256, 0, context->stream>>>(
        input,
        positions,
        input_channels,
        kernel_size,
        dilation,
        history,
        history_positions,
        context->decoder_im2col
    );
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess && history_positions > 0) {
        update_causal_history_kernel<<<
            blocks_for(input_channels),
            256,
            0,
            context->stream>>>(
            input,
            positions,
            input_channels,
            history_positions,
            history
        );
        status = cudaGetLastError();
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch causal convolution input", status);
    }
    return launch_linear_matrix(
        context,
        context->decoder_im2col,
        positions,
        column_width,
        weight_name,
        bias_name,
        output_channels,
        output,
        error,
        error_capacity
    );
}

size_t decoder_transpose_tail_offset(uint32_t stage) noexcept {
    constexpr size_t kSizes[] = {768 * 8, 384 * 5, 192 * 4, 96 * 3};
    size_t offset = kDecoderTransposeHistoryOffset;
    for (uint32_t prior = 0; prior < stage; ++prior) {
        offset += kSizes[prior];
    }
    return offset;
}

size_t decoder_residual_history_offset(
    uint32_t stage,
    uint32_t unit
) noexcept {
    constexpr size_t kChannels[] = {768, 384, 192, 96};
    constexpr size_t kDilations[] = {1, 3, 9};
    size_t offset = kDecoderResidualHistoryOffset;
    for (uint32_t prior_stage = 0; prior_stage < stage; ++prior_stage) {
        offset += kChannels[prior_stage] * 78;
    }
    for (uint32_t prior_unit = 0; prior_unit < unit; ++prior_unit) {
        offset += kChannels[stage] * 6 * kDilations[prior_unit];
    }
    return offset;
}

int32_t run_decoder_transpose(
    Qwen3TtsCodecContextV1* context,
    uint32_t stage,
    const float* input,
    size_t positions,
    size_t input_channels,
    size_t output_channels,
    size_t stride,
    float* output,
    char* error,
    size_t error_capacity
) noexcept {
    char weight_name[128];
    char bias_name[128];
    std::snprintf(
        weight_name,
        sizeof(weight_name),
        "decoder.decoder.%u.block.1.conv.weight",
        stage + 1
    );
    std::snprintf(
        bias_name,
        sizeof(bias_name),
        "decoder.decoder.%u.block.1.conv.bias",
        stage + 1
    );
    const float* weight = require_f32_weight(
        context, weight_name, error, error_capacity
    );
    const float* bias = require_f32_weight(
        context, bias_name, error, error_capacity
    );
    if (weight == nullptr || bias == nullptr) {
        return kStatusModel;
    }
    float* tail = context->decoder_history + decoder_transpose_tail_offset(stage);
    const size_t output_count = positions * stride * output_channels;
    transpose_overlap_neural_kernel<<<
        blocks_for(output_count),
        256,
        0,
        context->stream>>>(
        input,
        positions,
        input_channels,
        output_channels,
        stride,
        weight,
        bias,
        tail,
        output
    );
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
        update_transpose_tail_neural_kernel<<<
            blocks_for(stride * output_channels),
            256,
            0,
            context->stream>>>(
            input,
            positions,
            input_channels,
            output_channels,
            stride,
            weight,
            tail
        );
        status = cudaGetLastError();
    }
    return status == cudaSuccess
               ? kStatusOk
               : cuda_error(error, error_capacity, "launch decoder transposed convolution", status);
}

int32_t run_decoder_residual_unit(
    Qwen3TtsCodecContextV1* context,
    uint32_t stage,
    uint32_t unit,
    const float* input,
    size_t positions,
    size_t channels,
    float* output,
    char* error,
    size_t error_capacity
) noexcept {
    constexpr size_t kDilations[] = {1, 3, 9};
    const uint32_t block_index = unit + 2;
    char alpha_name[128];
    char beta_name[128];
    char weight_name[128];
    char bias_name[128];
    std::snprintf(
        alpha_name,
        sizeof(alpha_name),
        "decoder.decoder.%u.block.%u.act1.alpha",
        stage + 1,
        block_index
    );
    std::snprintf(
        beta_name,
        sizeof(beta_name),
        "decoder.decoder.%u.block.%u.act1.beta",
        stage + 1,
        block_index
    );
    int32_t result = run_snake_beta(
        context,
        input,
        positions,
        channels,
        alpha_name,
        beta_name,
        output,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    std::snprintf(
        weight_name,
        sizeof(weight_name),
        "decoder.decoder.%u.block.%u.conv1.conv.weight",
        stage + 1,
        block_index
    );
    std::snprintf(
        bias_name,
        sizeof(bias_name),
        "decoder.decoder.%u.block.%u.conv1.conv.bias",
        stage + 1,
        block_index
    );
    float* history = context->decoder_history +
                     decoder_residual_history_offset(stage, unit);
    result = run_causal_convolution(
        context,
        output,
        positions,
        channels,
        channels,
        7,
        kDilations[unit],
        history,
        weight_name,
        bias_name,
        output,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    std::snprintf(
        alpha_name,
        sizeof(alpha_name),
        "decoder.decoder.%u.block.%u.act2.alpha",
        stage + 1,
        block_index
    );
    std::snprintf(
        beta_name,
        sizeof(beta_name),
        "decoder.decoder.%u.block.%u.act2.beta",
        stage + 1,
        block_index
    );
    result = run_snake_beta(
        context,
        output,
        positions,
        channels,
        alpha_name,
        beta_name,
        output,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    std::snprintf(
        weight_name,
        sizeof(weight_name),
        "decoder.decoder.%u.block.%u.conv2.conv.weight",
        stage + 1,
        block_index
    );
    std::snprintf(
        bias_name,
        sizeof(bias_name),
        "decoder.decoder.%u.block.%u.conv2.conv.bias",
        stage + 1,
        block_index
    );
    result = run_causal_convolution(
        context,
        output,
        positions,
        channels,
        channels,
        1,
        1,
        history,
        weight_name,
        bias_name,
        output,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    add_residual_kernel<<<blocks_for(positions * channels), 256, 0, context->stream>>>(
        input, output, positions * channels
    );
    const cudaError_t status = cudaGetLastError();
    return status == cudaSuccess
               ? kStatusOk
               : cuda_error(error, error_capacity, "launch decoder residual", status);
}

int32_t copy_decoder_checkpoint(
    Qwen3TtsCodecContextV1* context,
    const float* input,
    size_t positions,
    size_t channels,
    float* output,
    size_t output_capacity,
    char* error,
    size_t error_capacity
) noexcept {
    if (output == nullptr || output_capacity < positions * channels) {
        write_error(error, error_capacity, "decoder checkpoint capacity is too small");
        return kStatusInvalidArgument;
    }
    std::vector<float> row_major(positions * channels);
    cudaError_t status = cudaMemcpyAsync(
        row_major.data(),
        input,
        row_major.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        context->stream
    );
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context->stream);
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "copy decoder checkpoint", status);
    }
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t position = 0; position < positions; ++position) {
            output[channel * positions + position] =
                row_major[position * channels + channel];
        }
    }
    return kStatusOk;
}

int32_t run_waveform_decoder(
    Qwen3TtsCodecContextV1* context,
    const float* latent,
    size_t latent_positions,
    uint32_t checkpoint,
    float* checkpoint_output,
    size_t checkpoint_capacity,
    float** waveform,
    size_t* waveform_positions,
    char* error,
    size_t error_capacity
) noexcept {
    int32_t result = run_causal_convolution(
        context,
        latent,
        latent_positions,
        1024,
        1536,
        7,
        1,
        context->decoder_history + kDecoderPreconvHistoryOffset,
        "decoder.decoder.0.conv.weight",
        "decoder.decoder.0.conv.bias",
        context->decoder_a,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    size_t positions = latent_positions;
    size_t channels = 1536;
    float* current = context->decoder_a;
    float* other = context->decoder_b;
    if (checkpoint == 6) {
        return copy_decoder_checkpoint(
            context,
            current,
            positions,
            channels,
            checkpoint_output,
            checkpoint_capacity,
            error,
            error_capacity
        );
    }

    constexpr size_t kStrides[] = {8, 5, 4, 3};
    constexpr size_t kOutputChannels[] = {768, 384, 192, 96};
    for (uint32_t stage = 0; stage < 4; ++stage) {
        char alpha_name[128];
        char beta_name[128];
        std::snprintf(
            alpha_name,
            sizeof(alpha_name),
            "decoder.decoder.%u.block.0.alpha",
            stage + 1
        );
        std::snprintf(
            beta_name,
            sizeof(beta_name),
            "decoder.decoder.%u.block.0.beta",
            stage + 1
        );
        result = run_snake_beta(
            context,
            current,
            positions,
            channels,
            alpha_name,
            beta_name,
            other,
            error,
            error_capacity
        );
        if (result != kStatusOk) {
            return result;
        }
        std::swap(current, other);
        result = run_decoder_transpose(
            context,
            stage,
            current,
            positions,
            channels,
            kOutputChannels[stage],
            kStrides[stage],
            other,
            error,
            error_capacity
        );
        if (result != kStatusOk) {
            return result;
        }
        std::swap(current, other);
        positions *= kStrides[stage];
        channels = kOutputChannels[stage];
        for (uint32_t unit = 0; unit < 3; ++unit) {
            result = run_decoder_residual_unit(
                context,
                stage,
                unit,
                current,
                positions,
                channels,
                other,
                error,
                error_capacity
            );
            if (result != kStatusOk) {
                return result;
            }
            std::swap(current, other);
        }
        if (checkpoint == stage + 7) {
            return copy_decoder_checkpoint(
                context,
                current,
                positions,
                channels,
                checkpoint_output,
                checkpoint_capacity,
                error,
                error_capacity
            );
        }
    }

    result = run_snake_beta(
        context,
        current,
        positions,
        96,
        "decoder.decoder.5.alpha",
        "decoder.decoder.5.beta",
        other,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    std::swap(current, other);
    if (checkpoint == 11) {
        return copy_decoder_checkpoint(
            context,
            current,
            positions,
            96,
            checkpoint_output,
            checkpoint_capacity,
            error,
            error_capacity
        );
    }
    result = run_causal_convolution(
        context,
        current,
        positions,
        96,
        1,
        7,
        1,
        context->decoder_history + kDecoderFinalHistoryOffset,
        "decoder.decoder.6.conv.weight",
        "decoder.decoder.6.conv.bias",
        other,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    std::swap(current, other);
    if (checkpoint == 12) {
        return copy_decoder_checkpoint(
            context,
            current,
            positions,
            1,
            checkpoint_output,
            checkpoint_capacity,
            error,
            error_capacity
        );
    }
    clamp_waveform_kernel<<<blocks_for(positions), 256, 0, context->stream>>>(
        current, positions
    );
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch waveform clamp", status);
    }
    if (checkpoint == 13) {
        return copy_decoder_checkpoint(
            context,
            current,
            positions,
            1,
            checkpoint_output,
            checkpoint_capacity,
            error,
            error_capacity
        );
    }
    *waveform = current;
    *waveform_positions = positions;
    return kStatusOk;
}

int32_t run_transformer_frame(
    Qwen3TtsCodecContextV1* context,
    const float* input,
    uint64_t position,
    float* output,
    char* error,
    size_t error_capacity
) noexcept {
    float* hidden_a = context->transformer_scratch;
    float* hidden_b = hidden_a + 512;
    float* normalized = hidden_b + 512;
    float* query = normalized + 512;
    float* key = query + 1024;
    float* value = key + 1024;
    float* attention = value + 1024;
    float* gate = attention + 1024;
    float* up = gate + 1024;
    float* projection = up + 1024;

    int32_t result = launch_linear_vector(
        context,
        input,
        1024,
        "decoder.pre_transformer.input_proj.weight",
        "decoder.pre_transformer.input_proj.bias",
        512,
        hidden_a,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }

    for (uint32_t layer = 0; layer < 8; ++layer) {
        char name[128];
        std::snprintf(
            name,
            sizeof(name),
            "decoder.pre_transformer.layers.%u.input_layernorm.weight",
            layer
        );
        const float* norm_weight = require_f32_weight(
            context, name, error, error_capacity
        );
        if (norm_weight == nullptr) {
            return kStatusModel;
        }
        rms_norm_kernel<<<1, 1, 0, context->stream>>>(
            hidden_a, norm_weight, 512, 1.0e-5F, normalized
        );
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch attention RMS norm", status);
        }

        const char* projections[] = {"q_proj", "k_proj", "v_proj"};
        float* projection_outputs[] = {query, key, value};
        for (size_t index = 0; index < 3; ++index) {
            std::snprintf(
                name,
                sizeof(name),
                "decoder.pre_transformer.layers.%u.self_attn.%s.weight",
                layer,
                projections[index]
            );
            result = launch_linear_vector(
                context,
                normalized,
                512,
                name,
                nullptr,
                1024,
                projection_outputs[index],
                error,
                error_capacity
            );
            if (result != kStatusOk) {
                return result;
            }
        }
        apply_rope_kernel<<<4, 256, 0, context->stream>>>(query, key, position);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch decoder RoPE", status);
        }
        sliding_attention_kernel<<<1, 16, 0, context->stream>>>(
            query,
            key,
            value,
            layer,
            position,
            context->neural_kv,
            attention
        );
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch sliding attention", status);
        }
        std::snprintf(
            name,
            sizeof(name),
            "decoder.pre_transformer.layers.%u.self_attn.o_proj.weight",
            layer
        );
        result = launch_linear_vector(
            context,
            attention,
            1024,
            name,
            nullptr,
            512,
            projection,
            error,
            error_capacity
        );
        if (result != kStatusOk) {
            return result;
        }
        std::snprintf(
            name,
            sizeof(name),
            "decoder.pre_transformer.layers.%u.self_attn_layer_scale.scale",
            layer
        );
        const float* attention_scale = require_f32_weight(
            context, name, error, error_capacity
        );
        if (attention_scale == nullptr) {
            return kStatusModel;
        }
        scaled_residual_kernel<<<2, 256, 0, context->stream>>>(
            hidden_a, projection, attention_scale, 512, hidden_b
        );
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch attention residual", status);
        }

        std::snprintf(
            name,
            sizeof(name),
            "decoder.pre_transformer.layers.%u.post_attention_layernorm.weight",
            layer
        );
        norm_weight = require_f32_weight(context, name, error, error_capacity);
        if (norm_weight == nullptr) {
            return kStatusModel;
        }
        rms_norm_kernel<<<1, 1, 0, context->stream>>>(
            hidden_b, norm_weight, 512, 1.0e-5F, normalized
        );
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch MLP RMS norm", status);
        }
        const char* mlp_projections[] = {"gate_proj", "up_proj"};
        float* mlp_outputs[] = {gate, up};
        for (size_t index = 0; index < 2; ++index) {
            std::snprintf(
                name,
                sizeof(name),
                "decoder.pre_transformer.layers.%u.mlp.%s.weight",
                layer,
                mlp_projections[index]
            );
            result = launch_linear_vector(
                context,
                normalized,
                512,
                name,
                nullptr,
                1024,
                mlp_outputs[index],
                error,
                error_capacity
            );
            if (result != kStatusOk) {
                return result;
            }
        }
        silu_product_kernel<<<4, 256, 0, context->stream>>>(
            gate, up, 1024, value
        );
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch gated SiLU", status);
        }
        std::snprintf(
            name,
            sizeof(name),
            "decoder.pre_transformer.layers.%u.mlp.down_proj.weight",
            layer
        );
        result = launch_linear_vector(
            context,
            value,
            1024,
            name,
            nullptr,
            512,
            projection,
            error,
            error_capacity
        );
        if (result != kStatusOk) {
            return result;
        }
        std::snprintf(
            name,
            sizeof(name),
            "decoder.pre_transformer.layers.%u.mlp_layer_scale.scale",
            layer
        );
        const float* mlp_scale = require_f32_weight(
            context, name, error, error_capacity
        );
        if (mlp_scale == nullptr) {
            return kStatusModel;
        }
        scaled_residual_kernel<<<2, 256, 0, context->stream>>>(
            hidden_b, projection, mlp_scale, 512, hidden_a
        );
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch MLP residual", status);
        }
    }

    const float* final_norm = require_f32_weight(
        context, "decoder.pre_transformer.norm.weight", error, error_capacity
    );
    if (final_norm == nullptr) {
        return kStatusModel;
    }
    rms_norm_kernel<<<1, 1, 0, context->stream>>>(
        hidden_a, final_norm, 512, 1.0e-5F, normalized
    );
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch final transformer norm", status);
    }
    return launch_linear_vector(
        context,
        normalized,
        512,
        "decoder.pre_transformer.output_proj.weight",
        "decoder.pre_transformer.output_proj.bias",
        1024,
        output,
        error,
        error_capacity
    );
}

void release_model(Qwen3TtsCodecContextV1* context) noexcept {
    if (context == nullptr) {
        return;
    }
    for (auto& [name, tensor] : context->weights) {
        (void)name;
        cudaFree(tensor.data);
        tensor.data = nullptr;
    }
    context->weights.clear();
    context->model_info = Qwen3TtsCodecModelInfoV1{};
}

void release_context(Qwen3TtsCodecContextV1* context) noexcept {
    if (context == nullptr) {
        return;
    }
    if (context->device_index >= 0) {
        cudaSetDevice(context->device_index);
    }
    if (context->stream != nullptr) {
        cudaStreamSynchronize(context->stream);
    }
    release_model(context);
    cudaFree(context->scratch6);
    cudaFree(context->scratch5);
    cudaFree(context->scratch4);
    cudaFree(context->scratch3);
    cudaFree(context->scratch2);
    cudaFree(context->scratch1);
    cudaFree(context->scratch0);
    cudaFree(context->frontend_history);
    cudaFree(context->frontend_preconv);
    cudaFree(context->frontend_rvq);
    cudaFree(context->frontend_quantized);
    cudaFree(context->transformer_scratch);
    cudaFree(context->transformer_packet);
    cudaFree(context->neural_kv);
    cudaFree(context->latent_history);
    cudaFree(context->latent_expanded);
    cudaFree(context->latent_b);
    cudaFree(context->latent_a);
    cudaFree(context->decoder_history);
    cudaFree(context->decoder_im2col);
    cudaFree(context->decoder_b);
    cudaFree(context->decoder_a);
    cudaFree(context->fixture_history);
    cudaFree(context->convolution_history);
    cudaFree(context->transformer_kv);
    cudaFree(context->pcm_ring);
    cudaFree(context->codec_ring);
    if (context->host_pcm_ring != nullptr) {
        cudaFreeHost(context->host_pcm_ring);
    }
    if (context->start_event != nullptr) {
        cudaEventDestroy(context->start_event);
    }
    if (context->stop_event != nullptr) {
        cudaEventDestroy(context->stop_event);
    }
    if (context->cublas != nullptr) {
        cublasDestroy(context->cublas);
    }
    if (context->stream != nullptr) {
        cudaStreamDestroy(context->stream);
    }
    delete context;
}

int32_t reset_device_state(
    Qwen3TtsCodecContextV1* context,
    char* error,
    size_t error_capacity
) noexcept {
    cudaError_t status = cudaMemsetAsync(
        context->codec_ring, 0, kCodecRingBytes, context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "clear codec ring", status);
    }
    status = cudaMemsetAsync(
        context->pcm_ring, 0, kPcmRingBytes, context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "clear PCM ring", status);
    }
    status = cudaMemsetAsync(
        context->transformer_kv, 0, kTransformerKvBytes, context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "clear transformer KV", status);
    }
    status = cudaMemsetAsync(
        context->convolution_history,
        0,
        kConvolutionHistoryBytes,
        context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(
            error, error_capacity, "clear convolution history", status
        );
    }
    status = cudaMemsetAsync(
        context->fixture_history, 0, kFixtureHistoryBytes, context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "clear fixture history", status);
    }
    status = cudaMemsetAsync(
        context->frontend_history,
        0,
        kFrontendHistoryElements * sizeof(float),
        context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "clear neural frontend history", status);
    }
    status = cudaMemsetAsync(
        context->neural_kv, 0, kNeuralKvElements * sizeof(float), context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "clear neural transformer KV", status);
    }
    status = cudaMemsetAsync(
        context->latent_history,
        0,
        kLatentHistoryElements * sizeof(float),
        context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "clear latent ConvNeXt history", status);
    }
    status = cudaMemsetAsync(
        context->decoder_history,
        0,
        kDecoderHistoryElements * sizeof(float),
        context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "clear waveform decoder history", status);
    }
    status = cudaStreamSynchronize(context->stream);
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "synchronize reset", status);
    }
    std::memset(context->host_pcm_ring, 0, kPcmRingBytes);
    context->frame_position = 0;
    context->emitted_samples = 0;
    context->neural_frame_position = 0;
    context->next_ring_slot = 0;
    context->kv_ring_head = 0;
    context->finalized = false;
    return kStatusOk;
}

int32_t launch_repeat(
    const int32_t* input,
    size_t input_count,
    uint32_t repeat,
    int32_t* output,
    cudaStream_t stream,
    char* error,
    size_t error_capacity
) noexcept {
    const size_t output_count = input_count * repeat;
    repeat_kernel<<<blocks_for(output_count), 256, 0, stream>>>(
        input, input_count, repeat, output
    );
    const cudaError_t status = cudaGetLastError();
    return status == cudaSuccess
               ? kStatusOk
               : cuda_error(error, error_capacity, "launch repeat kernel", status);
}

int32_t launch_transpose(
    const int32_t* input,
    size_t input_count,
    uint32_t stride,
    int32_t* tail,
    int32_t* output,
    cudaStream_t stream,
    char* error,
    size_t error_capacity
) noexcept {
    const size_t output_count = input_count * stride;
    transpose_overlap_kernel<<<blocks_for(output_count), 256, 0, stream>>>(
        input, input_count, stride, tail, output
    );
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(
            error, error_capacity, "launch transpose overlap kernel", status
        );
    }
    update_tail_kernel<<<1, 32, 0, stream>>>(
        input, input_count, stride, tail
    );
    status = cudaGetLastError();
    return status == cudaSuccess
               ? kStatusOk
               : cuda_error(error, error_capacity, "launch tail update", status);
}

}  // namespace

extern "C" QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_abi_version_v1(void) {
    return QWEN3_TTS_CODEC_ABI_VERSION_V1;
}

extern "C" QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_create_v1(
    const Qwen3TtsCodecConfigV1* config,
    Qwen3TtsCodecContextV1** output,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (config == nullptr || output == nullptr) {
        write_error(error, error_capacity, "config and output are required");
        return kStatusInvalidArgument;
    }
    *output = nullptr;
    if (config->ring_slots != QWEN3_TTS_CODEC_RING_SLOTS ||
        config->max_packet_frames != QWEN3_TTS_CODEC_MAX_PACKET_FRAMES ||
        config->reserved != 0 || config->device_index < 0) {
        write_error(
            error,
            error_capacity,
            "config must request device >= 0, three ring slots, four frames, and reserved=0"
        );
        return kStatusInvalidArgument;
    }

    int32_t device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "query CUDA devices", status);
    }
    if (config->device_index >= device_count) {
        write_error(error, error_capacity, "CUDA device index is out of range");
        return kStatusInvalidArgument;
    }
    status = cudaSetDevice(config->device_index);
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "select CUDA device", status);
    }

    auto* context = new (std::nothrow) Qwen3TtsCodecContextV1{};
    if (context == nullptr) {
        write_error(error, error_capacity, "allocate codec context");
        return kStatusAllocation;
    }
    context->device_index = config->device_index;

    status = cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking);
    if (status == cudaSuccess && cublasCreate(&context->cublas) != CUBLAS_STATUS_SUCCESS) {
        status = cudaErrorInitializationError;
    }
    if (status == cudaSuccess &&
        cublasSetStream(context->cublas, context->stream) != CUBLAS_STATUS_SUCCESS) {
        status = cudaErrorInitializationError;
    }
    if (status == cudaSuccess) {
        status = cudaEventCreate(&context->start_event);
    }
    if (status == cudaSuccess) {
        status = cudaEventCreate(&context->stop_event);
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->codec_ring, kCodecRingElements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->pcm_ring, kPcmRingElements);
    }
    if (status == cudaSuccess) {
        status = cudaHostAlloc(
            reinterpret_cast<void**>(&context->host_pcm_ring),
            kPcmRingBytes,
            cudaHostAllocPortable
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->transformer_kv, kTransformerKvElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->convolution_history, kConvolutionHistoryElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->fixture_history, kFixtureHistoryElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->scratch0, kScratch0Elements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->scratch1, kScratch1Elements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->scratch2, kScratch2Elements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->scratch3, kScratch3Elements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->scratch4, kScratch4Elements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->scratch5, kScratch5Elements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->scratch6, kScratch6Elements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->frontend_quantized, kFrontendQuantizedElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->frontend_rvq, kFrontendRvqElements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->frontend_preconv, kFrontendPreconvElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->frontend_history, kFrontendHistoryElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->neural_kv, kNeuralKvElements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->transformer_packet, kTransformerPacketElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->transformer_scratch, kTransformerScratchElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->latent_a, kLatentElements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->latent_b, kLatentElements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->latent_expanded, kLatentExpandedElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(&context->latent_history, kLatentHistoryElements);
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->decoder_a, kDecoderMaxActivationElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->decoder_b, kDecoderMaxActivationElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->decoder_im2col, kDecoderMaxIm2colElements
        );
    }
    if (status == cudaSuccess) {
        status = allocate_device(
            &context->decoder_history, kDecoderHistoryElements
        );
    }
    if (status != cudaSuccess) {
        const int32_t result =
            cuda_error(error, error_capacity, "allocate codec state", status);
        release_context(context);
        return result;
    }

    const int32_t reset_status =
        reset_device_state(context, error, error_capacity);
    if (reset_status != kStatusOk) {
        release_context(context);
        return reset_status;
    }
    *output = context;
    return kStatusOk;
}

extern "C" QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_destroy_v1(
    Qwen3TtsCodecContextV1* context,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    release_context(context);
    return kStatusOk;
}

extern "C" QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_reset_v1(
    Qwen3TtsCodecContextV1* context,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr) {
        write_error(error, error_capacity, "context is required");
        return kStatusInvalidArgument;
    }
    const cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "select CUDA device", status);
    }
    return reset_device_state(context, error, error_capacity);
}

extern "C" QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_state_info_v1(
    const Qwen3TtsCodecContextV1* context,
    Qwen3TtsCodecStateInfoV1* output,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr || output == nullptr) {
        write_error(error, error_capacity, "context and output are required");
        return kStatusInvalidArgument;
    }
    const uint64_t device_bytes =
        kTransformerKvBytes + kConvolutionHistoryBytes + kCodecRingBytes +
        kPcmRingBytes + kFixtureHistoryBytes + kScratchBytes +
        kFrontendDeviceBytes + kTransformerDeviceBytes +
        kLatentDeviceBytes + kDecoderDeviceBytes +
        context->model_info.device_bytes;
    *output = Qwen3TtsCodecStateInfoV1{
        context->frame_position,
        context->emitted_samples,
        device_bytes,
        kPcmRingBytes,
        kTransformerKvBytes + kNeuralTransformerKvBytes,
        kConvolutionHistoryBytes + kNeuralConvolutionHistoryBytes,
        kCodecRingBytes,
        kPcmRingBytes,
        context->kv_ring_head,
        context->next_ring_slot,
        context->ring_slots,
        context->max_packet_frames,
    };
    return kStatusOk;
}

extern "C" QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_load_model_v1(
    Qwen3TtsCodecContextV1* context,
    const Qwen3TtsCodecWeightProviderV1* provider,
    Qwen3TtsCodecModelInfoV1* output,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr || provider == nullptr || output == nullptr ||
        provider->tensor_at == nullptr) {
        write_error(
            error,
            error_capacity,
            "context, weight provider, output, and tensor callback are required"
        );
        return kStatusInvalidArgument;
    }
    if (provider->abi_version != QWEN3_TTS_CODEC_ABI_VERSION_V1 ||
        provider->reserved != 0) {
        write_error(error, error_capacity, "unsupported weight-provider ABI");
        return kStatusInvalidArgument;
    }
    if (!context->weights.empty()) {
        write_error(error, error_capacity, "model is already loaded");
        return kStatusState;
    }
    cudaError_t cuda_status = cudaSetDevice(context->device_index);
    if (cuda_status != cudaSuccess) {
        return cuda_error(error, error_capacity, "select CUDA device", cuda_status);
    }

    struct Bf16StagingGuard {
        void* data = nullptr;
        ~Bf16StagingGuard() {
            cudaFree(data);
        }
    } bf16_staging;

    Qwen3TtsCodecModelInfoV1 info{};
    for (uint64_t index = 0; index < provider->tensor_count; ++index) {
        const char* name = nullptr;
        Qwen3TtsCodecTensorViewV1 source{};
        const int32_t provider_status = provider->tensor_at(
            provider->user_data,
            index,
            &name,
            &source,
            error,
            error_capacity
        );
        if (provider_status != kStatusOk) {
            release_model(context);
            if (error == nullptr || error_capacity == 0 || error[0] == '\0') {
                write_error(error, error_capacity, "weight provider rejected tensor index");
            }
            return kStatusModel;
        }
        if (name == nullptr || std::strncmp(name, "decoder.", 8) != 0) {
            continue;
        }
        if (source.data == nullptr || source.byte_length == 0 || source.rank == 0 ||
            source.rank > QWEN3_TTS_CODEC_MAX_TENSOR_RANK ||
            (source.dtype != QWEN3_TTS_CODEC_TENSOR_F32 &&
             source.dtype != QWEN3_TTS_CODEC_TENSOR_BF16)) {
            release_model(context);
            write_error(error, error_capacity, "decoder tensor metadata is invalid");
            return kStatusModel;
        }
        uint64_t element_count = 1;
        for (uint32_t dimension = 0; dimension < source.rank; ++dimension) {
            if (source.shape[dimension] == 0 ||
                element_count > UINT64_MAX / source.shape[dimension]) {
                release_model(context);
                write_error(error, error_capacity, "decoder tensor shape is invalid");
                return kStatusModel;
            }
            element_count *= source.shape[dimension];
        }
        const uint64_t scalar_bytes =
            source.dtype == QWEN3_TTS_CODEC_TENSOR_F32 ? 4 : 2;
        if (element_count > UINT64_MAX / scalar_bytes ||
            element_count * scalar_bytes != source.byte_length) {
            release_model(context);
            write_error(error, error_capacity, "decoder tensor byte length is invalid");
            return kStatusModel;
        }
        if (element_count > UINT64_MAX / sizeof(float)) {
            release_model(context);
            write_error(error, error_capacity, "decoder tensor is too large for F32 execution");
            return kStatusModel;
        }
        const uint64_t device_byte_length = element_count * sizeof(float);
        DeviceTensor tensor{};
        tensor.byte_length = device_byte_length;
        tensor.rank = source.rank;
        tensor.dtype = QWEN3_TTS_CODEC_TENSOR_F32;
        std::memcpy(tensor.shape, source.shape, sizeof(tensor.shape));
        cuda_status = cudaMalloc(
            &tensor.data, static_cast<size_t>(device_byte_length)
        );
        if (cuda_status == cudaSuccess &&
            source.dtype == QWEN3_TTS_CODEC_TENSOR_F32) {
            cuda_status = cudaMemcpyAsync(
                tensor.data,
                source.data,
                static_cast<size_t>(source.byte_length),
                cudaMemcpyHostToDevice,
                context->stream
            );
        }
        if (cuda_status == cudaSuccess &&
            source.dtype == QWEN3_TTS_CODEC_TENSOR_BF16) {
            if (bf16_staging.data == nullptr) {
                cuda_status = cudaMalloc(
                    &bf16_staging.data, kBf16UploadStagingBytes
                );
            }
            constexpr size_t kStagingElements =
                kBf16UploadStagingBytes / sizeof(__nv_bfloat16);
            const auto* source_bytes =
                static_cast<const uint8_t*>(source.data);
            for (uint64_t offset = 0;
                 cuda_status == cudaSuccess && offset < element_count;
                 offset += kStagingElements) {
                const size_t count = static_cast<size_t>(
                    std::min<uint64_t>(kStagingElements, element_count - offset)
                );
                cuda_status = cudaMemcpyAsync(
                    bf16_staging.data,
                    source_bytes + offset * sizeof(__nv_bfloat16),
                    count * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice,
                    context->stream
                );
                if (cuda_status == cudaSuccess) {
                    bf16_to_f32_kernel<<<
                        blocks_for(count), 256, 0, context->stream>>>(
                        static_cast<const __nv_bfloat16*>(bf16_staging.data),
                        count,
                        static_cast<float*>(tensor.data) + offset
                    );
                    cuda_status = cudaGetLastError();
                }
            }
        }
        if (cuda_status != cudaSuccess) {
            cudaFree(tensor.data);
            release_model(context);
            return cuda_error(error, error_capacity, "upload decoder tensor", cuda_status);
        }
        const auto [position, inserted] =
            context->weights.emplace(std::string(name), tensor);
        (void)position;
        if (!inserted) {
            cudaFree(tensor.data);
            release_model(context);
            write_error(error, error_capacity, "duplicate decoder tensor name");
            return kStatusModel;
        }
        info.source_bytes += source.byte_length;
        info.device_bytes += device_byte_length;
        info.parameter_count += element_count;
        info.tensor_count += 1;
        if (source.dtype == QWEN3_TTS_CODEC_TENSOR_F32) {
            info.source_dtype_f32_count += 1;
        } else {
            info.source_dtype_bf16_count += 1;
        }
    }
    cuda_status = cudaStreamSynchronize(context->stream);
    if (cuda_status != cudaSuccess) {
        release_model(context);
        return cuda_error(error, error_capacity, "finish decoder upload", cuda_status);
    }
    if (info.tensor_count != kExpectedDecoderTensors) {
        release_model(context);
        write_error(error, error_capacity, "model does not contain exactly 271 decoder tensors");
        return kStatusModel;
    }
    const char* required[] = {
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        "decoder.pre_transformer.layers.0.self_attn.q_proj.weight",
        "decoder.decoder.6.conv.weight",
    };
    for (const char* name : required) {
        if (context->weights.find(name) == context->weights.end()) {
            release_model(context);
            write_error(error, error_capacity, "model is missing a required decoder tensor");
            return kStatusModel;
        }
    }
    info.loaded = 1;
    context->model_info = info;
    *output = info;
    return kStatusOk;
}

extern "C" QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_model_info_v1(
    const Qwen3TtsCodecContextV1* context,
    Qwen3TtsCodecModelInfoV1* output,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr || output == nullptr) {
        write_error(error, error_capacity, "context and output are required");
        return kStatusInvalidArgument;
    }
    *output = context->model_info;
    return kStatusOk;
}

namespace {

int32_t run_frontend_device(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint16_t* device_codes,
    uint32_t frame_count,
    char* error,
    size_t error_capacity
) noexcept {
    cudaError_t status = cudaMemcpyAsync(
        device_codes,
        codec_frames,
        static_cast<size_t>(frame_count) * QWEN3_TTS_CODEC_CODEBOOKS * sizeof(uint16_t),
        cudaMemcpyHostToDevice,
        context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "upload frontend codes", status);
    }

    float* semantic = context->frontend_quantized;
    float* acoustic = context->frontend_quantized +
                      QWEN3_TTS_CODEC_MAX_PACKET_FRAMES * 256;
    const float* first_embedding = require_f32_weight(
        context,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        error,
        error_capacity
    );
    const float* first_usage = require_f32_weight(
        context,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
        error,
        error_capacity
    );
    const float* first_projection = require_f32_weight(
        context,
        "decoder.quantizer.rvq_first.output_proj.weight",
        error,
        error_capacity
    );
    const float* rest_projection = require_f32_weight(
        context,
        "decoder.quantizer.rvq_rest.output_proj.weight",
        error,
        error_capacity
    );
    if (first_embedding == nullptr || first_usage == nullptr ||
        first_projection == nullptr || rest_projection == nullptr) {
        return kStatusModel;
    }
    gather_codebook_kernel<<<
        blocks_for(static_cast<size_t>(frame_count) * 256),
        256,
        0,
        context->stream>>>(
        device_codes,
        frame_count,
        0,
        first_embedding,
        first_usage,
        semantic,
        0
    );
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch semantic codebook", status);
    }
    for (uint32_t codebook = 1; codebook < QWEN3_TTS_CODEC_CODEBOOKS; ++codebook) {
        char embedding_name[128];
        char usage_name[128];
        std::snprintf(
            embedding_name,
            sizeof(embedding_name),
            "decoder.quantizer.rvq_rest.vq.layers.%u._codebook.embedding_sum",
            codebook - 1
        );
        std::snprintf(
            usage_name,
            sizeof(usage_name),
            "decoder.quantizer.rvq_rest.vq.layers.%u._codebook.cluster_usage",
            codebook - 1
        );
        const float* embedding = require_f32_weight(
            context, embedding_name, error, error_capacity
        );
        const float* usage = require_f32_weight(
            context, usage_name, error, error_capacity
        );
        if (embedding == nullptr || usage == nullptr) {
            return kStatusModel;
        }
        gather_codebook_kernel<<<
            blocks_for(static_cast<size_t>(frame_count) * 256),
            256,
            0,
            context->stream>>>(
            device_codes,
            frame_count,
            codebook,
            embedding,
            usage,
            acoustic,
            codebook == 1 ? 0 : 1
        );
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch acoustic codebook", status);
        }
    }
    constexpr float kOne = 1.0F;
    constexpr float kZero = 0.0F;
    cublasStatus_t cublas_status = cublasSgemm(
        context->cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        512,
        static_cast<int>(frame_count),
        256,
        &kOne,
        first_projection,
        256,
        semantic,
        256,
        &kZero,
        context->frontend_rvq,
        512
    );
    float* rest_projected = context->frontend_preconv;
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
        cublas_status = cublasSgemm(
            context->cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            512,
            static_cast<int>(frame_count),
            256,
            &kOne,
            rest_projection,
            256,
            acoustic,
            256,
            &kZero,
            rest_projected,
            512
        );
    }
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_error(
            error, error_capacity, "project split RVQ", cublas_status
        );
    }
    add_rvq_projection_kernel<<<
        blocks_for(static_cast<size_t>(frame_count) * 512),
        256,
        0,
        context->stream>>>(rest_projected, frame_count, context->frontend_rvq);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "combine split RVQ", status);
    }
    const float* preconv_weight = require_f32_weight(
        context, "decoder.pre_conv.conv.weight", error, error_capacity
    );
    const float* preconv_bias = require_f32_weight(
        context, "decoder.pre_conv.conv.bias", error, error_capacity
    );
    if (preconv_weight == nullptr || preconv_bias == nullptr) {
        return kStatusModel;
    }
    causal_preconv_kernel<<<
        blocks_for(static_cast<size_t>(frame_count) * 1024),
        256,
        0,
        context->stream>>>(
        context->frontend_rvq,
        context->frontend_history,
        frame_count,
        preconv_weight,
        preconv_bias,
        context->frontend_preconv
    );
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        update_frontend_history_kernel<<<2, 256, 0, context->stream>>>(
            context->frontend_rvq, frame_count, context->frontend_history
        );
        status = cudaGetLastError();
    }
    return status == cudaSuccess
               ? kStatusOk
               : cuda_error(error, error_capacity, "launch causal pre-convolution", status);
}

}  // namespace

extern "C" QWEN3_TTS_CODEC_API int32_t
qwen3_tts_codec_debug_frontend_packet_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    float* rvq_output,
    size_t rvq_capacity,
    float* preconv_output,
    size_t preconv_capacity,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr || codec_frames == nullptr || rvq_output == nullptr ||
        preconv_output == nullptr) {
        write_error(error, error_capacity, "context, codes, and checkpoint outputs are required");
        return kStatusInvalidArgument;
    }
    if (frame_count == 0 || frame_count > QWEN3_TTS_CODEC_MAX_PACKET_FRAMES ||
        rvq_capacity < static_cast<size_t>(frame_count) * 512 ||
        preconv_capacity < static_cast<size_t>(frame_count) * 1024) {
        write_error(error, error_capacity, "invalid frontend packet or output capacity");
        return kStatusInvalidArgument;
    }
    if (context->model_info.loaded == 0) {
        write_error(error, error_capacity, "decoder model is not loaded");
        return kStatusState;
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "select CUDA device", status);
    }
    uint16_t* device_codes = context->codec_ring;
    status = cudaMemcpyAsync(
        device_codes,
        codec_frames,
        static_cast<size_t>(frame_count) * QWEN3_TTS_CODEC_CODEBOOKS * sizeof(uint16_t),
        cudaMemcpyHostToDevice,
        context->stream
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "upload frontend codes", status);
    }

    float* semantic = context->frontend_quantized;
    float* acoustic = context->frontend_quantized +
                      QWEN3_TTS_CODEC_MAX_PACKET_FRAMES * 256;
    const float* first_embedding = require_f32_weight(
        context,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        error,
        error_capacity
    );
    const float* first_usage = require_f32_weight(
        context,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
        error,
        error_capacity
    );
    const float* first_projection = require_f32_weight(
        context,
        "decoder.quantizer.rvq_first.output_proj.weight",
        error,
        error_capacity
    );
    const float* rest_projection = require_f32_weight(
        context,
        "decoder.quantizer.rvq_rest.output_proj.weight",
        error,
        error_capacity
    );
    if (first_embedding == nullptr || first_usage == nullptr ||
        first_projection == nullptr || rest_projection == nullptr) {
        return kStatusModel;
    }
    gather_codebook_kernel<<<
        blocks_for(static_cast<size_t>(frame_count) * 256),
        256,
        0,
        context->stream>>>(
        device_codes,
        frame_count,
        0,
        first_embedding,
        first_usage,
        semantic,
        0
    );
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch semantic codebook", status);
    }
    for (uint32_t codebook = 1; codebook < QWEN3_TTS_CODEC_CODEBOOKS; ++codebook) {
        char embedding_name[128];
        char usage_name[128];
        std::snprintf(
            embedding_name,
            sizeof(embedding_name),
            "decoder.quantizer.rvq_rest.vq.layers.%u._codebook.embedding_sum",
            codebook - 1
        );
        std::snprintf(
            usage_name,
            sizeof(usage_name),
            "decoder.quantizer.rvq_rest.vq.layers.%u._codebook.cluster_usage",
            codebook - 1
        );
        const float* embedding = require_f32_weight(
            context, embedding_name, error, error_capacity
        );
        const float* usage = require_f32_weight(
            context, usage_name, error, error_capacity
        );
        if (embedding == nullptr || usage == nullptr) {
            return kStatusModel;
        }
        gather_codebook_kernel<<<
            blocks_for(static_cast<size_t>(frame_count) * 256),
            256,
            0,
            context->stream>>>(
            device_codes,
            frame_count,
            codebook,
            embedding,
            usage,
            acoustic,
            codebook == 1 ? 0 : 1
        );
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return cuda_error(error, error_capacity, "launch acoustic codebook", status);
        }
    }
    constexpr float kOne = 1.0F;
    constexpr float kZero = 0.0F;
    cublasStatus_t cublas_status = cublasSgemm(
        context->cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        512,
        static_cast<int>(frame_count),
        256,
        &kOne,
        first_projection,
        256,
        semantic,
        256,
        &kZero,
        context->frontend_rvq,
        512
    );
    float* rest_projected = context->frontend_preconv;
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
        cublas_status = cublasSgemm(
            context->cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            512,
            static_cast<int>(frame_count),
            256,
            &kOne,
            rest_projection,
            256,
            acoustic,
            256,
            &kZero,
            rest_projected,
            512
        );
    }
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_error(
            error, error_capacity, "project split RVQ", cublas_status
        );
    }
    add_rvq_projection_kernel<<<
        blocks_for(static_cast<size_t>(frame_count) * 512),
        256,
        0,
        context->stream>>>(rest_projected, frame_count, context->frontend_rvq);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "combine split RVQ", status);
    }
    const float* preconv_weight = require_f32_weight(
        context, "decoder.pre_conv.conv.weight", error, error_capacity
    );
    const float* preconv_bias = require_f32_weight(
        context, "decoder.pre_conv.conv.bias", error, error_capacity
    );
    if (preconv_weight == nullptr || preconv_bias == nullptr) {
        return kStatusModel;
    }
    causal_preconv_kernel<<<
        blocks_for(static_cast<size_t>(frame_count) * 1024),
        256,
        0,
        context->stream>>>(
        context->frontend_rvq,
        context->frontend_history,
        frame_count,
        preconv_weight,
        preconv_bias,
        context->frontend_preconv
    );
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        update_frontend_history_kernel<<<2, 256, 0, context->stream>>>(
            context->frontend_rvq, frame_count, context->frontend_history
        );
        status = cudaGetLastError();
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch causal pre-convolution", status);
    }

    std::vector<float> rvq_row_major(static_cast<size_t>(frame_count) * 512);
    status = cudaMemcpyAsync(
        rvq_row_major.data(),
        context->frontend_rvq,
        rvq_row_major.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        context->stream
    );
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            preconv_output,
            context->frontend_preconv,
            static_cast<size_t>(frame_count) * 1024 * sizeof(float),
            cudaMemcpyDeviceToHost,
            context->stream
        );
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context->stream);
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "copy neural frontend checkpoints", status);
    }
    for (size_t channel = 0; channel < 512; ++channel) {
        for (size_t frame = 0; frame < frame_count; ++frame) {
            rvq_output[channel * frame_count + frame] =
                rvq_row_major[frame * 512 + channel];
        }
    }
    return kStatusOk;
}

extern "C" QWEN3_TTS_CODEC_API int32_t
qwen3_tts_codec_debug_transformer_packet_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    float* transformer_output,
    size_t transformer_capacity,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr || codec_frames == nullptr ||
        transformer_output == nullptr || frame_count == 0 ||
        frame_count > QWEN3_TTS_CODEC_MAX_PACKET_FRAMES ||
        transformer_capacity < static_cast<size_t>(frame_count) * 1024) {
        write_error(error, error_capacity, "invalid transformer checkpoint packet");
        return kStatusInvalidArgument;
    }
    std::vector<float> rvq(static_cast<size_t>(frame_count) * 512);
    std::vector<float> preconv(static_cast<size_t>(frame_count) * 1024);
    int32_t result = qwen3_tts_codec_debug_frontend_packet_v1(
        context,
        codec_frames,
        frame_count,
        rvq.data(),
        rvq.size(),
        preconv.data(),
        preconv.size(),
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        result = run_transformer_frame(
            context,
            context->frontend_preconv + static_cast<size_t>(frame) * 1024,
            context->neural_frame_position,
            context->transformer_packet + static_cast<size_t>(frame) * 1024,
            error,
            error_capacity
        );
        if (result != kStatusOk) {
            return result;
        }
        context->neural_frame_position += 1;
    }
    cudaError_t status = cudaMemcpyAsync(
        transformer_output,
        context->transformer_packet,
        static_cast<size_t>(frame_count) * 1024 * sizeof(float),
        cudaMemcpyDeviceToHost,
        context->stream
    );
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context->stream);
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "copy transformer checkpoint", status);
    }
    return kStatusOk;
}

extern "C" QWEN3_TTS_CODEC_API int32_t
qwen3_tts_codec_debug_latent_packet_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    float* stage_one_output,
    size_t stage_one_capacity,
    float* stage_two_output,
    size_t stage_two_capacity,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    const size_t stage_one_positions = static_cast<size_t>(frame_count) * 2;
    const size_t stage_two_positions = static_cast<size_t>(frame_count) * 4;
    if (context == nullptr || codec_frames == nullptr ||
        stage_one_output == nullptr || stage_two_output == nullptr ||
        frame_count == 0 || frame_count > QWEN3_TTS_CODEC_MAX_PACKET_FRAMES ||
        stage_one_capacity < stage_one_positions * 1024 ||
        stage_two_capacity < stage_two_positions * 1024) {
        write_error(error, error_capacity, "invalid latent checkpoint packet");
        return kStatusInvalidArgument;
    }
    std::vector<float> transformer(static_cast<size_t>(frame_count) * 1024);
    int32_t result = qwen3_tts_codec_debug_transformer_packet_v1(
        context,
        codec_frames,
        frame_count,
        transformer.data(),
        transformer.size(),
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    std::vector<float> stage_one_row(stage_one_positions * 1024);
    float* stage_two_device = nullptr;
    size_t actual_stage_two_positions = 0;
    result = run_latent_upsampling(
        context,
        context->transformer_packet,
        frame_count,
        stage_one_row.data(),
        &stage_two_device,
        &actual_stage_two_positions,
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    if (actual_stage_two_positions != stage_two_positions) {
        write_error(error, error_capacity, "latent upsampler returned an invalid length");
        return kStatusState;
    }
    std::vector<float> stage_two_row(stage_two_positions * 1024);
    cudaError_t status = cudaMemcpyAsync(
        stage_two_row.data(),
        stage_two_device,
        stage_two_row.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        context->stream
    );
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(context->stream);
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "copy latent stage two", status);
    }
    for (size_t channel = 0; channel < 1024; ++channel) {
        for (size_t position = 0; position < stage_one_positions; ++position) {
            stage_one_output[channel * stage_one_positions + position] =
                stage_one_row[position * 1024 + channel];
        }
        for (size_t position = 0; position < stage_two_positions; ++position) {
            stage_two_output[channel * stage_two_positions + position] =
                stage_two_row[position * 1024 + channel];
        }
    }
    return kStatusOk;
}

extern "C" QWEN3_TTS_CODEC_API int32_t
qwen3_tts_codec_debug_decoder_checkpoint_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    uint32_t checkpoint,
    float* checkpoint_output,
    size_t checkpoint_capacity,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr || codec_frames == nullptr ||
        checkpoint_output == nullptr || frame_count == 0 ||
        frame_count > QWEN3_TTS_CODEC_MAX_PACKET_FRAMES || checkpoint < 6 ||
        checkpoint > 13) {
        write_error(error, error_capacity, "invalid waveform decoder checkpoint request");
        return kStatusInvalidArgument;
    }
    std::vector<float> stage_one(static_cast<size_t>(frame_count) * 2 * 1024);
    std::vector<float> stage_two(static_cast<size_t>(frame_count) * 4 * 1024);
    int32_t result = qwen3_tts_codec_debug_latent_packet_v1(
        context,
        codec_frames,
        frame_count,
        stage_one.data(),
        stage_one.size(),
        stage_two.data(),
        stage_two.size(),
        error,
        error_capacity
    );
    if (result != kStatusOk) {
        return result;
    }
    float* waveform = nullptr;
    size_t waveform_positions = 0;
    return run_waveform_decoder(
        context,
        context->latent_b,
        static_cast<size_t>(frame_count) * 4,
        checkpoint,
        checkpoint_output,
        checkpoint_capacity,
        &waveform,
        &waveform_positions,
        error,
        error_capacity
    );
}

extern "C" QWEN3_TTS_CODEC_API int32_t
qwen3_tts_codec_process_packet_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    int32_t is_final,
    int16_t* pcm_output,
    size_t pcm_capacity_samples,
    Qwen3TtsCodecPacketResultV1* result,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr || codec_frames == nullptr || pcm_output == nullptr ||
        result == nullptr) {
        write_error(
            error,
            error_capacity,
            "context, codec frames, PCM output, and result are required"
        );
        return kStatusInvalidArgument;
    }
    if (frame_count == 0 ||
        frame_count > QWEN3_TTS_CODEC_MAX_PACKET_FRAMES ||
        (is_final != 0 && is_final != 1)) {
        write_error(error, error_capacity, "packet must contain 1-4 frames");
        return kStatusInvalidArgument;
    }
    const size_t sample_count =
        static_cast<size_t>(frame_count) * QWEN3_TTS_CODEC_SAMPLES_PER_FRAME;
    if (pcm_capacity_samples < sample_count) {
        write_error(error, error_capacity, "PCM output capacity is too small");
        return kStatusInvalidArgument;
    }
    if (context->model_info.loaded == 0) {
        write_error(error, error_capacity, "decoder model is not loaded");
        return kStatusState;
    }
    if (context->finalized) {
        write_error(error, error_capacity, "stream is finalized; reset is required");
        return kStatusState;
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "select CUDA device", status);
    }

    const auto end_to_end_start = std::chrono::steady_clock::now();
    const uint32_t slot = context->next_ring_slot;
    uint16_t* codec_slot =
        context->codec_ring +
        static_cast<size_t>(slot) * QWEN3_TTS_CODEC_MAX_PACKET_FRAMES *
            QWEN3_TTS_CODEC_CODEBOOKS;
    int16_t* pcm_slot =
        context->pcm_ring +
        static_cast<size_t>(slot) * QWEN3_TTS_CODEC_MAX_PACKET_SAMPLES;
    int16_t* host_pcm_slot =
        context->host_pcm_ring +
        static_cast<size_t>(slot) * QWEN3_TTS_CODEC_MAX_PACKET_SAMPLES;

    status = cudaEventRecord(context->start_event, context->stream);
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "start packet timing", status);
    }
    int32_t pipeline_status = run_frontend_device(
        context,
        codec_frames,
        codec_slot,
        frame_count,
        error,
        error_capacity
    );
    if (pipeline_status != kStatusOk) {
        return pipeline_status;
    }
    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        pipeline_status = run_transformer_frame(
            context,
            context->frontend_preconv + static_cast<size_t>(frame) * 1024,
            context->neural_frame_position,
            context->transformer_packet + static_cast<size_t>(frame) * 1024,
            error,
            error_capacity
        );
        if (pipeline_status != kStatusOk) {
            return pipeline_status;
        }
        context->neural_frame_position += 1;
    }
    float* latent = nullptr;
    size_t latent_positions = 0;
    pipeline_status = run_latent_upsampling(
        context,
        context->transformer_packet,
        frame_count,
        nullptr,
        &latent,
        &latent_positions,
        error,
        error_capacity
    );
    if (pipeline_status != kStatusOk) {
        return pipeline_status;
    }
    float* waveform = nullptr;
    size_t waveform_positions = 0;
    pipeline_status = run_waveform_decoder(
        context,
        latent,
        latent_positions,
        0,
        nullptr,
        0,
        &waveform,
        &waveform_positions,
        error,
        error_capacity
    );
    if (pipeline_status != kStatusOk) {
        return pipeline_status;
    }
    if (waveform_positions != sample_count) {
        write_error(error, error_capacity, "waveform decoder returned an invalid length");
        return kStatusState;
    }
    waveform_to_pcm_kernel<<<
        blocks_for(sample_count), 256, 0, context->stream>>>(
        waveform, sample_count, pcm_slot
    );
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            host_pcm_slot,
            pcm_slot,
            sample_count * sizeof(int16_t),
            cudaMemcpyDeviceToHost,
            context->stream
        );
    }
    if (status == cudaSuccess) {
        status = cudaEventRecord(context->stop_event, context->stream);
    }
    if (status == cudaSuccess) {
        status = cudaEventSynchronize(context->stop_event);
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "complete neural decoder packet", status);
    }

    float gpu_milliseconds = 0.0F;
    status = cudaEventElapsedTime(
        &gpu_milliseconds, context->start_event, context->stop_event
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "measure packet GPU time", status);
    }
    std::memcpy(pcm_output, host_pcm_slot, sample_count * sizeof(int16_t));

    const uint64_t first_frame = context->frame_position;
    const uint64_t first_sample = context->emitted_samples;
    context->frame_position += frame_count;
    context->emitted_samples += sample_count;
    context->kv_ring_head =
        static_cast<uint32_t>(context->neural_frame_position % QWEN3_TTS_CODEC_KV_WINDOW);
    context->next_ring_slot =
        (context->next_ring_slot + 1) % context->ring_slots;
    context->finalized = is_final != 0;

    const auto end_to_end_stop = std::chrono::steady_clock::now();
    const float end_to_end_microseconds =
        std::chrono::duration<float, std::micro>(
            end_to_end_stop - end_to_end_start
        )
            .count();
    *result = Qwen3TtsCodecPacketResultV1{
        first_frame,
        first_sample,
        frame_count,
        static_cast<uint32_t>(sample_count),
        slot,
        static_cast<uint32_t>(is_final),
        gpu_milliseconds * 1000.0F,
        end_to_end_microseconds,
    };
    return kStatusOk;
}

extern "C" QWEN3_TTS_CODEC_API int32_t qwen3_tts_codec_warmup_v1(
    Qwen3TtsCodecContextV1* context,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr) {
        write_error(error, error_capacity, "context is required");
        return kStatusInvalidArgument;
    }
    if (context->model_info.loaded == 0) {
        write_error(error, error_capacity, "decoder model is not loaded");
        return kStatusState;
    }
    if (context->frame_position != 0 || context->neural_frame_position != 0 ||
        context->emitted_samples != 0 || context->next_ring_slot != 0 ||
        context->finalized) {
        write_error(error, error_capacity, "warmup requires a fresh decoder state");
        return kStatusState;
    }
    uint16_t codes[
        QWEN3_TTS_CODEC_MAX_PACKET_FRAMES * QWEN3_TTS_CODEC_CODEBOOKS
    ]{};
    int16_t pcm[QWEN3_TTS_CODEC_MAX_PACKET_SAMPLES]{};
    Qwen3TtsCodecPacketResultV1 result{};
    const int32_t process_status = qwen3_tts_codec_process_packet_v1(
        context,
        codes,
        QWEN3_TTS_CODEC_MAX_PACKET_FRAMES,
        0,
        pcm,
        QWEN3_TTS_CODEC_MAX_PACKET_SAMPLES,
        &result,
        error,
        error_capacity
    );
    const int32_t reset_status = reset_device_state(
        context, error, error_capacity
    );
    return process_status == kStatusOk ? reset_status : process_status;
}

extern "C" QWEN3_TTS_CODEC_API int32_t
qwen3_tts_codec_process_batch_v1(
    Qwen3TtsCodecBatchItemV1* items,
    uint32_t item_count,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (items == nullptr || item_count == 0 ||
        item_count > QWEN3_TTS_CODEC_MAX_BATCH_STREAMS) {
        write_error(error, error_capacity, "batch must contain 1-6 stream items");
        return kStatusInvalidArgument;
    }
    for (uint32_t item = 0; item < item_count; ++item) {
        if (items[item].context == nullptr) {
            write_error(error, error_capacity, "batch item context is required");
            return kStatusInvalidArgument;
        }
        for (uint32_t prior = 0; prior < item; ++prior) {
            if (items[item].context == items[prior].context) {
                write_error(error, error_capacity, "batch state handles must be independent");
                return kStatusInvalidArgument;
            }
        }
    }
    for (uint32_t item = 0; item < item_count; ++item) {
        const int32_t status = qwen3_tts_codec_process_packet_v1(
            items[item].context,
            items[item].codec_frames,
            items[item].frame_count,
            items[item].is_final,
            items[item].pcm_output,
            items[item].pcm_capacity_samples,
            items[item].result,
            error,
            error_capacity
        );
        if (status != kStatusOk) {
            return status;
        }
    }
    return kStatusOk;
}

extern "C" QWEN3_TTS_CODEC_API int32_t
qwen3_tts_codec_process_fixture_packet_v1(
    Qwen3TtsCodecContextV1* context,
    const uint16_t* codec_frames,
    uint32_t frame_count,
    int32_t is_final,
    int16_t* pcm_output,
    size_t pcm_capacity_samples,
    Qwen3TtsCodecPacketResultV1* result,
    char* error,
    size_t error_capacity
) {
    clear_error(error, error_capacity);
    if (context == nullptr || codec_frames == nullptr || pcm_output == nullptr ||
        result == nullptr) {
        write_error(
            error,
            error_capacity,
            "context, codec frames, PCM output, and result are required"
        );
        return kStatusInvalidArgument;
    }
    if (frame_count == 0 ||
        frame_count > QWEN3_TTS_CODEC_MAX_PACKET_FRAMES ||
        (is_final != 0 && is_final != 1)) {
        write_error(error, error_capacity, "packet must contain 1-4 frames");
        return kStatusInvalidArgument;
    }
    const size_t sample_count =
        static_cast<size_t>(frame_count) * QWEN3_TTS_CODEC_SAMPLES_PER_FRAME;
    if (pcm_capacity_samples < sample_count) {
        write_error(error, error_capacity, "PCM output capacity is too small");
        return kStatusInvalidArgument;
    }
    if (context->finalized) {
        write_error(error, error_capacity, "stream is finalized; reset is required");
        return kStatusState;
    }
    cudaError_t status = cudaSetDevice(context->device_index);
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "select CUDA device", status);
    }

    const auto end_to_end_start = std::chrono::steady_clock::now();
    const uint32_t slot = context->next_ring_slot;
    uint16_t* codec_slot =
        context->codec_ring +
        static_cast<size_t>(slot) * QWEN3_TTS_CODEC_MAX_PACKET_FRAMES *
            QWEN3_TTS_CODEC_CODEBOOKS;
    int16_t* pcm_slot =
        context->pcm_ring +
        static_cast<size_t>(slot) * QWEN3_TTS_CODEC_MAX_PACKET_SAMPLES;
    int16_t* host_pcm_slot =
        context->host_pcm_ring +
        static_cast<size_t>(slot) * QWEN3_TTS_CODEC_MAX_PACKET_SAMPLES;

    status = cudaEventRecord(context->start_event, context->stream);
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            codec_slot,
            codec_frames,
            static_cast<size_t>(frame_count) * QWEN3_TTS_CODEC_CODEBOOKS *
                sizeof(uint16_t),
            cudaMemcpyHostToDevice,
            context->stream
        );
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "queue codec copy", status);
    }

    int32_t* preconv_history = context->fixture_history;
    int32_t* representative_kv = context->fixture_history + 2;
    int32_t* transpose_tails =
        context->fixture_history + 2 + QWEN3_TTS_CODEC_KV_WINDOW;
    ingest_fixture_kernel<<<1, 1, 0, context->stream>>>(
        codec_slot,
        frame_count,
        context->frame_position,
        preconv_history,
        representative_kv,
        context->scratch0
    );
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch fixture ingest", status);
    }

    constexpr size_t kKvItemsPerFrame =
        2ULL * QWEN3_TTS_CODEC_TRANSFORMER_LAYERS *
        QWEN3_TTS_CODEC_KV_HEADS * QWEN3_TTS_CODEC_HEAD_DIM;
    update_exact_kv_fixture_kernel<<<
        blocks_for(static_cast<size_t>(frame_count) * kKvItemsPerFrame),
        256,
        0,
        context->stream>>>(
        context->scratch0,
        frame_count,
        context->frame_position,
        context->transformer_kv
    );
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "launch KV update", status);
    }

    int32_t launch_status = launch_repeat(
        context->scratch0,
        frame_count,
        2,
        context->scratch1,
        context->stream,
        error,
        error_capacity
    );
    if (launch_status != kStatusOk) {
        return launch_status;
    }
    launch_status = launch_repeat(
        context->scratch1,
        static_cast<size_t>(frame_count) * 2,
        2,
        context->scratch2,
        context->stream,
        error,
        error_capacity
    );
    if (launch_status != kStatusOk) {
        return launch_status;
    }
    launch_status = launch_transpose(
        context->scratch2,
        static_cast<size_t>(frame_count) * 4,
        8,
        transpose_tails,
        context->scratch3,
        context->stream,
        error,
        error_capacity
    );
    if (launch_status != kStatusOk) {
        return launch_status;
    }
    launch_status = launch_transpose(
        context->scratch3,
        static_cast<size_t>(frame_count) * 32,
        5,
        transpose_tails + 8,
        context->scratch4,
        context->stream,
        error,
        error_capacity
    );
    if (launch_status != kStatusOk) {
        return launch_status;
    }
    launch_status = launch_transpose(
        context->scratch4,
        static_cast<size_t>(frame_count) * 160,
        4,
        transpose_tails + 13,
        context->scratch5,
        context->stream,
        error,
        error_capacity
    );
    if (launch_status != kStatusOk) {
        return launch_status;
    }
    launch_status = launch_transpose(
        context->scratch5,
        static_cast<size_t>(frame_count) * 640,
        3,
        transpose_tails + 17,
        context->scratch6,
        context->stream,
        error,
        error_capacity
    );
    if (launch_status != kStatusOk) {
        return launch_status;
    }

    convert_pcm_kernel<<<blocks_for(sample_count), 256, 0, context->stream>>>(
        context->scratch6, sample_count, pcm_slot
    );
    status = cudaGetLastError();
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
            host_pcm_slot,
            pcm_slot,
            sample_count * sizeof(int16_t),
            cudaMemcpyDeviceToHost,
            context->stream
        );
    }
    if (status == cudaSuccess) {
        status = cudaEventRecord(context->stop_event, context->stream);
    }
    if (status == cudaSuccess) {
        status = cudaEventSynchronize(context->stop_event);
    }
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "complete fixture packet", status);
    }

    float gpu_milliseconds = 0.0F;
    status = cudaEventElapsedTime(
        &gpu_milliseconds, context->start_event, context->stop_event
    );
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "measure packet GPU time", status);
    }
    std::memcpy(pcm_output, host_pcm_slot, sample_count * sizeof(int16_t));

    const uint64_t first_frame = context->frame_position;
    const uint64_t first_sample = context->emitted_samples;
    context->frame_position += frame_count;
    context->emitted_samples += sample_count;
    context->kv_ring_head =
        static_cast<uint32_t>(context->frame_position % QWEN3_TTS_CODEC_KV_WINDOW);
    context->next_ring_slot =
        (context->next_ring_slot + 1) % context->ring_slots;
    context->finalized = is_final != 0;

    const auto end_to_end_stop = std::chrono::steady_clock::now();
    const float end_to_end_microseconds =
        std::chrono::duration<float, std::micro>(
            end_to_end_stop - end_to_end_start
        )
            .count();
    *result = Qwen3TtsCodecPacketResultV1{
        first_frame,
        first_sample,
        frame_count,
        static_cast<uint32_t>(sample_count),
        slot,
        static_cast<uint32_t>(is_final),
        gpu_milliseconds * 1000.0F,
        end_to_end_microseconds,
    };
    return kStatusOk;
}
