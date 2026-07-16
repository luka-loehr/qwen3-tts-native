#include "qwen3_tts_codec.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

constexpr int32_t kStatusOk = QWEN3_TTS_CODEC_STATUS_OK;
constexpr int32_t kStatusInvalidArgument =
    QWEN3_TTS_CODEC_STATUS_INVALID_ARGUMENT;
constexpr int32_t kStatusCuda = QWEN3_TTS_CODEC_STATUS_CUDA;
constexpr int32_t kStatusState = QWEN3_TTS_CODEC_STATUS_STATE;
constexpr int32_t kStatusAllocation = QWEN3_TTS_CODEC_STATUS_ALLOCATION;
constexpr int32_t kStatusModel = QWEN3_TTS_CODEC_STATUS_MODEL;
constexpr uint32_t kExpectedDecoderTensors = 271;

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
    uint64_t frame_position = 0;
    uint64_t emitted_samples = 0;
    uint32_t next_ring_slot = 0;
    uint32_t kv_ring_head = 0;
    bool finalized = false;
    std::unordered_map<std::string, DeviceTensor> weights;
    Qwen3TtsCodecModelInfoV1 model_info{};
};

namespace {

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
    status = cudaStreamSynchronize(context->stream);
    if (status != cudaSuccess) {
        return cuda_error(error, error_capacity, "synchronize reset", status);
    }
    std::memset(context->host_pcm_ring, 0, kPcmRingBytes);
    context->frame_position = 0;
    context->emitted_samples = 0;
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
        context->model_info.device_bytes;
    *output = Qwen3TtsCodecStateInfoV1{
        context->frame_position,
        context->emitted_samples,
        device_bytes,
        kPcmRingBytes,
        kTransformerKvBytes,
        kConvolutionHistoryBytes,
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
        DeviceTensor tensor{};
        tensor.byte_length = source.byte_length;
        tensor.rank = source.rank;
        tensor.dtype = source.dtype;
        std::memcpy(tensor.shape, source.shape, sizeof(tensor.shape));
        cuda_status = cudaMalloc(&tensor.data, static_cast<size_t>(source.byte_length));
        if (cuda_status == cudaSuccess) {
            cuda_status = cudaMemcpyAsync(
                tensor.data,
                source.data,
                static_cast<size_t>(source.byte_length),
                cudaMemcpyHostToDevice,
                context->stream
            );
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
        info.device_bytes += source.byte_length;
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
