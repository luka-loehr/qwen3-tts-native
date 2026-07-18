#include "qwen3_tts_native.h"
#include "talker_internal.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kTalkerLayers = 28;
constexpr int kPredictorLayers = 5;
constexpr int kResidualCodebooks = 15;
constexpr int kTalkerHidden = 2'048;
constexpr int kTalkerIntermediate = 6'144;
constexpr int kTalkerQueryHeads = 16;
constexpr int kTalkerKeyValueHeads = 8;
constexpr int kTalkerHeadDimension = 128;
constexpr int kTalkerKeyValueWidth = kTalkerKeyValueHeads * kTalkerHeadDimension;
constexpr int kTalkerVocabulary = 3'072;
constexpr int kTextVocabulary = 151'936;
constexpr int kPredictorHidden = 1'024;
constexpr int kPredictorIntermediate = 3'072;
constexpr int kPredictorQueryHeads = 16;
constexpr int kPredictorKeyValueHeads = 8;
constexpr int kPredictorHeadDimension = 128;
constexpr int kPredictorQueryWidth = kPredictorQueryHeads * kPredictorHeadDimension;
constexpr int kPredictorKeyValueWidth = kPredictorKeyValueHeads * kPredictorHeadDimension;
constexpr int kPredictorVocabulary = 2'048;
constexpr int kPredictorSequence = 16;
constexpr int kPrefillGemmCapacity = 512;
constexpr float kRmsEpsilon = 1.0e-6f;
constexpr float kRopeTheta = 1.0e6f;
constexpr uint16_t kCodecEos = 2'150;

void write_error(char* destination, size_t capacity, const char* message) {
    if (destination != nullptr && capacity != 0) {
        std::snprintf(destination, capacity, "%s", message);
    }
}

class CudaFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw CudaFailure(
            std::string(operation) + " failed: " + cudaGetErrorString(status)
        );
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CudaFailure(
            std::string(operation) + " failed with cuBLAS status "
                + std::to_string(static_cast<int>(status))
        );
    }
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
        if (bytes != 0) {
            check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)),
          bytes_(std::exchange(other.bytes_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (pointer_ != nullptr) {
                cudaFree(pointer_);
            }
            pointer_ = std::exchange(other.pointer_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    template <typename T>
    T* as() {
        return static_cast<T*>(pointer_);
    }

    template <typename T>
    const T* as() const {
        return static_cast<const T*>(pointer_);
    }

    size_t bytes() const {
        return bytes_;
    }

private:
    void* pointer_ = nullptr;
    size_t bytes_ = 0;
};

struct DeviceTensor {
    DeviceBuffer storage;
    std::vector<uint64_t> shape;
};

struct DecoderWeights {
    const __nv_bfloat16* input_norm = nullptr;
    const __nv_bfloat16* q_projection = nullptr;
    const __nv_bfloat16* k_projection = nullptr;
    const __nv_bfloat16* v_projection = nullptr;
    const __nv_bfloat16* output_projection = nullptr;
    const __nv_bfloat16* q_norm = nullptr;
    const __nv_bfloat16* k_norm = nullptr;
    const __nv_bfloat16* post_attention_norm = nullptr;
    const __nv_bfloat16* gate_projection = nullptr;
    const __nv_bfloat16* up_projection = nullptr;
    const __nv_bfloat16* down_projection = nullptr;
};

/* Weight-only INT8 image of one decode matrix: int8 data in the same
 * [out_features x in_features] layout as the BF16 tensor plus one FP32 scale
 * per output channel. Null data means the BF16 cuBLAS path is used. */
struct QuantizedTensor {
    const int8_t* data = nullptr;
    const float* scales = nullptr;
};

struct DecoderWeightsInt8 {
    QuantizedTensor q_projection;
    QuantizedTensor k_projection;
    QuantizedTensor v_projection;
    QuantizedTensor output_projection;
    QuantizedTensor gate_projection;
    QuantizedTensor up_projection;
    QuantizedTensor down_projection;
};

struct LayerCache {
    DeviceBuffer key;
    DeviceBuffer value;
};

struct ModelDimensions {
    int hidden;
    int intermediate;
    int query_heads;
    int key_value_heads;
    int head_dimension;
    int query_width;
    int key_value_width;
};

constexpr ModelDimensions kTalkerDimensions{
    kTalkerHidden,
    kTalkerIntermediate,
    kTalkerQueryHeads,
    kTalkerKeyValueHeads,
    kTalkerHeadDimension,
    kTalkerHidden,
    kTalkerKeyValueWidth,
};

constexpr ModelDimensions kPredictorDimensions{
    kPredictorHidden,
    kPredictorIntermediate,
    kPredictorQueryHeads,
    kPredictorKeyValueHeads,
    kPredictorHeadDimension,
    kPredictorQueryWidth,
    kPredictorKeyValueWidth,
};

/* Shared lockstep-decode workspace. One frame step for up to kCapacity
 * sessions shares every weight-read GEMM (M = batch rows) while KV caches,
 * sampling state, histories, and events stay session-local. */
struct BatchWorkspace {
    static constexpr int kCapacity = 8;
    static constexpr int kPositionSlots = kPredictorSequence + 1;
    static constexpr int kKvPointerCount =
        (kTalkerLayers + kPredictorLayers) * 2 * kCapacity;

    explicit BatchWorkspace(int device_index) : device_index(device_index) {
        check_cuda(cudaSetDevice(device_index), "cudaSetDevice(batch workspace)");
        try {
            check_cuda(
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags(batch)"
            );
            check_cublas(cublasCreate(&cublas), "cublasCreate(batch)");
            check_cublas(cublasSetStream(cublas, stream), "cublasSetStream(batch)");
            check_cublas(
                cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH),
                "cublasSetMathMode(batch)"
            );
            const size_t element = sizeof(__nv_bfloat16);
            hidden = DeviceBuffer(kCapacity * kTalkerHidden * element);
            normalized = DeviceBuffer(kCapacity * kTalkerHidden * element);
            query = DeviceBuffer(kCapacity * kPredictorQueryWidth * element);
            key = DeviceBuffer(kCapacity * kTalkerKeyValueWidth * element);
            value = DeviceBuffer(kCapacity * kTalkerKeyValueWidth * element);
            attention = DeviceBuffer(kCapacity * kPredictorQueryWidth * element);
            projection = DeviceBuffer(kCapacity * kTalkerHidden * element);
            gate = DeviceBuffer(kCapacity * kTalkerIntermediate * element);
            up = DeviceBuffer(kCapacity * kTalkerIntermediate * element);
            logits = DeviceBuffer(kCapacity * kTalkerVocabulary * element);
            text_input = DeviceBuffer(kCapacity * kTalkerHidden * element);
            device_positions = DeviceBuffer(
                static_cast<size_t>(kPositionSlots) * kCapacity * sizeof(int)
            );
            device_kv_bases = DeviceBuffer(
                static_cast<size_t>(kKvPointerCount) * sizeof(__nv_bfloat16*)
            );
            device_token_bases = DeviceBuffer(kCapacity * sizeof(const int*));
            device_trailing = DeviceBuffer(kCapacity * sizeof(int));
            device_trailing_ptrs = DeviceBuffer(kCapacity * sizeof(const int*));
            device_counts = DeviceBuffer(kCapacity * sizeof(int));
            check_cuda(
                cudaMallocHost(
                    reinterpret_cast<void**>(&pinned_positions),
                    static_cast<size_t>(kPositionSlots) * kCapacity * sizeof(int)
                ),
                "cudaMallocHost(batch positions)"
            );
            check_cuda(
                cudaMallocHost(
                    reinterpret_cast<void**>(&pinned_kv_bases),
                    static_cast<size_t>(kKvPointerCount) * sizeof(__nv_bfloat16*)
                ),
                "cudaMallocHost(batch kv bases)"
            );
            check_cuda(
                cudaMallocHost(
                    reinterpret_cast<void**>(&pinned_token_bases),
                    kCapacity * sizeof(const int*)
                ),
                "cudaMallocHost(batch token bases)"
            );
            check_cuda(
                cudaMallocHost(
                    reinterpret_cast<void**>(&pinned_trailing),
                    kCapacity * sizeof(int)
                ),
                "cudaMallocHost(batch trailing tokens)"
            );
            check_cuda(
                cudaMallocHost(
                    reinterpret_cast<void**>(&pinned_counts),
                    kCapacity * sizeof(int)
                ),
                "cudaMallocHost(batch history counts)"
            );
            {
                const int* trailing_ptrs[kCapacity];
                for (int slot = 0; slot < kCapacity; ++slot) {
                    trailing_ptrs[slot] = device_trailing.as<int>() + slot;
                }
                check_cuda(
                    cudaMemcpy(
                        device_trailing_ptrs.as<const int*>(),
                        trailing_ptrs,
                        kCapacity * sizeof(const int*),
                        cudaMemcpyHostToDevice
                    ),
                    "upload batch trailing pointers"
                );
            }
            for (int slot = 0; slot < kCapacity; ++slot) {
                check_cuda(
                    cudaEventCreateWithFlags(&join_events[slot], cudaEventDisableTiming),
                    "cudaEventCreate(batch join)"
                );
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~BatchWorkspace() {
        cudaSetDevice(device_index);
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        release();
    }

    BatchWorkspace(const BatchWorkspace&) = delete;
    BatchWorkspace& operator=(const BatchWorkspace&) = delete;

    void release() noexcept {
        destroy_graph();
        for (int slot = 0; slot < kCapacity; ++slot) {
            if (join_events[slot] != nullptr) {
                cudaEventDestroy(join_events[slot]);
                join_events[slot] = nullptr;
            }
        }
        if (pinned_counts != nullptr) {
            cudaFreeHost(pinned_counts);
            pinned_counts = nullptr;
        }
        if (pinned_trailing != nullptr) {
            cudaFreeHost(pinned_trailing);
            pinned_trailing = nullptr;
        }
        if (pinned_token_bases != nullptr) {
            cudaFreeHost(pinned_token_bases);
            pinned_token_bases = nullptr;
        }
        if (pinned_kv_bases != nullptr) {
            cudaFreeHost(pinned_kv_bases);
            pinned_kv_bases = nullptr;
        }
        if (pinned_positions != nullptr) {
            cudaFreeHost(pinned_positions);
            pinned_positions = nullptr;
        }
        if (cublas != nullptr) {
            cublasDestroy(cublas);
            cublas = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    int device_index;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    DeviceBuffer hidden;
    DeviceBuffer normalized;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer attention;
    DeviceBuffer projection;
    DeviceBuffer gate;
    DeviceBuffer up;
    DeviceBuffer logits;
    DeviceBuffer text_input;
    DeviceBuffer device_positions;
    DeviceBuffer device_kv_bases;
    DeviceBuffer device_token_bases;
    int* pinned_positions = nullptr;
    __nv_bfloat16** pinned_kv_bases = nullptr;
    const int** pinned_token_bases = nullptr;
    int* pinned_trailing = nullptr;
    int* pinned_counts = nullptr;
    std::array<cudaEvent_t, kCapacity> join_events{};
    DeviceBuffer device_trailing;
    DeviceBuffer device_trailing_ptrs;
    DeviceBuffer device_counts;

    /* Captured lockstep frame. Valid while the session tuple, sampling
     * configurations, and capacities in graph_key are unchanged. */
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    std::vector<uint64_t> graph_key;
    int uncaptured_runs = 0;

    void destroy_graph() noexcept {
        if (graph_exec != nullptr) {
            cudaGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        graph_key.clear();
        uncaptured_runs = 0;
    }
};

class TalkerModel {
public:
    explicit TalkerModel(int device_index) : device_index_(device_index) {
        check_cuda(cudaSetDevice(device_index_), "cudaSetDevice(model)");
        check_cuda(
            cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags(model upload)"
        );
    }

    ~TalkerModel() {
        cudaSetDevice(device_index_);
        if (upload_stream_ != nullptr) {
            cudaStreamSynchronize(upload_stream_);
            cudaStreamDestroy(upload_stream_);
        }
    }

    TalkerModel(const TalkerModel&) = delete;
    TalkerModel& operator=(const TalkerModel&) = delete;

    void upload_tensor(
        const char* name,
        const void* data,
        uint64_t byte_size,
        int rank,
        const uint64_t* shape
    ) {
        if (finalized_) {
            throw std::runtime_error("model weights are already finalized");
        }
        if (name == nullptr || *name == '\0' || data == nullptr || shape == nullptr) {
            throw std::runtime_error("tensor upload received a null argument");
        }
        if (rank <= 0 || rank > 4 || byte_size == 0) {
            throw std::runtime_error("tensor upload has an invalid rank or byte size");
        }
        uint64_t elements = 1;
        std::vector<uint64_t> dimensions;
        dimensions.reserve(rank);
        for (int index = 0; index < rank; ++index) {
            if (shape[index] == 0
                || elements > std::numeric_limits<uint64_t>::max() / shape[index]) {
                throw std::runtime_error("tensor shape is empty or overflows");
            }
            elements *= shape[index];
            dimensions.push_back(shape[index]);
        }
        if (elements > std::numeric_limits<uint64_t>::max() / sizeof(__nv_bfloat16)
            || elements * sizeof(__nv_bfloat16) != byte_size) {
            throw std::runtime_error("tensor byte size does not match BF16 shape");
        }
        const std::string tensor_name(name);
        if (tensors_.find(tensor_name) != tensors_.end()) {
            throw std::runtime_error("duplicate tensor upload: " + tensor_name);
        }
        DeviceTensor tensor{DeviceBuffer(static_cast<size_t>(byte_size)), std::move(dimensions)};
        check_cuda(
            cudaMemcpyAsync(
                tensor.storage.as<void>(),
                data,
                static_cast<size_t>(byte_size),
                cudaMemcpyHostToDevice,
                upload_stream_
            ),
            "cudaMemcpyAsync(model weight H2D)"
        );
        weight_bytes_ += byte_size;
        tensors_.emplace(tensor_name, std::move(tensor));
    }

    Qwen3TtsModelMemory finalize() {
        if (finalized_) {
            throw std::runtime_error("model weights are already finalized");
        }
        if (tensors_.size() != 404) {
            throw std::runtime_error(
                "expected exactly 404 VoiceDesign tensors, found "
                    + std::to_string(tensors_.size())
            );
        }

        codec_embedding_ = require(
            "talker.model.codec_embedding.weight",
            {kTalkerVocabulary, kTalkerHidden}
        );
        text_embedding_ = require(
            "talker.model.text_embedding.weight",
            {kTextVocabulary, kTalkerHidden}
        );
        text_fc1_ = require(
            "talker.text_projection.linear_fc1.weight",
            {kTalkerHidden, kTalkerHidden}
        );
        text_fc1_bias_ = require(
            "talker.text_projection.linear_fc1.bias",
            {kTalkerHidden}
        );
        text_fc2_ = require(
            "talker.text_projection.linear_fc2.weight",
            {kTalkerHidden, kTalkerHidden}
        );
        text_fc2_bias_ = require(
            "talker.text_projection.linear_fc2.bias",
            {kTalkerHidden}
        );
        talker_norm_ = require("talker.model.norm.weight", {kTalkerHidden});
        codec_head_ = require(
            "talker.codec_head.weight",
            {kTalkerVocabulary, kTalkerHidden}
        );

        talker_layers_.reserve(kTalkerLayers);
        for (int layer = 0; layer < kTalkerLayers; ++layer) {
            talker_layers_.push_back(load_layer(
                "talker.model.layers." + std::to_string(layer),
                kTalkerDimensions
            ));
        }

        small_to_predictor_ = require(
            "talker.code_predictor.small_to_mtp_projection.weight",
            {kPredictorHidden, kTalkerHidden}
        );
        small_to_predictor_bias_ = require(
            "talker.code_predictor.small_to_mtp_projection.bias",
            {kPredictorHidden}
        );
        predictor_norm_ = require(
            "talker.code_predictor.model.norm.weight",
            {kPredictorHidden}
        );
        predictor_layers_.reserve(kPredictorLayers);
        for (int layer = 0; layer < kPredictorLayers; ++layer) {
            predictor_layers_.push_back(load_layer(
                "talker.code_predictor.model.layers." + std::to_string(layer),
                kPredictorDimensions
            ));
        }
        for (int group = 0; group < kResidualCodebooks; ++group) {
            predictor_embeddings_[group] = require(
                "talker.code_predictor.model.codec_embedding."
                    + std::to_string(group) + ".weight",
                {kPredictorVocabulary, kTalkerHidden}
            );
            predictor_heads_[group] = require(
                "talker.code_predictor.lm_head." + std::to_string(group) + ".weight",
                {kPredictorVocabulary, kPredictorHidden}
            );
        }

        const char* int8_environment = std::getenv("QWEN3_TTS_INT8_DECODE");
        int8_decode_ = int8_environment != nullptr
            && (std::strcmp(int8_environment, "1") == 0
                || std::strcmp(int8_environment, "true") == 0);
        if (int8_decode_) {
            const auto quantize = [this](
                const __nv_bfloat16* weight,
                int in_features,
                int out_features
            ) {
                DeviceBuffer data(
                    static_cast<size_t>(in_features) * out_features
                );
                DeviceBuffer scales(
                    static_cast<size_t>(out_features) * sizeof(float)
                );
                check_cuda(
                    qwen3_tts::launch_quantize_weight_rows(
                        weight,
                        data.as<int8_t>(),
                        scales.as<float>(),
                        in_features,
                        out_features,
                        upload_stream_
                    ),
                    "quantize decode weight"
                );
                QuantizedTensor tensor{data.as<int8_t>(), scales.as<float>()};
                int8_storage_.push_back(std::move(data));
                int8_storage_.push_back(std::move(scales));
                return tensor;
            };
            const auto quantize_layers = [this, &quantize](
                const std::vector<DecoderWeights>& layers,
                const ModelDimensions& dimensions
            ) {
                std::vector<DecoderWeightsInt8> result;
                result.reserve(layers.size());
                for (const DecoderWeights& layer : layers) {
                    result.push_back({
                        quantize(layer.q_projection, dimensions.hidden, dimensions.query_width),
                        quantize(layer.k_projection, dimensions.hidden, dimensions.key_value_width),
                        quantize(layer.v_projection, dimensions.hidden, dimensions.key_value_width),
                        quantize(layer.output_projection, dimensions.query_width, dimensions.hidden),
                        quantize(layer.gate_projection, dimensions.hidden, dimensions.intermediate),
                        quantize(layer.up_projection, dimensions.hidden, dimensions.intermediate),
                        quantize(layer.down_projection, dimensions.intermediate, dimensions.hidden),
                    });
                }
                return result;
            };
            codec_head_int8_ = quantize(codec_head_, kTalkerHidden, kTalkerVocabulary);
            small_to_predictor_int8_ =
                quantize(small_to_predictor_, kTalkerHidden, kPredictorHidden);
            text_fc1_int8_ = quantize(text_fc1_, kTalkerHidden, kTalkerHidden);
            text_fc2_int8_ = quantize(text_fc2_, kTalkerHidden, kTalkerHidden);
            for (int group = 0; group < kResidualCodebooks; ++group) {
                predictor_heads_int8_[group] = quantize(
                    predictor_heads_[group], kPredictorHidden, kPredictorVocabulary
                );
            }
            talker_layers_int8_ = quantize_layers(talker_layers_, kTalkerDimensions);
            predictor_layers_int8_ = quantize_layers(predictor_layers_, kPredictorDimensions);
        }

        check_cuda(
            cudaStreamSynchronize(upload_stream_),
            "cudaStreamSynchronize(model weight upload)"
        );
        finalized_ = true;
        Qwen3TtsModelMemory result{};
        result.shared_weight_bytes = weight_bytes_;
        result.tensor_count = static_cast<uint32_t>(tensors_.size());
        result.device_index = device_index_;
        return result;
    }

    bool finalized() const {
        return finalized_;
    }

    int device_index() const {
        return device_index_;
    }

private:
    friend class TalkerContext;

    const __nv_bfloat16* require(const std::string& name, std::vector<uint64_t> shape) {
        const auto found = tensors_.find(name);
        if (found == tensors_.end()) {
            throw std::runtime_error("checkpoint is missing required tensor " + name);
        }
        if (found->second.shape != shape) {
            throw std::runtime_error("unexpected shape for tensor " + name);
        }
        return found->second.storage.as<__nv_bfloat16>();
    }

    DecoderWeights load_layer(const std::string& prefix, const ModelDimensions& dimensions) {
        return {
            require(prefix + ".input_layernorm.weight", {
                static_cast<uint64_t>(dimensions.hidden),
            }),
            require(prefix + ".self_attn.q_proj.weight", {
                static_cast<uint64_t>(dimensions.query_width),
                static_cast<uint64_t>(dimensions.hidden),
            }),
            require(prefix + ".self_attn.k_proj.weight", {
                static_cast<uint64_t>(dimensions.key_value_width),
                static_cast<uint64_t>(dimensions.hidden),
            }),
            require(prefix + ".self_attn.v_proj.weight", {
                static_cast<uint64_t>(dimensions.key_value_width),
                static_cast<uint64_t>(dimensions.hidden),
            }),
            require(prefix + ".self_attn.o_proj.weight", {
                static_cast<uint64_t>(dimensions.hidden),
                static_cast<uint64_t>(dimensions.query_width),
            }),
            require(prefix + ".self_attn.q_norm.weight", {
                static_cast<uint64_t>(dimensions.head_dimension),
            }),
            require(prefix + ".self_attn.k_norm.weight", {
                static_cast<uint64_t>(dimensions.head_dimension),
            }),
            require(prefix + ".post_attention_layernorm.weight", {
                static_cast<uint64_t>(dimensions.hidden),
            }),
            require(prefix + ".mlp.gate_proj.weight", {
                static_cast<uint64_t>(dimensions.intermediate),
                static_cast<uint64_t>(dimensions.hidden),
            }),
            require(prefix + ".mlp.up_proj.weight", {
                static_cast<uint64_t>(dimensions.intermediate),
                static_cast<uint64_t>(dimensions.hidden),
            }),
            require(prefix + ".mlp.down_proj.weight", {
                static_cast<uint64_t>(dimensions.hidden),
                static_cast<uint64_t>(dimensions.intermediate),
            }),
        };
    }

    BatchWorkspace& batch_workspace() {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        if (!batch_) {
            batch_ = std::make_unique<BatchWorkspace>(device_index_);
        }
        return *batch_;
    }

    int device_index_;
    bool finalized_ = false;
    bool int8_decode_ = false;
    uint64_t weight_bytes_ = 0;
    std::vector<DeviceBuffer> int8_storage_;
    std::vector<DecoderWeightsInt8> talker_layers_int8_;
    std::vector<DecoderWeightsInt8> predictor_layers_int8_;
    QuantizedTensor codec_head_int8_;
    QuantizedTensor small_to_predictor_int8_;
    QuantizedTensor text_fc1_int8_;
    QuantizedTensor text_fc2_int8_;
    std::array<QuantizedTensor, kResidualCodebooks> predictor_heads_int8_{};
    std::mutex batch_mutex_;
    std::unique_ptr<BatchWorkspace> batch_;
    cudaStream_t upload_stream_ = nullptr;
    std::unordered_map<std::string, DeviceTensor> tensors_;
    std::vector<DecoderWeights> talker_layers_;
    std::vector<DecoderWeights> predictor_layers_;
    const __nv_bfloat16* codec_embedding_ = nullptr;
    const __nv_bfloat16* text_embedding_ = nullptr;
    const __nv_bfloat16* text_fc1_ = nullptr;
    const __nv_bfloat16* text_fc1_bias_ = nullptr;
    const __nv_bfloat16* text_fc2_ = nullptr;
    const __nv_bfloat16* text_fc2_bias_ = nullptr;
    const __nv_bfloat16* talker_norm_ = nullptr;
    const __nv_bfloat16* codec_head_ = nullptr;
    const __nv_bfloat16* small_to_predictor_ = nullptr;
    const __nv_bfloat16* small_to_predictor_bias_ = nullptr;
    const __nv_bfloat16* predictor_norm_ = nullptr;
    std::array<const __nv_bfloat16*, kResidualCodebooks> predictor_embeddings_{};
    std::array<const __nv_bfloat16*, kResidualCodebooks> predictor_heads_{};
};

class TalkerContext {
public:
    TalkerContext(
        std::shared_ptr<TalkerModel> model,
        int max_sequence_length,
        uint64_t seed
    )
        : model_(std::move(model)),
          device_index_(model_->device_index()),
          max_sequence_length_(max_sequence_length),
          prefill_gemm_capacity_(std::min(max_sequence_length, kPrefillGemmCapacity)),
          trace_enabled_(std::getenv("QWEN3_TTS_PARITY_TRACE") != nullptr),
          stage_dump_enabled_(std::getenv("QWEN3_TTS_STAGE_DUMP") != nullptr) {
        if (max_sequence_length < 16 || max_sequence_length > 8'192) {
            throw std::runtime_error("max sequence length must be in [16, 8192]");
        }
        if (!model_->finalized()) {
            throw std::runtime_error("model weights must be finalized before creating a session");
        }
        codec_embedding_ = model_->codec_embedding_;
        text_embedding_ = model_->text_embedding_;
        text_fc1_ = model_->text_fc1_;
        text_fc1_bias_ = model_->text_fc1_bias_;
        text_fc2_ = model_->text_fc2_;
        text_fc2_bias_ = model_->text_fc2_bias_;
        talker_norm_ = model_->talker_norm_;
        codec_head_ = model_->codec_head_;
        small_to_predictor_ = model_->small_to_predictor_;
        small_to_predictor_bias_ = model_->small_to_predictor_bias_;
        predictor_norm_ = model_->predictor_norm_;
        talker_layers_ = model_->talker_layers_;
        predictor_layers_ = model_->predictor_layers_;
        predictor_embeddings_ = model_->predictor_embeddings_;
        predictor_heads_ = model_->predictor_heads_;
        talker_layers_int8_ = model_->talker_layers_int8_;
        predictor_layers_int8_ = model_->predictor_layers_int8_;
        predictor_heads_int8_ = model_->predictor_heads_int8_;
        codec_head_int8_ = model_->codec_head_int8_;
        small_to_predictor_int8_ = model_->small_to_predictor_int8_;
        text_fc1_int8_ = model_->text_fc1_int8_;
        text_fc2_int8_ = model_->text_fc2_int8_;
        check_cuda(cudaSetDevice(device_index_), "cudaSetDevice");
        try {
            check_cuda(
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags"
            );
            check_cublas(cublasCreate(&cublas_), "cublasCreate");
            check_cublas(cublasSetStream(cublas_, stream_), "cublasSetStream");
            check_cublas(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
            check_cuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
            check_cuda(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
            check_cuda(
                cudaEventCreate(&predictor_start_),
                "cudaEventCreate(predictor start)"
            );
            check_cuda(
                cudaEventCreate(&predictor_stop_),
                "cudaEventCreate(predictor stop)"
            );
            check_cuda(
                cudaEventCreateWithFlags(&frame_codes_ready_, cudaEventDisableTiming),
                "cudaEventCreate(frame codes ready)"
            );
            check_cuda(
                cudaEventCreateWithFlags(&semantic_ready_, cudaEventDisableTiming),
                "cudaEventCreate(semantic ready)"
            );

            const size_t workspace_rows = static_cast<size_t>(max_sequence_length_);
            hidden_ = DeviceBuffer(workspace_rows * kTalkerHidden * sizeof(__nv_bfloat16));
            normalized_ = DeviceBuffer(workspace_rows * kTalkerHidden * sizeof(__nv_bfloat16));
            query_ = DeviceBuffer(workspace_rows * kPredictorQueryWidth * sizeof(__nv_bfloat16));
            key_ = DeviceBuffer(workspace_rows * kTalkerKeyValueWidth * sizeof(__nv_bfloat16));
            value_ = DeviceBuffer(workspace_rows * kTalkerKeyValueWidth * sizeof(__nv_bfloat16));
            attention_ = DeviceBuffer(workspace_rows * kPredictorQueryWidth * sizeof(__nv_bfloat16));
            projection_ = DeviceBuffer(workspace_rows * kTalkerHidden * sizeof(__nv_bfloat16));
            gate_ = DeviceBuffer(workspace_rows * kTalkerIntermediate * sizeof(__nv_bfloat16));
            up_ = DeviceBuffer(workspace_rows * kTalkerIntermediate * sizeof(__nv_bfloat16));
            logits_ = DeviceBuffer(kTalkerVocabulary * sizeof(__nv_bfloat16));
            text_output_ = DeviceBuffer(kTalkerHidden * sizeof(__nv_bfloat16));
            last_hidden_ = DeviceBuffer(kTalkerHidden * sizeof(__nv_bfloat16));
            const size_t packed_elements = static_cast<size_t>(prefill_gemm_capacity_)
                * kTalkerQueryHeads * kTalkerHeadDimension;
            packed_query_ = DeviceBuffer(packed_elements * sizeof(__nv_bfloat16));
            packed_key_ = DeviceBuffer(packed_elements * sizeof(__nv_bfloat16));
            packed_value_ = DeviceBuffer(packed_elements * sizeof(__nv_bfloat16));
            packed_attention_ = DeviceBuffer(packed_elements * sizeof(__nv_bfloat16));
            const size_t score_elements = static_cast<size_t>(kTalkerQueryHeads)
                * prefill_gemm_capacity_ * prefill_gemm_capacity_;
            attention_scores_ = DeviceBuffer(score_elements * sizeof(__nv_bfloat16));
            sampled_token_ = DeviceBuffer(sizeof(int));
            frame_tokens_ = DeviceBuffer(kPredictorSequence * sizeof(int));
            frame_codes_ = DeviceBuffer(kPredictorSequence * sizeof(uint16_t));
            semantic_history_ = DeviceBuffer(
                static_cast<size_t>(max_sequence_length_) * sizeof(int)
            );
            random_state_ = DeviceBuffer(sizeof(uint64_t));
            check_cuda(
                cudaMallocHost(
                    reinterpret_cast<void**>(&host_sampled_token_),
                    sizeof(int)
                ),
                "cudaMallocHost(sampled token)"
            );
            check_cuda(
                cudaMallocHost(
                    reinterpret_cast<void**>(&host_frame_codes_),
                    kPredictorSequence * sizeof(uint16_t)
                ),
                "cudaMallocHost(frame codes)"
            );

            talker_cache_.reserve(kTalkerLayers);
            const size_t talker_cache_bytes = static_cast<size_t>(max_sequence_length_)
                * kTalkerKeyValueWidth * sizeof(__nv_bfloat16);
            for (int layer = 0; layer < kTalkerLayers; ++layer) {
                talker_cache_.push_back({
                    DeviceBuffer(talker_cache_bytes),
                    DeviceBuffer(talker_cache_bytes),
                });
            }

            predictor_cache_.reserve(kPredictorLayers);
            const size_t predictor_cache_bytes = static_cast<size_t>(kPredictorSequence)
                * kPredictorKeyValueWidth * sizeof(__nv_bfloat16);
            for (int layer = 0; layer < kPredictorLayers; ++layer) {
                predictor_cache_.push_back({
                    DeviceBuffer(predictor_cache_bytes),
                    DeviceBuffer(predictor_cache_bytes),
                });
            }
            reset(seed);
        } catch (...) {
            if (host_frame_codes_ != nullptr) {
                cudaFreeHost(host_frame_codes_);
                host_frame_codes_ = nullptr;
            }
            if (host_sampled_token_ != nullptr) {
                cudaFreeHost(host_sampled_token_);
                host_sampled_token_ = nullptr;
            }
            if (semantic_ready_ != nullptr) {
                cudaEventDestroy(semantic_ready_);
                semantic_ready_ = nullptr;
            }
            if (frame_codes_ready_ != nullptr) {
                cudaEventDestroy(frame_codes_ready_);
                frame_codes_ready_ = nullptr;
            }
            if (predictor_stop_ != nullptr) {
                cudaEventDestroy(predictor_stop_);
                predictor_stop_ = nullptr;
            }
            if (predictor_start_ != nullptr) {
                cudaEventDestroy(predictor_start_);
                predictor_start_ = nullptr;
            }
            if (stop_ != nullptr) {
                cudaEventDestroy(stop_);
                stop_ = nullptr;
            }
            if (start_ != nullptr) {
                cudaEventDestroy(start_);
                start_ = nullptr;
            }
            if (cublas_ != nullptr) {
                cublasDestroy(cublas_);
                cublas_ = nullptr;
            }
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
            throw;
        }
    }

    ~TalkerContext() {
        cudaSetDevice(device_index_);
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }
        if (host_frame_codes_ != nullptr) {
            cudaFreeHost(host_frame_codes_);
        }
        if (host_sampled_token_ != nullptr) {
            cudaFreeHost(host_sampled_token_);
        }
        if (semantic_ready_ != nullptr) {
            cudaEventDestroy(semantic_ready_);
        }
        if (frame_codes_ready_ != nullptr) {
            cudaEventDestroy(frame_codes_ready_);
        }
        if (predictor_stop_ != nullptr) {
            cudaEventDestroy(predictor_stop_);
        }
        if (predictor_start_ != nullptr) {
            cudaEventDestroy(predictor_start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (cublas_ != nullptr) {
            cublasDestroy(cublas_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    TalkerContext(const TalkerContext&) = delete;
    TalkerContext& operator=(const TalkerContext&) = delete;

    void reset(uint64_t seed) {
        check_cuda(cudaSetDevice(device_index_), "cudaSetDevice(reset)");
        if (frame_in_flight_) {
            throw std::runtime_error("cannot reset while a device frame lease is in flight");
        }
        position_ = 0;
        generated_semantic_count_ = 0;
        frames_generated_ = 0;
        device_sample_count_ = 0;
        host_sync_count_ = 0;
        current_semantic_token_ = 0;
        phase_ = model_->finalized()
            ? QWEN3_TTS_TALKER_READY
            : QWEN3_TTS_TALKER_CREATED;
        const uint64_t random_state = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
        check_cuda(
            cudaMemcpyAsync(
                random_state_.as<uint64_t>(),
                &random_state,
                sizeof(random_state),
                cudaMemcpyHostToDevice,
                stream_
            ),
            "reset device random state"
        );
        check_cuda(cudaStreamSynchronize(stream_), "synchronize reset random state");
    }

    Qwen3TtsTalkerPrefillResult prefill(
        const int32_t* text_ids,
        const int32_t* codec_ids,
        int token_count,
        const Qwen3TtsSamplingConfig& sampling
    ) {
        check_cuda(cudaSetDevice(device_index_), "cudaSetDevice(prefill)");
        ensure_ready();
        if (poisoned_) {
            throw std::runtime_error("talker session is poisoned and must be destroyed");
        }
        if (frame_in_flight_) {
            throw std::runtime_error("cannot prefill while a device frame lease is in flight");
        }
        if (text_ids == nullptr || codec_ids == nullptr || token_count <= 0) {
            throw std::runtime_error("prefill requires non-empty text and codec ID arrays");
        }
        if (token_count > max_sequence_length_) {
            throw std::runtime_error("prompt exceeds the configured KV-cache capacity");
        }
        position_ = 0;
        generated_semantic_count_ = 0;
        for (int index = 0; index < token_count; ++index) {
            prepare_prompt_embedding(
                text_ids[index],
                codec_ids[index],
                hidden_.as<__nv_bfloat16>() + static_cast<size_t>(index) * kTalkerHidden
            );
        }
        trace_active_ = trace_enabled_;
        trace_vector(
            "input",
            -1,
            hidden_.as<__nv_bfloat16>()
                + static_cast<size_t>(token_count - 1) * kTalkerHidden,
            kTalkerHidden
        );
        dump_stage(
            "input",
            hidden_.as<__nv_bfloat16>()
                + static_cast<size_t>(token_count - 1) * kTalkerHidden,
            kTalkerHidden
        );
        const float total_gpu_ms = run_talker_prefill(token_count);
        trace_active_ = false;
        position_ = token_count;
        sample_logits_device(kTalkerVocabulary, sampling, true);
        const int first = copy_sampled_token_to_host();
        current_semantic_token_ = static_cast<uint16_t>(first);
        phase_ = first == kCodecEos
            ? QWEN3_TTS_TALKER_ENDED
            : QWEN3_TTS_TALKER_PREFILLED;
        Qwen3TtsTalkerPrefillResult result{};
        result.first_semantic_token = static_cast<uint16_t>(first);
        result.prompt_tokens = static_cast<uint32_t>(token_count);
        result.talker_gpu_milliseconds = total_gpu_ms;
        return result;
    }

    Qwen3TtsDeviceFrameViewV2 begin_frame(
        uint16_t semantic_token,
        int trailing_text_token,
        const Qwen3TtsSamplingConfig& talker_sampling,
        const Qwen3TtsSamplingConfig& predictor_sampling,
        bool copy_codes_to_host
    ) {
        check_cuda(cudaSetDevice(device_index_), "cudaSetDevice(begin frame)");
        ensure_ready();
        if (poisoned_) {
            throw std::runtime_error("talker session is poisoned and must be destroyed");
        }
        if (frame_in_flight_) {
            throw std::runtime_error("a device frame lease is already in flight");
        }
        if (phase_ != QWEN3_TTS_TALKER_PREFILLED) {
            throw std::runtime_error("talker must be prefilled before frame generation");
        }
        if (semantic_token == kCodecEos) {
            throw std::runtime_error("EOS must not be expanded into a codec frame");
        }
        if (semantic_token >= kTalkerVocabulary) {
            throw std::runtime_error("semantic token is outside the talker vocabulary");
        }
        if (semantic_token != current_semantic_token_) {
            throw std::runtime_error("semantic token does not match talker session state");
        }
        if (trailing_text_token < 0 || trailing_text_token >= kTextVocabulary) {
            throw std::runtime_error("next-frame text token is outside the text vocabulary");
        }
        if (position_ >= max_sequence_length_) {
            throw std::runtime_error("talker KV cache is full");
        }

        uint64_t lease_id = next_lease_id_++;
        if (next_lease_id_ == 0) {
            next_lease_id_ = 1;
        }
        frame_in_flight_ = true;
        pending_host_code_copy_ = copy_codes_to_host;
        pending_lease_id_ = lease_id;
        pending_talker_position_ = static_cast<uint32_t>(position_);

        try {
            check_cuda(
                qwen3_tts::launch_store_token(
                    frame_tokens_.as<int>(),
                    0,
                    semantic_token,
                    stream_
                ),
                "store semantic frame token"
            );
            check_cuda(
                cudaEventRecord(predictor_start_, stream_),
                "cudaEventRecord(predictor start)"
            );

            run_predictor_position(last_hidden_.as<__nv_bfloat16>(), 0, -1);
            check_cuda(
                qwen3_tts::launch_gather_embedding(
                    codec_embedding_,
                    kTalkerVocabulary,
                    kTalkerHidden,
                    frame_tokens_.as<int>(),
                    text_output_.as<__nv_bfloat16>(),
                    stream_
                ),
                "gather semantic predictor embedding"
            );
            run_predictor_position(text_output_.as<__nv_bfloat16>(), 1, 0);
            sample_logits_device(kPredictorVocabulary, predictor_sampling, false);
            check_cuda(
                qwen3_tts::launch_store_sampled_token(
                    frame_tokens_.as<int>(),
                    1,
                    sampled_token_.as<int>(),
                    stream_
                ),
                "store predictor codebook token"
            );

            for (int residual = 1; residual < kResidualCodebooks; ++residual) {
                check_cuda(
                    qwen3_tts::launch_gather_embedding(
                        predictor_embeddings_[residual - 1],
                        kPredictorVocabulary,
                        kTalkerHidden,
                        frame_tokens_.as<int>() + residual,
                        text_output_.as<__nv_bfloat16>(),
                        stream_
                    ),
                    "gather residual predictor embedding"
                );
                run_predictor_position(
                    text_output_.as<__nv_bfloat16>(),
                    residual + 1,
                    residual
                );
                sample_logits_device(kPredictorVocabulary, predictor_sampling, false);
                check_cuda(
                    qwen3_tts::launch_store_sampled_token(
                        frame_tokens_.as<int>(),
                        residual + 1,
                        sampled_token_.as<int>(),
                        stream_
                    ),
                    "store residual codebook token"
                );
            }
            check_cuda(
                cudaEventRecord(predictor_stop_, stream_),
                "cudaEventRecord(predictor stop)"
            );
            check_cuda(
                qwen3_tts::launch_pack_frame_codes(
                    frame_tokens_.as<int>(),
                    frame_codes_.as<uint16_t>(),
                    stream_
                ),
                "pack codec frame"
            );
            check_cuda(
                cudaEventRecord(frame_codes_ready_, stream_),
                "cudaEventRecord(frame codes ready)"
            );
            if (copy_codes_to_host) {
                check_cuda(
                    cudaMemcpyAsync(
                        host_frame_codes_,
                        frame_codes_.as<uint16_t>(),
                        kPredictorSequence * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost,
                        stream_
                    ),
                    "copy codec frame to pinned host memory"
                );
            }

            prepare_generated_embedding(trailing_text_token);
            check_cuda(cudaEventRecord(start_, stream_), "cudaEventRecord(talker start)");
            run_talker_step(position_, true);
            check_cuda(cudaEventRecord(stop_, stream_), "cudaEventRecord(talker stop)");
            ++position_;
            sample_logits_device(kTalkerVocabulary, talker_sampling, true);
            enqueue_sampled_token_to_host();
            check_cuda(
                cudaEventRecord(semantic_ready_, stream_),
                "cudaEventRecord(semantic ready)"
            );
        } catch (...) {
            poisoned_ = true;
            cudaStreamSynchronize(stream_);
            frame_in_flight_ = false;
            pending_host_code_copy_ = false;
            pending_lease_id_ = 0;
            throw;
        }

        Qwen3TtsDeviceFrameViewV2 view{};
        view.struct_size = sizeof(Qwen3TtsDeviceFrameViewV2);
        view.code_count = kPredictorSequence;
        view.device_codes = frame_codes_.as<uint16_t>();
        view.ready_event = reinterpret_cast<Qwen3TtsCudaEventHandle>(frame_codes_ready_);
        view.lease_id = lease_id;
        view.device_index = device_index_;
        return view;
    }

    Qwen3TtsCodecFrameResult finish_frame(
        uint64_t lease_id,
        Qwen3TtsCudaEventHandle consumer_done_event
    ) {
        check_cuda(cudaSetDevice(device_index_), "cudaSetDevice(finish frame)");
        ensure_ready();
        if (poisoned_) {
            throw std::runtime_error("talker session is poisoned and must be destroyed");
        }
        if (!frame_in_flight_) {
            throw std::runtime_error("no device frame lease is in flight");
        }
        if (lease_id == 0 || lease_id != pending_lease_id_) {
            throw std::runtime_error("device frame lease ID is stale or invalid");
        }
        if (consumer_done_event != nullptr) {
            check_cuda(
                cudaStreamWaitEvent(
                    stream_,
                    reinterpret_cast<cudaEvent_t>(consumer_done_event),
                    0
                ),
                "cudaStreamWaitEvent(frame consumer done)"
            );
        }
        check_cuda(
            cudaEventSynchronize(semantic_ready_),
            "cudaEventSynchronize(semantic ready)"
        );
        ++host_sync_count_;
        const int next_semantic = validated_host_sampled_token();

        Qwen3TtsCodecFrameResult result{};
        result.talker_position = pending_talker_position_;
        check_cuda(
            cudaEventElapsedTime(
                &result.predictor_gpu_milliseconds,
                predictor_start_,
                predictor_stop_
            ),
            "cudaEventElapsedTime(predictor)"
        );
        check_cuda(
            cudaEventElapsedTime(&result.talker_gpu_milliseconds, start_, stop_),
            "cudaEventElapsedTime(talker)"
        );
        if (pending_host_code_copy_) {
            std::copy_n(host_frame_codes_, kPredictorSequence, result.codes);
        }
        result.next_semantic_token = static_cast<uint16_t>(next_semantic);
        result.ended_by_eos = next_semantic == kCodecEos ? 1 : 0;
        current_semantic_token_ = result.next_semantic_token;
        ++frames_generated_;
        phase_ = result.ended_by_eos != 0
            ? QWEN3_TTS_TALKER_ENDED
            : QWEN3_TTS_TALKER_PREFILLED;
        frame_in_flight_ = false;
        pending_host_code_copy_ = false;
        pending_lease_id_ = 0;
        return result;
    }

    Qwen3TtsCodecFrameResult next_frame(
        uint16_t semantic_token,
        int trailing_text_token,
        const Qwen3TtsSamplingConfig& talker_sampling,
        const Qwen3TtsSamplingConfig& predictor_sampling
    ) {
        const Qwen3TtsDeviceFrameViewV2 view = begin_frame(
            semantic_token,
            trailing_text_token,
            talker_sampling,
            predictor_sampling,
            true
        );
        return finish_frame(view.lease_id, nullptr);
    }

    uint16_t current_semantic_token() const noexcept {
        return current_semantic_token_;
    }

    /* Lockstep batched frame generation. Enqueues one complete codec frame for
     * every session in one pass so each weight matrix is read once per frame
     * instead of once per session. KV caches, sampling state, semantic
     * histories, random states, pinned outputs, and lifecycle events remain
     * session-local; every session is finished individually through the
     * existing finish_frame lease. All sessions must belong to `model`. */
    static void batch_begin_frames(
        TalkerModel* model,
        TalkerContext** s,
        int n,
        const int* trailing_text_tokens,
        const Qwen3TtsSamplingConfig* talker_sampling,
        const Qwen3TtsSamplingConfig* predictor_sampling,
        Qwen3TtsDeviceFrameViewV2* views
    ) {
        if (model == nullptr || s == nullptr || n <= 0 || n > BatchWorkspace::kCapacity) {
            throw std::runtime_error("invalid batch frame arguments");
        }
        BatchWorkspace& w = model->batch_workspace();
        check_cuda(cudaSetDevice(w.device_index), "cudaSetDevice(batch frame)");

        // Per-session preamble: mirrors begin_frame validation exactly.
        for (int i = 0; i < n; ++i) {
            TalkerContext* c = s[i];
            c->ensure_ready();
            if (c->model_.get() != model) {
                throw std::runtime_error("batched session belongs to a different model");
            }
            if (c->poisoned_) {
                throw std::runtime_error("talker session is poisoned and must be destroyed");
            }
            if (c->frame_in_flight_) {
                throw std::runtime_error("a device frame lease is already in flight");
            }
            if (c->phase_ != QWEN3_TTS_TALKER_PREFILLED) {
                throw std::runtime_error("talker must be prefilled before frame generation");
            }
            if (c->current_semantic_token_ == kCodecEos) {
                throw std::runtime_error("EOS must not be expanded into a codec frame");
            }
            if (c->current_semantic_token_ >= kTalkerVocabulary) {
                throw std::runtime_error("semantic token is outside the talker vocabulary");
            }
            if (trailing_text_tokens[i] < 0 || trailing_text_tokens[i] >= kTextVocabulary) {
                throw std::runtime_error("next-frame text token is outside the text vocabulary");
            }
            if (c->position_ >= c->max_sequence_length_) {
                throw std::runtime_error("talker KV cache is full");
            }
            if (c->generated_semantic_count_ >= c->max_sequence_length_) {
                throw std::runtime_error("semantic history exceeds configured sequence capacity");
            }
            for (const Qwen3TtsSamplingConfig* config :
                 {&talker_sampling[i], &predictor_sampling[i]}) {
                if (config->top_k < 0 || config->top_p <= 0.0f || config->top_p > 1.0f
                    || config->temperature <= 0.0f || config->repetition_penalty <= 0.0f) {
                    throw std::runtime_error("invalid sampling configuration");
                }
            }
        }

        // Allocate leases only after every session validated.
        for (int i = 0; i < n; ++i) {
            TalkerContext* c = s[i];
            uint64_t lease_id = c->next_lease_id_++;
            if (c->next_lease_id_ == 0) {
                c->next_lease_id_ = 1;
            }
            c->frame_in_flight_ = true;
            c->pending_host_code_copy_ = false;
            c->pending_lease_id_ = lease_id;
            c->pending_talker_position_ = static_cast<uint32_t>(c->position_);
        }

        // Host-visible pinned tables. The captured upload nodes read these at
        // every replay, so refreshing them re-parameterizes the graph without
        // re-instantiation.
        for (int slot = 0; slot < kPredictorSequence; ++slot) {
            for (int i = 0; i < n; ++i) {
                w.pinned_positions[slot * BatchWorkspace::kCapacity + i] = slot;
            }
        }
        for (int i = 0; i < n; ++i) {
            w.pinned_positions[kPredictorSequence * BatchWorkspace::kCapacity + i] =
                s[i]->position_;
            w.pinned_token_bases[i] = s[i]->frame_tokens_.as<int>();
            w.pinned_trailing[i] = trailing_text_tokens[i];
            w.pinned_counts[i] = s[i]->generated_semantic_count_;
        }
        for (int layer = 0; layer < kTalkerLayers; ++layer) {
            for (int i = 0; i < n; ++i) {
                w.pinned_kv_bases[(layer * 2) * BatchWorkspace::kCapacity + i] =
                    s[i]->talker_cache_[layer].key.as<__nv_bfloat16>();
                w.pinned_kv_bases[(layer * 2 + 1) * BatchWorkspace::kCapacity + i] =
                    s[i]->talker_cache_[layer].value.as<__nv_bfloat16>();
            }
        }
        for (int layer = 0; layer < kPredictorLayers; ++layer) {
            const int slot = kTalkerLayers + layer;
            for (int i = 0; i < n; ++i) {
                w.pinned_kv_bases[(slot * 2) * BatchWorkspace::kCapacity + i] =
                    s[i]->predictor_cache_[layer].key.as<__nv_bfloat16>();
                w.pinned_kv_bases[(slot * 2 + 1) * BatchWorkspace::kCapacity + i] =
                    s[i]->predictor_cache_[layer].value.as<__nv_bfloat16>();
            }
        }

        // The graph stays valid while the session tuple, per-session sampling
        // configuration, and KV capacities are unchanged. Everything that
        // varies per frame flows through device memory or the pinned tables.
        int max_capacity = kPredictorSequence;
        std::vector<uint64_t> key;
        key.reserve(1 + static_cast<size_t>(n) * 12);
        key.push_back(static_cast<uint64_t>(n));
        for (int i = 0; i < n; ++i) {
            max_capacity = std::max(max_capacity, s[i]->max_sequence_length_);
            key.push_back(reinterpret_cast<uint64_t>(s[i]));
            key.push_back(static_cast<uint64_t>(s[i]->max_sequence_length_));
            for (const Qwen3TtsSamplingConfig* config :
                 {&talker_sampling[i], &predictor_sampling[i]}) {
                uint32_t top_p_bits = 0;
                uint32_t temperature_bits = 0;
                uint32_t penalty_bits = 0;
                std::memcpy(&top_p_bits, &config->top_p, sizeof(top_p_bits));
                std::memcpy(&temperature_bits, &config->temperature, sizeof(temperature_bits));
                std::memcpy(&penalty_bits, &config->repetition_penalty, sizeof(penalty_bits));
                key.push_back(static_cast<uint64_t>(config->do_sample));
                key.push_back(static_cast<uint64_t>(config->top_k));
                key.push_back(top_p_bits);
                key.push_back(temperature_bits);
                key.push_back(penalty_bits);
            }
        }

        TalkerContext* c0 = s[0];
        const int* const* token_bases = w.device_token_bases.as<const int*>();
        const int* const* trailing_ptrs = w.device_trailing_ptrs.as<const int*>();

        const auto record_event = [&w](cudaEvent_t event, bool capturing) {
            if (capturing) {
                check_cuda(
                    cudaEventRecordWithFlags(event, w.stream, cudaEventRecordExternal),
                    "cudaEventRecordWithFlags(batch)"
                );
            } else {
                check_cuda(cudaEventRecord(event, w.stream), "cudaEventRecord(batch)");
            }
        };

        const auto enqueue_frame = [&](bool capturing) {
            check_cuda(
                cudaMemcpyAsync(
                    w.device_positions.as<int>(),
                    w.pinned_positions,
                    static_cast<size_t>(BatchWorkspace::kPositionSlots)
                        * BatchWorkspace::kCapacity * sizeof(int),
                    cudaMemcpyHostToDevice,
                    w.stream
                ),
                "upload batch positions"
            );
            check_cuda(
                cudaMemcpyAsync(
                    w.device_kv_bases.as<__nv_bfloat16*>(),
                    w.pinned_kv_bases,
                    static_cast<size_t>(BatchWorkspace::kKvPointerCount)
                        * sizeof(__nv_bfloat16*),
                    cudaMemcpyHostToDevice,
                    w.stream
                ),
                "upload batch kv bases"
            );
            check_cuda(
                cudaMemcpyAsync(
                    w.device_token_bases.as<const int*>(),
                    w.pinned_token_bases,
                    BatchWorkspace::kCapacity * sizeof(const int*),
                    cudaMemcpyHostToDevice,
                    w.stream
                ),
                "upload batch token bases"
            );
            check_cuda(
                cudaMemcpyAsync(
                    w.device_trailing.as<int>(),
                    w.pinned_trailing,
                    BatchWorkspace::kCapacity * sizeof(int),
                    cudaMemcpyHostToDevice,
                    w.stream
                ),
                "upload batch trailing tokens"
            );
            check_cuda(
                cudaMemcpyAsync(
                    w.device_counts.as<int>(),
                    w.pinned_counts,
                    BatchWorkspace::kCapacity * sizeof(int),
                    cudaMemcpyHostToDevice,
                    w.stream
                ),
                "upload batch history counts"
            );

            // frame_tokens[0] = current semantic token, taken from each
            // session's device-resident last sampled token.
            for (int i = 0; i < n; ++i) {
                check_cuda(
                    qwen3_tts::launch_store_sampled_token(
                        s[i]->frame_tokens_.as<int>(),
                        0,
                        s[i]->sampled_token_.as<int>(),
                        w.stream
                    ),
                    "store semantic frame token"
                );
                record_event(s[i]->predictor_start_, capturing);
            }

            // ---- Predictor: 16 lockstep positions ----
            for (int position = 0; position < kPredictorSequence; ++position) {
                if (position == 0) {
                    for (int i = 0; i < n; ++i) {
                        check_cuda(
                            cudaMemcpyAsync(
                                w.text_input.as<__nv_bfloat16>()
                                    + static_cast<size_t>(i) * kTalkerHidden,
                                s[i]->last_hidden_.as<__nv_bfloat16>(),
                                kTalkerHidden * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToDevice,
                                w.stream
                            ),
                            "gather last talker hidden"
                        );
                    }
                } else if (position == 1) {
                    check_cuda(
                        qwen3_tts::launch_gather_embedding_rows(
                            c0->codec_embedding_,
                            kTalkerVocabulary,
                            kTalkerHidden,
                            token_bases,
                            0,
                            w.text_input.as<__nv_bfloat16>(),
                            n,
                            w.stream
                        ),
                        "gather semantic predictor embedding"
                    );
                } else {
                    const int residual = position - 1;
                    check_cuda(
                        qwen3_tts::launch_gather_embedding_rows(
                            c0->predictor_embeddings_[residual - 1],
                            kPredictorVocabulary,
                            kTalkerHidden,
                            token_bases,
                            residual,
                            w.text_input.as<__nv_bfloat16>(),
                            n,
                            w.stream
                        ),
                        "gather residual predictor embedding"
                    );
                }
                batch_gemm_quantized(
                    w,
                    c0->small_to_predictor_int8_,
                    c0->small_to_predictor_,
                    w.text_input.as<__nv_bfloat16>(),
                    w.hidden.as<__nv_bfloat16>(),
                    kTalkerHidden,
                    kPredictorHidden,
                    n
                );
                check_cuda(
                    qwen3_tts::launch_bias_activation_rows(
                        w.hidden.as<__nv_bfloat16>(),
                        c0->small_to_predictor_bias_,
                        n,
                        kPredictorHidden,
                        false,
                        w.stream
                    ),
                    "predictor input projection bias"
                );
                batch_run_decoder(w, s, n, c0->predictor_layers_, true,
                                  kPredictorDimensions, position, kPredictorSequence);
                const int head = position - 1;
                if (head >= 0) {
                    check_cuda(
                        qwen3_tts::launch_rms_norm_rows(
                            w.hidden.as<__nv_bfloat16>(),
                            c0->predictor_norm_,
                            w.normalized.as<__nv_bfloat16>(),
                            n,
                            kPredictorHidden,
                            kRmsEpsilon,
                            w.stream
                        ),
                        "predictor final RMSNorm"
                    );
                    batch_gemm_quantized(
                        w,
                        c0->predictor_heads_int8_[head],
                                    c0->predictor_heads_[head],
                        w.normalized.as<__nv_bfloat16>(),
                        w.logits.as<__nv_bfloat16>(),
                        kPredictorHidden,
                        kPredictorVocabulary,
                        n
                    );
                    for (int i = 0; i < n; ++i) {
                        check_cuda(
                            qwen3_tts::launch_sample_logits_at(
                                w.logits.as<__nv_bfloat16>()
                                    + static_cast<size_t>(i) * kPredictorVocabulary,
                                kPredictorVocabulary,
                                false,
                                kCodecEos,
                                s[i]->semantic_history_.as<int>(),
                                w.device_counts.as<int>() + i,
                                predictor_sampling[i].do_sample,
                                predictor_sampling[i].top_k,
                                predictor_sampling[i].top_p,
                                predictor_sampling[i].temperature,
                                predictor_sampling[i].repetition_penalty,
                                s[i]->random_state_.as<uint64_t>(),
                                s[i]->sampled_token_.as<int>(),
                                w.stream
                            ),
                            "sample predictor logits"
                        );
                        check_cuda(
                            qwen3_tts::launch_store_sampled_token(
                                s[i]->frame_tokens_.as<int>(),
                                position,
                                s[i]->sampled_token_.as<int>(),
                                w.stream
                            ),
                            "store predictor codebook token"
                        );
                    }
                }
            }
            for (int i = 0; i < n; ++i) {
                record_event(s[i]->predictor_stop_, capturing);
                check_cuda(
                    qwen3_tts::launch_pack_frame_codes(
                        s[i]->frame_tokens_.as<int>(),
                        s[i]->frame_codes_.as<uint16_t>(),
                        w.stream
                    ),
                    "pack codec frame"
                );
                record_event(s[i]->frame_codes_ready_, capturing);
            }

            // ---- Next talker embedding: codes + trailing text ----
            check_cuda(
                qwen3_tts::launch_gather_embedding_rows(
                    c0->codec_embedding_,
                    kTalkerVocabulary,
                    kTalkerHidden,
                    token_bases,
                    0,
                    w.hidden.as<__nv_bfloat16>(),
                    n,
                    w.stream
                ),
                "gather semantic talker embedding"
            );
            for (int residual = 0; residual < kResidualCodebooks; ++residual) {
                check_cuda(
                    qwen3_tts::launch_add_embedding_rows(
                        c0->predictor_embeddings_[residual],
                        kPredictorVocabulary,
                        kTalkerHidden,
                        token_bases,
                        residual + 1,
                        w.hidden.as<__nv_bfloat16>(),
                        n,
                        w.stream
                    ),
                    "add residual codec embedding"
                );
            }
            check_cuda(
                qwen3_tts::launch_gather_embedding_rows(
                    c0->text_embedding_,
                    kTextVocabulary,
                    kTalkerHidden,
                    trailing_ptrs,
                    0,
                    w.text_input.as<__nv_bfloat16>(),
                    n,
                    w.stream
                ),
                "gather trailing text embedding"
            );
            batch_gemm_quantized(
                w,
                c0->text_fc1_int8_,
                c0->text_fc1_,
                w.text_input.as<__nv_bfloat16>(),
                w.projection.as<__nv_bfloat16>(),
                kTalkerHidden,
                kTalkerHidden,
                n
            );
            check_cuda(
                qwen3_tts::launch_bias_activation_rows(
                    w.projection.as<__nv_bfloat16>(),
                    c0->text_fc1_bias_,
                    n,
                    kTalkerHidden,
                    true,
                    w.stream
                ),
                "text projection fc1 activation"
            );
            batch_gemm_quantized(
                w,
                c0->text_fc2_int8_,
                c0->text_fc2_,
                w.projection.as<__nv_bfloat16>(),
                w.attention.as<__nv_bfloat16>(),
                kTalkerHidden,
                kTalkerHidden,
                n
            );
            check_cuda(
                qwen3_tts::launch_bias_activation_rows(
                    w.attention.as<__nv_bfloat16>(),
                    c0->text_fc2_bias_,
                    n,
                    kTalkerHidden,
                    false,
                    w.stream
                ),
                "text projection fc2 bias"
            );
            check_cuda(
                qwen3_tts::launch_add_in_place(
                    w.hidden.as<__nv_bfloat16>(),
                    w.attention.as<__nv_bfloat16>(),
                    n * kTalkerHidden,
                    w.stream
                ),
                "add trailing text embedding"
            );

            // ---- Talker step at each session's own position ----
            for (int i = 0; i < n; ++i) {
                record_event(s[i]->start_, capturing);
            }
            batch_run_decoder(w, s, n, c0->talker_layers_, false,
                              kTalkerDimensions, kPredictorSequence, max_capacity);
            check_cuda(
                qwen3_tts::launch_rms_norm_rows(
                    w.hidden.as<__nv_bfloat16>(),
                    c0->talker_norm_,
                    w.normalized.as<__nv_bfloat16>(),
                    n,
                    kTalkerHidden,
                    kRmsEpsilon,
                    w.stream
                ),
                "talker final RMSNorm"
            );
            for (int i = 0; i < n; ++i) {
                check_cuda(
                    cudaMemcpyAsync(
                        s[i]->last_hidden_.as<__nv_bfloat16>(),
                        w.normalized.as<__nv_bfloat16>()
                            + static_cast<size_t>(i) * kTalkerHidden,
                        kTalkerHidden * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice,
                        w.stream
                    ),
                    "copy talker hidden"
                );
            }
            batch_gemm_quantized(
                w,
                c0->codec_head_int8_,
                c0->codec_head_,
                w.normalized.as<__nv_bfloat16>(),
                w.logits.as<__nv_bfloat16>(),
                kTalkerHidden,
                kTalkerVocabulary,
                n
            );
            for (int i = 0; i < n; ++i) {
                record_event(s[i]->stop_, capturing);
                check_cuda(
                    qwen3_tts::launch_sample_logits_at(
                        w.logits.as<__nv_bfloat16>()
                            + static_cast<size_t>(i) * kTalkerVocabulary,
                        kTalkerVocabulary,
                        true,
                        kCodecEos,
                        s[i]->semantic_history_.as<int>(),
                        w.device_counts.as<int>() + i,
                        talker_sampling[i].do_sample,
                        talker_sampling[i].top_k,
                        talker_sampling[i].top_p,
                        talker_sampling[i].temperature,
                        talker_sampling[i].repetition_penalty,
                        s[i]->random_state_.as<uint64_t>(),
                        s[i]->sampled_token_.as<int>(),
                        w.stream
                    ),
                    "sample talker logits"
                );
                check_cuda(
                    qwen3_tts::launch_store_sampled_token_at(
                        s[i]->semantic_history_.as<int>(),
                        w.device_counts.as<int>() + i,
                        s[i]->sampled_token_.as<int>(),
                        w.stream
                    ),
                    "append sampled semantic token"
                );
                check_cuda(
                    cudaMemcpyAsync(
                        s[i]->host_sampled_token_,
                        s[i]->sampled_token_.as<int>(),
                        sizeof(int),
                        cudaMemcpyDeviceToHost,
                        w.stream
                    ),
                    "copy sampled token to pinned host memory"
                );
                record_event(s[i]->semantic_ready_, capturing);
            }
        };

        try {
            // Order the batch stream after each session's prior work. These
            // waits stay outside the captured graph.
            for (int i = 0; i < n; ++i) {
                check_cuda(
                    cudaEventRecord(w.join_events[i], s[i]->stream_),
                    "cudaEventRecord(batch join)"
                );
                check_cuda(
                    cudaStreamWaitEvent(w.stream, w.join_events[i], 0),
                    "cudaStreamWaitEvent(batch join)"
                );
            }

            if (w.graph_key != key) {
                w.destroy_graph();
                w.graph_key = key;
            }
            if (w.graph_exec != nullptr) {
                check_cuda(cudaGraphLaunch(w.graph_exec, w.stream), "cudaGraphLaunch(batch frame)");
            } else if (w.uncaptured_runs < 1) {
                // First run for this tuple executes uncaptured so cuBLAS can
                // finish heuristic and workspace setup outside capture.
                enqueue_frame(false);
                ++w.uncaptured_runs;
            } else {
                check_cuda(
                    cudaStreamBeginCapture(w.stream, cudaStreamCaptureModeThreadLocal),
                    "cudaStreamBeginCapture(batch frame)"
                );
                try {
                    enqueue_frame(true);
                } catch (...) {
                    cudaGraph_t aborted = nullptr;
                    cudaStreamEndCapture(w.stream, &aborted);
                    if (aborted != nullptr) {
                        cudaGraphDestroy(aborted);
                    }
                    throw;
                }
                check_cuda(
                    cudaStreamEndCapture(w.stream, &w.graph),
                    "cudaStreamEndCapture(batch frame)"
                );
                check_cuda(
                    cudaGraphInstantiate(&w.graph_exec, w.graph, 0),
                    "cudaGraphInstantiate(batch frame)"
                );
                check_cuda(
                    cudaGraphLaunch(w.graph_exec, w.stream),
                    "cudaGraphLaunch(first batch frame)"
                );
            }

            for (int i = 0; i < n; ++i) {
                s[i]->device_sample_count_ += static_cast<uint64_t>(kPredictorSequence);
                ++s[i]->generated_semantic_count_;
                ++s[i]->position_;
                check_cuda(
                    cudaStreamWaitEvent(s[i]->stream_, s[i]->semantic_ready_, 0),
                    "cudaStreamWaitEvent(session rejoin)"
                );
            }
        } catch (...) {
            for (int i = 0; i < n; ++i) {
                s[i]->poisoned_ = true;
                s[i]->frame_in_flight_ = false;
                s[i]->pending_host_code_copy_ = false;
                s[i]->pending_lease_id_ = 0;
            }
            w.destroy_graph();
            cudaStreamSynchronize(w.stream);
            throw;
        }

        for (int i = 0; i < n; ++i) {
            Qwen3TtsDeviceFrameViewV2 view{};
            view.struct_size = sizeof(Qwen3TtsDeviceFrameViewV2);
            view.code_count = kPredictorSequence;
            view.device_codes = s[i]->frame_codes_.as<uint16_t>();
            view.ready_event =
                reinterpret_cast<Qwen3TtsCudaEventHandle>(s[i]->frame_codes_ready_);
            view.lease_id = s[i]->pending_lease_id_;
            view.device_index = s[i]->device_index_;
            views[i] = view;
        }
    }

private:
    static void batch_gemm_quantized(
        BatchWorkspace& w,
        const QuantizedTensor& quantized,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* input,
        __nv_bfloat16* output,
        int input_features,
        int output_features,
        int rows
    ) {
        if (quantized.data != nullptr) {
            check_cuda(
                qwen3_tts::launch_int8_gemm_rows(
                    quantized.data,
                    quantized.scales,
                    input,
                    output,
                    input_features,
                    output_features,
                    rows,
                    w.stream
                ),
                "int8 batch decode GEMM"
            );
            return;
        }
        batch_gemm(w, weight, input, output, input_features, output_features, rows);
    }

    static void batch_gemm(
        BatchWorkspace& w,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* input,
        __nv_bfloat16* output,
        int input_features,
        int output_features,
        int rows
    ) {
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        check_cublas(
            cublasGemmEx(
                w.cublas,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                output_features,
                rows,
                input_features,
                &alpha,
                weight,
                CUDA_R_16BF,
                input_features,
                input,
                CUDA_R_16BF,
                input_features,
                &beta,
                output,
                CUDA_R_16BF,
                output_features,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "cublasGemmEx(batch)"
        );
    }

    static void batch_run_decoder(
        BatchWorkspace& w,
        TalkerContext** s,
        int n,
        const std::vector<DecoderWeights>& layers,
        bool predictor,
        const ModelDimensions& dims,
        int position_slot,
        int span_capacity
    ) {
        const int* positions = w.device_positions.as<int>()
            + static_cast<size_t>(position_slot) * BatchWorkspace::kCapacity;
        const int layer_slot_base = predictor ? kTalkerLayers : 0;
        const std::vector<DecoderWeightsInt8>& int8_layers = predictor
            ? s[0]->predictor_layers_int8_
            : s[0]->talker_layers_int8_;
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const DecoderWeights& layer = layers[layer_index];
            const DecoderWeightsInt8 quantized = int8_layers.empty()
                ? DecoderWeightsInt8{}
                : int8_layers[layer_index];
            check_cuda(
                qwen3_tts::launch_rms_norm_rows(
                    w.hidden.as<__nv_bfloat16>(),
                    layer.input_norm,
                    w.normalized.as<__nv_bfloat16>(),
                    n,
                    dims.hidden,
                    kRmsEpsilon,
                    w.stream
                ),
                "input RMSNorm"
            );
            batch_gemm_quantized(
                w,
                quantized.q_projection,
                layer.q_projection,
                w.normalized.as<__nv_bfloat16>(),
                w.query.as<__nv_bfloat16>(),
                dims.hidden,
                dims.query_width,
                n
            );
            batch_gemm_quantized(
                w,
                quantized.k_projection,
                layer.k_projection,
                w.normalized.as<__nv_bfloat16>(),
                w.key.as<__nv_bfloat16>(),
                dims.hidden,
                dims.key_value_width,
                n
            );
            batch_gemm_quantized(
                w,
                quantized.v_projection,
                layer.v_projection,
                w.normalized.as<__nv_bfloat16>(),
                w.value.as<__nv_bfloat16>(),
                dims.hidden,
                dims.key_value_width,
                n
            );
            check_cuda(
                qwen3_tts::launch_head_rms_norm_rows(
                    w.query.as<__nv_bfloat16>(),
                    layer.q_norm,
                    n,
                    dims.query_heads,
                    dims.head_dimension,
                    kRmsEpsilon,
                    w.stream
                ),
                "query head RMSNorm"
            );
            check_cuda(
                qwen3_tts::launch_head_rms_norm_rows(
                    w.key.as<__nv_bfloat16>(),
                    layer.k_norm,
                    n,
                    dims.key_value_heads,
                    dims.head_dimension,
                    kRmsEpsilon,
                    w.stream
                ),
                "key head RMSNorm"
            );
            check_cuda(
                qwen3_tts::launch_rope_rows_at(
                    w.query.as<__nv_bfloat16>(),
                    n,
                    dims.query_heads,
                    dims.head_dimension,
                    positions,
                    kRopeTheta,
                    w.stream
                ),
                "query RoPE"
            );
            check_cuda(
                qwen3_tts::launch_rope_rows_at(
                    w.key.as<__nv_bfloat16>(),
                    n,
                    dims.key_value_heads,
                    dims.head_dimension,
                    positions,
                    kRopeTheta,
                    w.stream
                ),
                "key RoPE"
            );
            __nv_bfloat16* const* key_bases = w.device_kv_bases.as<__nv_bfloat16*>()
                + static_cast<size_t>(layer_slot_base + layer_index) * 2
                    * BatchWorkspace::kCapacity;
            __nv_bfloat16* const* value_bases =
                key_bases + BatchWorkspace::kCapacity;
            check_cuda(
                qwen3_tts::launch_kv_scatter_rows(
                    w.key.as<__nv_bfloat16>(),
                    key_bases,
                    positions,
                    n,
                    dims.key_value_width,
                    w.stream
                ),
                "append key cache"
            );
            check_cuda(
                qwen3_tts::launch_kv_scatter_rows(
                    w.value.as<__nv_bfloat16>(),
                    value_bases,
                    positions,
                    n,
                    dims.key_value_width,
                    w.stream
                ),
                "append value cache"
            );
            // The INT8 research mode also opts into the flash-style attention
            // sweep; the BF16 contract path keeps the exact kernel.
            if (!int8_layers.empty()) {
                check_cuda(
                    qwen3_tts::launch_batch_causal_gqa_attention_fast(
                        w.query.as<__nv_bfloat16>(),
                        key_bases,
                        value_bases,
                        positions,
                        w.attention.as<__nv_bfloat16>(),
                        n,
                        dims.query_heads,
                        dims.key_value_heads,
                        dims.head_dimension,
                        span_capacity,
                        w.stream
                    ),
                    "causal GQA attention"
                );
            } else {
                check_cuda(
                    qwen3_tts::launch_batch_causal_gqa_attention(
                        w.query.as<__nv_bfloat16>(),
                        key_bases,
                        value_bases,
                        positions,
                        w.attention.as<__nv_bfloat16>(),
                        n,
                        dims.query_heads,
                        dims.key_value_heads,
                        dims.head_dimension,
                        span_capacity,
                        w.stream
                    ),
                    "causal GQA attention"
                );
            }
            batch_gemm_quantized(
                w,
                quantized.output_projection,
                layer.output_projection,
                w.attention.as<__nv_bfloat16>(),
                w.projection.as<__nv_bfloat16>(),
                dims.query_width,
                dims.hidden,
                n
            );
            check_cuda(
                qwen3_tts::launch_add_in_place(
                    w.hidden.as<__nv_bfloat16>(),
                    w.projection.as<__nv_bfloat16>(),
                    n * dims.hidden,
                    w.stream
                ),
                "attention residual"
            );
            check_cuda(
                qwen3_tts::launch_rms_norm_rows(
                    w.hidden.as<__nv_bfloat16>(),
                    layer.post_attention_norm,
                    w.normalized.as<__nv_bfloat16>(),
                    n,
                    dims.hidden,
                    kRmsEpsilon,
                    w.stream
                ),
                "post-attention RMSNorm"
            );
            batch_gemm_quantized(
                w,
                quantized.gate_projection,
                layer.gate_projection,
                w.normalized.as<__nv_bfloat16>(),
                w.gate.as<__nv_bfloat16>(),
                dims.hidden,
                dims.intermediate,
                n
            );
            batch_gemm_quantized(
                w,
                quantized.up_projection,
                layer.up_projection,
                w.normalized.as<__nv_bfloat16>(),
                w.up.as<__nv_bfloat16>(),
                dims.hidden,
                dims.intermediate,
                n
            );
            check_cuda(
                qwen3_tts::launch_silu_gate(
                    w.gate.as<__nv_bfloat16>(),
                    w.up.as<__nv_bfloat16>(),
                    n * dims.intermediate,
                    w.stream
                ),
                "SiLU gate"
            );
            batch_gemm_quantized(
                w,
                quantized.down_projection,
                layer.down_projection,
                w.gate.as<__nv_bfloat16>(),
                w.projection.as<__nv_bfloat16>(),
                dims.intermediate,
                dims.hidden,
                n
            );
            check_cuda(
                qwen3_tts::launch_add_in_place(
                    w.hidden.as<__nv_bfloat16>(),
                    w.projection.as<__nv_bfloat16>(),
                    n * dims.hidden,
                    w.stream
                ),
                "MLP residual"
            );
        }
    }

    void ensure_ready() const {
        if (!model_->finalized()) {
            throw std::runtime_error("weights have not been finalized");
        }
    }

    void trace_vector(
        const char* stage,
        int layer,
        const __nv_bfloat16* values,
        int width
    ) {
        if (!trace_active_) {
            return;
        }
        std::vector<__nv_bfloat16> encoded(width);
        check_cuda(
            cudaMemcpyAsync(
                encoded.data(),
                values,
                static_cast<size_t>(width) * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost,
                stream_
            ),
            "copy parity trace to host"
        );
        check_cuda(cudaStreamSynchronize(stream_), "synchronize parity trace");
        double sum = 0.0;
        double square_sum = 0.0;
        float maximum = 0.0f;
        for (const __nv_bfloat16 value : encoded) {
            const float decoded = __bfloat162float(value);
            sum += decoded;
            square_sum += static_cast<double>(decoded) * decoded;
            maximum = std::max(maximum, std::abs(decoded));
        }
        std::fprintf(
            stderr,
            "QWEN3_TTS_TRACE {\"stage\":\"%s\",\"layer\":%d,\"mean\":%.9g,\"rms\":%.9g,\"maximum\":%.9g,\"first\":[",
            stage,
            layer,
            sum / width,
            std::sqrt(square_sum / width),
            maximum
        );
        for (int index = 0; index < std::min(width, 16); ++index) {
            std::fprintf(
                stderr,
                "%s%.9g",
                index == 0 ? "" : ",",
                __bfloat162float(encoded[index])
            );
        }
        std::fprintf(stderr, "]}\n");
    }

    void trace_logits() {
        if (!trace_active_) {
            return;
        }
        std::vector<__nv_bfloat16> encoded(kTalkerVocabulary);
        check_cuda(
            cudaMemcpyAsync(
                encoded.data(),
                logits_.as<__nv_bfloat16>(),
                encoded.size() * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost,
                stream_
            ),
            "copy parity logits to host"
        );
        check_cuda(cudaStreamSynchronize(stream_), "synchronize parity logits");
        std::vector<int> indices(kTalkerVocabulary);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(),
            indices.begin() + 20,
            indices.end(),
            [&encoded](int left, int right) {
                const float lhs = __bfloat162float(encoded[left]);
                const float rhs = __bfloat162float(encoded[right]);
                return lhs == rhs ? left < right : lhs > rhs;
            }
        );
        std::fprintf(stderr, "QWEN3_TTS_TRACE {\"stage\":\"logits\",\"top\":[");
        for (int rank = 0; rank < 20; ++rank) {
            const int token = indices[rank];
            std::fprintf(
                stderr,
                "%s{\"token\":%d,\"logit\":%.9g}",
                rank == 0 ? "" : ",",
                token,
                __bfloat162float(encoded[token])
            );
        }
        std::fprintf(stderr, "]}\n");
    }

    void trace_float_logits(const __nv_bfloat16* input) {
        if (!trace_active_) {
            return;
        }
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        float* device_logits = attention_scores_.as<float>();
        check_cublas(
            cublasGemmEx(
                cublas_,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                kTalkerVocabulary,
                1,
                kTalkerHidden,
                &alpha,
                codec_head_,
                CUDA_R_16BF,
                kTalkerHidden,
                input,
                CUDA_R_16BF,
                kTalkerHidden,
                &beta,
                device_logits,
                CUDA_R_32F,
                kTalkerVocabulary,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "cublasGemmEx(float codec logits)"
        );
        std::vector<float> encoded(kTalkerVocabulary);
        check_cuda(
            cudaMemcpyAsync(
                encoded.data(),
                device_logits,
                encoded.size() * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream_
            ),
            "copy float parity logits to host"
        );
        check_cuda(cudaStreamSynchronize(stream_), "synchronize float parity logits");
        std::vector<int> indices(kTalkerVocabulary);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(),
            indices.begin() + 5,
            indices.end(),
            [&encoded](int left, int right) {
                return encoded[left] == encoded[right]
                    ? left < right
                    : encoded[left] > encoded[right];
            }
        );
        std::fprintf(stderr, "QWEN3_TTS_TRACE {\"stage\":\"float_logits\",\"top\":[");
        for (int rank = 0; rank < 5; ++rank) {
            const int token = indices[rank];
            std::fprintf(
                stderr,
                "%s{\"token\":%d,\"logit\":%.9g}",
                rank == 0 ? "" : ",",
                token,
                encoded[token]
            );
        }
        std::fprintf(stderr, "]}\n");
    }

    void dump_stage(
        const char* name,
        const __nv_bfloat16* values,
        int width
    ) {
        if (!stage_dump_enabled_) {
            return;
        }
        std::vector<__nv_bfloat16> encoded(width);
        check_cuda(
            cudaMemcpyAsync(
                encoded.data(),
                values,
                static_cast<size_t>(width) * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost,
                stream_
            ),
            "copy stage dump to host"
        );
        check_cuda(cudaStreamSynchronize(stream_), "synchronize stage dump");
        const std::string path = std::string("/tmp/native_stages/") + name + ".bf16";
        std::FILE* file = std::fopen(path.c_str(), "wb");
        if (file == nullptr) {
            throw std::runtime_error("failed to create stage dump " + path);
        }
        const size_t written = std::fwrite(
            encoded.data(),
            sizeof(__nv_bfloat16),
            encoded.size(),
            file
        );
        const int close_status = std::fclose(file);
        if (written != encoded.size() || close_status != 0) {
            throw std::runtime_error("failed to write stage dump " + path);
        }
    }

    void dump_attention_rows(const char* name, int rows, int heads) {
        if (!stage_dump_enabled_) {
            return;
        }
        std::vector<__nv_bfloat16> encoded(static_cast<size_t>(rows) * heads);
        for (int head = 0; head < heads; ++head) {
            const size_t source_offset = (static_cast<size_t>(head) * rows + rows - 1) * rows;
            check_cuda(
                cudaMemcpyAsync(
                    encoded.data() + static_cast<size_t>(head) * rows,
                    attention_scores_.as<__nv_bfloat16>() + source_offset,
                    static_cast<size_t>(rows) * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToHost,
                    stream_
                ),
                "copy attention probabilities to host"
            );
        }
        check_cuda(cudaStreamSynchronize(stream_), "synchronize attention probabilities");
        const std::string path = std::string("/tmp/native_stages/") + name + ".bf16";
        std::FILE* file = std::fopen(path.c_str(), "wb");
        if (file == nullptr) {
            throw std::runtime_error("failed to create attention row dump " + path);
        }
        const size_t written = std::fwrite(
            encoded.data(),
            sizeof(__nv_bfloat16),
            encoded.size(),
            file
        );
        const int close_status = std::fclose(file);
        if (written != encoded.size() || close_status != 0) {
            throw std::runtime_error("failed to write attention row dump " + path);
        }
    }

    const __nv_bfloat16* embedding_row(
        const __nv_bfloat16* table,
        int token,
        int vocabulary,
        int width
    ) const {
        if (token < 0 || token >= vocabulary) {
            throw std::runtime_error("embedding token is outside its vocabulary");
        }
        return table + static_cast<size_t>(token) * width;
    }

    void gemv(
        const __nv_bfloat16* weight,
        const __nv_bfloat16* input,
        __nv_bfloat16* output,
        int input_features,
        int output_features
    ) {
        gemm_rows(weight, input, output, input_features, output_features, 1);
    }

    void decode_gemv(
        const QuantizedTensor& quantized,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* input,
        __nv_bfloat16* output,
        int input_features,
        int output_features
    ) {
        decode_gemm_rows(quantized, weight, input, output, input_features, output_features, 1);
    }

    void decode_gemm_rows(
        const QuantizedTensor& quantized,
        const __nv_bfloat16* weight,
        const __nv_bfloat16* input,
        __nv_bfloat16* output,
        int input_features,
        int output_features,
        int rows
    ) {
        if (quantized.data != nullptr) {
            check_cuda(
                qwen3_tts::launch_int8_gemm_rows(
                    quantized.data,
                    quantized.scales,
                    input,
                    output,
                    input_features,
                    output_features,
                    rows,
                    stream_
                ),
                "int8 decode GEMM"
            );
            return;
        }
        gemm_rows(weight, input, output, input_features, output_features, rows);
    }

    void gemm_rows(
        const __nv_bfloat16* weight,
        const __nv_bfloat16* input,
        __nv_bfloat16* output,
        int input_features,
        int output_features,
        int rows
    ) {
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        // CUBLAS_COMPUTE_32F is deliberate: CUBLAS_COMPUTE_32F_FAST_16BF was
        // measured on GB10 (2026-07-18, B1/B6 client benchmarks) and regressed
        // decode TTFA and aggregate RTF by roughly 20 percent.
        check_cublas(
            cublasGemmEx(
                cublas_,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                output_features,
                rows,
                input_features,
                &alpha,
                weight,
                CUDA_R_16BF,
                input_features,
                input,
                CUDA_R_16BF,
                input_features,
                &beta,
                output,
                CUDA_R_16BF,
                output_features,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "cublasGemmEx"
        );
    }

    void project_text(int token, __nv_bfloat16* output) {
        const __nv_bfloat16* embedding = embedding_row(
            text_embedding_,
            token,
            kTextVocabulary,
            kTalkerHidden
        );
        decode_gemv(
            text_fc1_int8_,
            text_fc1_,
            embedding,
            projection_.as<__nv_bfloat16>(),
            kTalkerHidden,
            kTalkerHidden
        );
        check_cuda(
            qwen3_tts::launch_bias_activation(
                projection_.as<__nv_bfloat16>(),
                text_fc1_bias_,
                kTalkerHidden,
                true,
                stream_
            ),
            "text projection fc1 activation"
        );
        decode_gemv(
            text_fc2_int8_,
            text_fc2_,
            projection_.as<__nv_bfloat16>(),
            output,
            kTalkerHidden,
            kTalkerHidden
        );
        check_cuda(
            qwen3_tts::launch_bias_activation(
                output,
                text_fc2_bias_,
                kTalkerHidden,
                false,
                stream_
            ),
            "text projection fc2 bias"
        );
    }

    void prepare_prompt_embedding(
        int text_token,
        int codec_token,
        __nv_bfloat16* output
    ) {
        if (text_token < 0 && codec_token < 0) {
            throw std::runtime_error("a prompt position must contain text or codec input");
        }
        if (text_token >= 0) {
            project_text(text_token, output);
        } else {
            check_cuda(
                qwen3_tts::launch_fill_zero(output, kTalkerHidden, stream_),
                "zero prompt embedding"
            );
        }
        if (codec_token >= 0) {
            check_cuda(
                qwen3_tts::launch_add_in_place(
                    output,
                    embedding_row(codec_embedding_, codec_token, kTalkerVocabulary, kTalkerHidden),
                    kTalkerHidden,
                    stream_
                ),
                "add prompt codec embedding"
            );
        }
    }

    void run_prefill_gqa_gemm(
        const ModelDimensions& dimensions,
        int rows,
        bool dump_probabilities
    ) {
        check_cuda(
            qwen3_tts::launch_pack_gqa_heads(
                query_.as<__nv_bfloat16>(),
                key_.as<__nv_bfloat16>(),
                value_.as<__nv_bfloat16>(),
                packed_query_.as<__nv_bfloat16>(),
                packed_key_.as<__nv_bfloat16>(),
                packed_value_.as<__nv_bfloat16>(),
                rows,
                dimensions.query_heads,
                dimensions.key_value_heads,
                dimensions.head_dimension,
                stream_
            ),
            "pack prefill GQA heads"
        );

        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        const long long head_stride = static_cast<long long>(rows)
            * dimensions.head_dimension;
        const long long score_stride = static_cast<long long>(rows) * rows;
        check_cublas(
            cublasGemmStridedBatchedEx(
                cublas_,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                rows,
                rows,
                dimensions.head_dimension,
                &alpha,
                packed_key_.as<__nv_bfloat16>(),
                CUDA_R_16BF,
                dimensions.head_dimension,
                head_stride,
                packed_query_.as<__nv_bfloat16>(),
                CUDA_R_16BF,
                dimensions.head_dimension,
                head_stride,
                &beta,
                attention_scores_.as<__nv_bfloat16>(),
                CUDA_R_16BF,
                rows,
                score_stride,
                dimensions.query_heads,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "cublasGemmStridedBatchedEx(prefill QK)"
        );
        if (dump_probabilities) {
            dump_attention_rows("attention_dot_products", rows, dimensions.query_heads);
        }
        check_cuda(
            qwen3_tts::launch_causal_softmax(
                attention_scores_.as<__nv_bfloat16>(),
                rows,
                dimensions.query_heads,
                dimensions.head_dimension,
                stream_
            ),
            "prefill causal softmax"
        );
        if (dump_probabilities) {
            dump_attention_rows("attention_probabilities", rows, dimensions.query_heads);
        }
        check_cublas(
            cublasGemmStridedBatchedEx(
                cublas_,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                dimensions.head_dimension,
                rows,
                rows,
                &alpha,
                packed_value_.as<__nv_bfloat16>(),
                CUDA_R_16BF,
                dimensions.head_dimension,
                head_stride,
                attention_scores_.as<__nv_bfloat16>(),
                CUDA_R_16BF,
                rows,
                score_stride,
                &beta,
                packed_attention_.as<__nv_bfloat16>(),
                CUDA_R_16BF,
                dimensions.head_dimension,
                head_stride,
                dimensions.query_heads,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ),
            "cublasGemmStridedBatchedEx(prefill PV)"
        );
        check_cuda(
            qwen3_tts::launch_unpack_heads(
                packed_attention_.as<__nv_bfloat16>(),
                attention_.as<__nv_bfloat16>(),
                rows,
                dimensions.query_heads,
                dimensions.head_dimension,
                stream_
            ),
            "unpack prefill attention heads"
        );
    }

    void run_decoder_prefill(
        const std::vector<DecoderWeights>& layers,
        std::vector<LayerCache>& caches,
        const ModelDimensions& dimensions,
        int rows
    ) {
        const int hidden_elements = rows * dimensions.hidden;
        const int intermediate_elements = rows * dimensions.intermediate;
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const DecoderWeights& layer = layers[layer_index];
            check_cuda(
                qwen3_tts::launch_rms_norm_rows(
                    hidden_.as<__nv_bfloat16>(),
                    layer.input_norm,
                    normalized_.as<__nv_bfloat16>(),
                    rows,
                    dimensions.hidden,
                    kRmsEpsilon,
                    stream_
                ),
                "prefill input RMSNorm"
            );
            if (layer_index == 0) {
                dump_stage(
                    "input_norm",
                    normalized_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.hidden,
                    dimensions.hidden
                );
            }
            gemm_rows(
                layer.q_projection,
                normalized_.as<__nv_bfloat16>(),
                query_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.query_width,
                rows
            );
            gemm_rows(
                layer.k_projection,
                normalized_.as<__nv_bfloat16>(),
                key_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.key_value_width,
                rows
            );
            gemm_rows(
                layer.v_projection,
                normalized_.as<__nv_bfloat16>(),
                value_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.key_value_width,
                rows
            );
            if (layer_index == 0) {
                dump_stage(
                    "q_projection",
                    query_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.query_width,
                    dimensions.query_width
                );
                dump_stage(
                    "k_projection",
                    key_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.key_value_width,
                    dimensions.key_value_width
                );
                dump_stage(
                    "v_projection",
                    value_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.key_value_width,
                    dimensions.key_value_width
                );
            }
            check_cuda(
                qwen3_tts::launch_head_rms_norm_rows(
                    query_.as<__nv_bfloat16>(),
                    layer.q_norm,
                    rows,
                    dimensions.query_heads,
                    dimensions.head_dimension,
                    kRmsEpsilon,
                    stream_
                ),
                "prefill query head RMSNorm"
            );
            check_cuda(
                qwen3_tts::launch_head_rms_norm_rows(
                    key_.as<__nv_bfloat16>(),
                    layer.k_norm,
                    rows,
                    dimensions.key_value_heads,
                    dimensions.head_dimension,
                    kRmsEpsilon,
                    stream_
                ),
                "prefill key head RMSNorm"
            );
            if (layer_index == 0) {
                dump_stage(
                    "q_norm",
                    query_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.query_width,
                    dimensions.query_width
                );
                dump_stage(
                    "k_norm",
                    key_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.key_value_width,
                    dimensions.key_value_width
                );
            }
            check_cuda(
                qwen3_tts::launch_rope_rows(
                    query_.as<__nv_bfloat16>(),
                    rows,
                    dimensions.query_heads,
                    dimensions.head_dimension,
                    0,
                    kRopeTheta,
                    stream_
                ),
                "prefill query RoPE"
            );
            check_cuda(
                qwen3_tts::launch_rope_rows(
                    key_.as<__nv_bfloat16>(),
                    rows,
                    dimensions.key_value_heads,
                    dimensions.head_dimension,
                    0,
                    kRopeTheta,
                    stream_
                ),
                "prefill key RoPE"
            );
            if (layer_index == 0) {
                dump_stage(
                    "q_rope",
                    query_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.query_width,
                    dimensions.query_width
                );
                dump_stage(
                    "k_rope",
                    key_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.key_value_width,
                    dimensions.key_value_width
                );
            }

            const size_t cache_bytes = static_cast<size_t>(rows)
                * dimensions.key_value_width * sizeof(__nv_bfloat16);
            check_cuda(
                cudaMemcpyAsync(
                    caches[layer_index].key.as<__nv_bfloat16>(),
                    key_.as<__nv_bfloat16>(),
                    cache_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "write prefill key cache"
            );
            check_cuda(
                cudaMemcpyAsync(
                    caches[layer_index].value.as<__nv_bfloat16>(),
                    value_.as<__nv_bfloat16>(),
                    cache_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "write prefill value cache"
            );
            if (rows <= prefill_gemm_capacity_) {
                run_prefill_gqa_gemm(dimensions, rows, layer_index == 0);
            } else {
                check_cuda(
                    qwen3_tts::launch_prefill_causal_gqa_attention(
                        query_.as<__nv_bfloat16>(),
                        caches[layer_index].key.as<__nv_bfloat16>(),
                        caches[layer_index].value.as<__nv_bfloat16>(),
                        attention_.as<__nv_bfloat16>(),
                        rows,
                        dimensions.query_heads,
                        dimensions.key_value_heads,
                        dimensions.head_dimension,
                        stream_
                    ),
                    "prefill causal GQA attention fallback"
                );
            }
            if (layer_index == 0) {
                dump_stage(
                    "attention",
                    attention_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.query_width,
                    dimensions.query_width
                );
            }
            gemm_rows(
                layer.output_projection,
                attention_.as<__nv_bfloat16>(),
                projection_.as<__nv_bfloat16>(),
                dimensions.query_width,
                dimensions.hidden,
                rows
            );
            if (layer_index == 0) {
                dump_stage(
                    "o_projection",
                    projection_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.hidden,
                    dimensions.hidden
                );
            }
            check_cuda(
                qwen3_tts::launch_add_in_place(
                    hidden_.as<__nv_bfloat16>(),
                    projection_.as<__nv_bfloat16>(),
                    hidden_elements,
                    stream_
                ),
                "prefill attention residual"
            );
            if (layer_index == 0) {
                dump_stage(
                    "attention_residual",
                    hidden_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.hidden,
                    dimensions.hidden
                );
            }
            check_cuda(
                qwen3_tts::launch_rms_norm_rows(
                    hidden_.as<__nv_bfloat16>(),
                    layer.post_attention_norm,
                    normalized_.as<__nv_bfloat16>(),
                    rows,
                    dimensions.hidden,
                    kRmsEpsilon,
                    stream_
                ),
                "prefill post-attention RMSNorm"
            );
            if (layer_index == 0) {
                dump_stage(
                    "post_attention_norm",
                    normalized_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.hidden,
                    dimensions.hidden
                );
            }
            gemm_rows(
                layer.gate_projection,
                normalized_.as<__nv_bfloat16>(),
                gate_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.intermediate,
                rows
            );
            gemm_rows(
                layer.up_projection,
                normalized_.as<__nv_bfloat16>(),
                up_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.intermediate,
                rows
            );
            if (layer_index == 0) {
                dump_stage(
                    "gate_projection",
                    gate_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.intermediate,
                    dimensions.intermediate
                );
                dump_stage(
                    "up_projection",
                    up_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.intermediate,
                    dimensions.intermediate
                );
            }
            check_cuda(
                qwen3_tts::launch_silu_gate(
                    gate_.as<__nv_bfloat16>(),
                    up_.as<__nv_bfloat16>(),
                    intermediate_elements,
                    stream_
                ),
                "prefill SiLU gate"
            );
            if (layer_index == 0) {
                dump_stage(
                    "gated_activation",
                    gate_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.intermediate,
                    dimensions.intermediate
                );
            }
            gemm_rows(
                layer.down_projection,
                gate_.as<__nv_bfloat16>(),
                projection_.as<__nv_bfloat16>(),
                dimensions.intermediate,
                dimensions.hidden,
                rows
            );
            if (layer_index == 0) {
                dump_stage(
                    "down_projection",
                    projection_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.hidden,
                    dimensions.hidden
                );
            }
            check_cuda(
                qwen3_tts::launch_add_in_place(
                    hidden_.as<__nv_bfloat16>(),
                    projection_.as<__nv_bfloat16>(),
                    hidden_elements,
                    stream_
                ),
                "prefill MLP residual"
            );
            if (layer_index == 0) {
                dump_stage(
                    "layer_output",
                    hidden_.as<__nv_bfloat16>()
                        + static_cast<size_t>(rows - 1) * dimensions.hidden,
                    dimensions.hidden
                );
            }
            trace_vector(
                "layer",
                static_cast<int>(layer_index),
                hidden_.as<__nv_bfloat16>()
                    + static_cast<size_t>(rows - 1) * dimensions.hidden,
                dimensions.hidden
            );
        }
    }

    float run_talker_prefill(int rows) {
        check_cuda(cudaEventRecord(start_, stream_), "cudaEventRecord(prefill start)");
        run_decoder_prefill(talker_layers_, talker_cache_, kTalkerDimensions, rows);
        const __nv_bfloat16* final_row = hidden_.as<__nv_bfloat16>()
            + static_cast<size_t>(rows - 1) * kTalkerHidden;
        check_cuda(
            qwen3_tts::launch_rms_norm(
                final_row,
                talker_norm_,
                normalized_.as<__nv_bfloat16>(),
                kTalkerHidden,
                kRmsEpsilon,
                stream_
            ),
            "prefill talker final RMSNorm"
        );
        check_cuda(
            cudaMemcpyAsync(
                last_hidden_.as<__nv_bfloat16>(),
                normalized_.as<__nv_bfloat16>(),
                kTalkerHidden * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice,
                stream_
            ),
            "copy prefill talker hidden"
        );
        trace_vector("final_norm", kTalkerLayers, normalized_.as<__nv_bfloat16>(), kTalkerHidden);
        gemv(
            codec_head_,
            normalized_.as<__nv_bfloat16>(),
            logits_.as<__nv_bfloat16>(),
            kTalkerHidden,
            kTalkerVocabulary
        );
        dump_stage("final_logits", logits_.as<__nv_bfloat16>(), kTalkerVocabulary);
        trace_logits();
        trace_float_logits(normalized_.as<__nv_bfloat16>());
        check_cuda(cudaEventRecord(stop_, stream_), "cudaEventRecord(prefill stop)");
        check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize(prefill)");
        float milliseconds = 0.0f;
        check_cuda(
            cudaEventElapsedTime(&milliseconds, start_, stop_),
            "cudaEventElapsedTime(prefill)"
        );
        return milliseconds;
    }

    void run_decoder(
        const std::vector<DecoderWeights>& layers,
        const std::vector<DecoderWeightsInt8>& int8_layers,
        std::vector<LayerCache>& caches,
        const ModelDimensions& dimensions,
        int position
    ) {
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const DecoderWeights& layer = layers[layer_index];
            const DecoderWeightsInt8 quantized = int8_layers.empty()
                ? DecoderWeightsInt8{}
                : int8_layers[layer_index];
            check_cuda(
                qwen3_tts::launch_rms_norm(
                    hidden_.as<__nv_bfloat16>(),
                    layer.input_norm,
                    normalized_.as<__nv_bfloat16>(),
                    dimensions.hidden,
                    kRmsEpsilon,
                    stream_
                ),
                "input RMSNorm"
            );
            decode_gemv(
                quantized.q_projection,
                layer.q_projection,
                normalized_.as<__nv_bfloat16>(),
                query_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.query_width
            );
            decode_gemv(
                quantized.k_projection,
                layer.k_projection,
                normalized_.as<__nv_bfloat16>(),
                key_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.key_value_width
            );
            decode_gemv(
                quantized.v_projection,
                layer.v_projection,
                normalized_.as<__nv_bfloat16>(),
                value_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.key_value_width
            );
            check_cuda(
                qwen3_tts::launch_head_rms_norm(
                    query_.as<__nv_bfloat16>(),
                    layer.q_norm,
                    dimensions.query_heads,
                    dimensions.head_dimension,
                    kRmsEpsilon,
                    stream_
                ),
                "query head RMSNorm"
            );
            check_cuda(
                qwen3_tts::launch_head_rms_norm(
                    key_.as<__nv_bfloat16>(),
                    layer.k_norm,
                    dimensions.key_value_heads,
                    dimensions.head_dimension,
                    kRmsEpsilon,
                    stream_
                ),
                "key head RMSNorm"
            );
            check_cuda(
                qwen3_tts::launch_rope(
                    query_.as<__nv_bfloat16>(),
                    dimensions.query_heads,
                    dimensions.head_dimension,
                    position,
                    kRopeTheta,
                    stream_
                ),
                "query RoPE"
            );
            check_cuda(
                qwen3_tts::launch_rope(
                    key_.as<__nv_bfloat16>(),
                    dimensions.key_value_heads,
                    dimensions.head_dimension,
                    position,
                    kRopeTheta,
                    stream_
                ),
                "key RoPE"
            );

            const size_t cache_offset = static_cast<size_t>(position)
                * dimensions.key_value_width;
            check_cuda(
                cudaMemcpyAsync(
                    caches[layer_index].key.as<__nv_bfloat16>() + cache_offset,
                    key_.as<__nv_bfloat16>(),
                    static_cast<size_t>(dimensions.key_value_width) * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "append key cache"
            );
            check_cuda(
                cudaMemcpyAsync(
                    caches[layer_index].value.as<__nv_bfloat16>() + cache_offset,
                    value_.as<__nv_bfloat16>(),
                    static_cast<size_t>(dimensions.key_value_width) * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "append value cache"
            );
            check_cuda(
                qwen3_tts::launch_causal_gqa_attention(
                    query_.as<__nv_bfloat16>(),
                    caches[layer_index].key.as<__nv_bfloat16>(),
                    caches[layer_index].value.as<__nv_bfloat16>(),
                    attention_.as<__nv_bfloat16>(),
                    dimensions.query_heads,
                    dimensions.key_value_heads,
                    dimensions.head_dimension,
                    position + 1,
                    stream_
                ),
                "causal GQA attention"
            );
            decode_gemv(
                quantized.output_projection,
                layer.output_projection,
                attention_.as<__nv_bfloat16>(),
                projection_.as<__nv_bfloat16>(),
                dimensions.query_width,
                dimensions.hidden
            );
            check_cuda(
                qwen3_tts::launch_add_in_place(
                    hidden_.as<__nv_bfloat16>(),
                    projection_.as<__nv_bfloat16>(),
                    dimensions.hidden,
                    stream_
                ),
                "attention residual"
            );
            check_cuda(
                qwen3_tts::launch_rms_norm(
                    hidden_.as<__nv_bfloat16>(),
                    layer.post_attention_norm,
                    normalized_.as<__nv_bfloat16>(),
                    dimensions.hidden,
                    kRmsEpsilon,
                    stream_
                ),
                "post-attention RMSNorm"
            );
            decode_gemv(
                quantized.gate_projection,
                layer.gate_projection,
                normalized_.as<__nv_bfloat16>(),
                gate_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.intermediate
            );
            decode_gemv(
                quantized.up_projection,
                layer.up_projection,
                normalized_.as<__nv_bfloat16>(),
                up_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.intermediate
            );
            check_cuda(
                qwen3_tts::launch_silu_gate(
                    gate_.as<__nv_bfloat16>(),
                    up_.as<__nv_bfloat16>(),
                    dimensions.intermediate,
                    stream_
                ),
                "SiLU gate"
            );
            decode_gemv(
                quantized.down_projection,
                layer.down_projection,
                gate_.as<__nv_bfloat16>(),
                projection_.as<__nv_bfloat16>(),
                dimensions.intermediate,
                dimensions.hidden
            );
            check_cuda(
                qwen3_tts::launch_add_in_place(
                    hidden_.as<__nv_bfloat16>(),
                    projection_.as<__nv_bfloat16>(),
                    dimensions.hidden,
                    stream_
                ),
                "MLP residual"
            );
            trace_vector(
                "layer",
                static_cast<int>(layer_index),
                hidden_.as<__nv_bfloat16>(),
                dimensions.hidden
            );
        }
    }

    void run_talker_step(int position, bool emit_logits) {
        run_decoder(
            talker_layers_, talker_layers_int8_, talker_cache_, kTalkerDimensions, position
        );
        if (emit_logits) {
            check_cuda(
                qwen3_tts::launch_rms_norm(
                    hidden_.as<__nv_bfloat16>(),
                    talker_norm_,
                    normalized_.as<__nv_bfloat16>(),
                    kTalkerHidden,
                    kRmsEpsilon,
                    stream_
                ),
                "talker final RMSNorm"
            );
            check_cuda(
                cudaMemcpyAsync(
                    last_hidden_.as<__nv_bfloat16>(),
                    normalized_.as<__nv_bfloat16>(),
                    kTalkerHidden * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice,
                    stream_
                ),
                "copy talker hidden"
            );
            trace_vector("final_norm", kTalkerLayers, normalized_.as<__nv_bfloat16>(), kTalkerHidden);
            decode_gemv(
                codec_head_int8_,
                codec_head_,
                normalized_.as<__nv_bfloat16>(),
                logits_.as<__nv_bfloat16>(),
                kTalkerHidden,
                kTalkerVocabulary
            );
            trace_logits();
        }
    }

    void run_predictor_position(const __nv_bfloat16* input, int position, int head) {
        decode_gemv(
            small_to_predictor_int8_,
            small_to_predictor_,
            input,
            hidden_.as<__nv_bfloat16>(),
            kTalkerHidden,
            kPredictorHidden
        );
        check_cuda(
            qwen3_tts::launch_bias_activation(
                hidden_.as<__nv_bfloat16>(),
                small_to_predictor_bias_,
                kPredictorHidden,
                false,
                stream_
            ),
            "predictor input projection bias"
        );
        run_decoder(
            predictor_layers_,
            predictor_layers_int8_,
            predictor_cache_,
            kPredictorDimensions,
            position
        );
        if (head >= 0) {
            check_cuda(
                qwen3_tts::launch_rms_norm(
                    hidden_.as<__nv_bfloat16>(),
                    predictor_norm_,
                    normalized_.as<__nv_bfloat16>(),
                    kPredictorHidden,
                    kRmsEpsilon,
                    stream_
                ),
                "predictor final RMSNorm"
            );
            decode_gemv(
                predictor_heads_int8_[head],
                predictor_heads_[head],
                normalized_.as<__nv_bfloat16>(),
                logits_.as<__nv_bfloat16>(),
                kPredictorHidden,
                kPredictorVocabulary
            );
        }
    }

    void prepare_generated_embedding(int text_token) {
        check_cuda(
            qwen3_tts::launch_gather_embedding(
                codec_embedding_,
                kTalkerVocabulary,
                kTalkerHidden,
                frame_tokens_.as<int>(),
                hidden_.as<__nv_bfloat16>(),
                stream_
            ),
            "gather semantic talker embedding"
        );
        for (int residual = 0; residual < kResidualCodebooks; ++residual) {
            check_cuda(
                qwen3_tts::launch_add_embedding(
                    hidden_.as<__nv_bfloat16>(),
                    predictor_embeddings_[residual],
                    kPredictorVocabulary,
                    kTalkerHidden,
                    frame_tokens_.as<int>() + residual + 1,
                    stream_
                ),
                "add residual codec embedding"
            );
        }
        project_text(text_token, text_output_.as<__nv_bfloat16>());
        check_cuda(
            qwen3_tts::launch_add_in_place(
                hidden_.as<__nv_bfloat16>(),
                text_output_.as<__nv_bfloat16>(),
                kTalkerHidden,
                stream_
            ),
            "add trailing text embedding"
        );
    }

    void sample_logits_device(
        int vocabulary,
        const Qwen3TtsSamplingConfig& config,
        bool talker
    ) {
        if (config.top_k < 0 || config.top_p <= 0.0f || config.top_p > 1.0f
            || config.temperature <= 0.0f || config.repetition_penalty <= 0.0f) {
            throw std::runtime_error("invalid sampling configuration");
        }
        if (talker && generated_semantic_count_ >= max_sequence_length_) {
            throw std::runtime_error("semantic history exceeds configured sequence capacity");
        }
        check_cuda(
            qwen3_tts::launch_sample_logits(
                logits_.as<__nv_bfloat16>(),
                vocabulary,
                talker,
                kCodecEos,
                semantic_history_.as<int>(),
                generated_semantic_count_,
                config.do_sample,
                config.top_k,
                config.top_p,
                config.temperature,
                config.repetition_penalty,
                random_state_.as<uint64_t>(),
                sampled_token_.as<int>(),
                stream_
            ),
            "sample logits on device"
        );
        ++device_sample_count_;
        if (talker) {
            check_cuda(
                qwen3_tts::launch_store_sampled_token(
                    semantic_history_.as<int>(),
                    generated_semantic_count_,
                    sampled_token_.as<int>(),
                    stream_
                ),
                "append sampled semantic token"
            );
            ++generated_semantic_count_;
        }
    }

    void enqueue_sampled_token_to_host() {
        check_cuda(
            cudaMemcpyAsync(
                host_sampled_token_,
                sampled_token_.as<int>(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream_
            ),
            "copy sampled token to pinned host memory"
        );
    }

    int validated_host_sampled_token() const {
        if (*host_sampled_token_ < 0 || *host_sampled_token_ >= kTalkerVocabulary) {
            throw std::runtime_error("device sampler returned an invalid token");
        }
        return *host_sampled_token_;
    }

    int copy_sampled_token_to_host() {
        enqueue_sampled_token_to_host();
        ++host_sync_count_;
        check_cuda(cudaStreamSynchronize(stream_), "synchronize sampled token");
        return validated_host_sampled_token();
    }

public:
    Qwen3TtsTalkerStateInfo state_info() const {
        Qwen3TtsTalkerStateInfo result{};
        result.abi_version = QWEN3_TTS_TALKER_ABI_VERSION;
        result.phase = static_cast<uint32_t>(phase_);
        result.talker_position = static_cast<uint32_t>(position_);
        result.semantic_history_count = static_cast<uint32_t>(generated_semantic_count_);
        result.frames_generated = frames_generated_;
        result.device_sample_count = device_sample_count_;
        result.host_sync_count = host_sync_count_;
        return result;
    }

    Qwen3TtsSessionMemory session_memory() const {
        uint64_t talker_kv = 0;
        for (const LayerCache& cache : talker_cache_) {
            talker_kv += cache.key.bytes() + cache.value.bytes();
        }
        uint64_t predictor_kv = 0;
        for (const LayerCache& cache : predictor_cache_) {
            predictor_kv += cache.key.bytes() + cache.value.bytes();
        }
        const uint64_t workspace = hidden_.bytes() + normalized_.bytes() + query_.bytes()
            + key_.bytes() + value_.bytes() + attention_.bytes() + projection_.bytes()
            + gate_.bytes() + up_.bytes() + logits_.bytes() + text_output_.bytes()
            + last_hidden_.bytes() + packed_query_.bytes() + packed_key_.bytes()
            + packed_value_.bytes() + packed_attention_.bytes() + attention_scores_.bytes()
            + sampled_token_.bytes() + frame_tokens_.bytes() + frame_codes_.bytes()
            + semantic_history_.bytes() + random_state_.bytes();
        Qwen3TtsSessionMemory result{};
        result.talker_kv_bytes = talker_kv;
        result.predictor_kv_bytes = predictor_kv;
        result.workspace_bytes = workspace;
        result.max_sequence_length = static_cast<uint32_t>(max_sequence_length_);
        return result;
    }

private:
    std::shared_ptr<TalkerModel> model_;
    int device_index_;
    int max_sequence_length_;
    int prefill_gemm_capacity_;
    bool trace_enabled_;
    bool stage_dump_enabled_;
    int position_ = 0;
    bool trace_active_ = false;
    int generated_semantic_count_ = 0;
    uint16_t current_semantic_token_ = 0;
    Qwen3TtsTalkerPhase phase_ = QWEN3_TTS_TALKER_CREATED;
    uint64_t frames_generated_ = 0;
    uint64_t device_sample_count_ = 0;
    uint64_t host_sync_count_ = 0;
    int* host_sampled_token_ = nullptr;
    uint16_t* host_frame_codes_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    cudaEvent_t predictor_start_ = nullptr;
    cudaEvent_t predictor_stop_ = nullptr;
    cudaEvent_t frame_codes_ready_ = nullptr;
    cudaEvent_t semantic_ready_ = nullptr;
    bool frame_in_flight_ = false;
    bool pending_host_code_copy_ = false;
    bool poisoned_ = false;
    uint64_t next_lease_id_ = 1;
    uint64_t pending_lease_id_ = 0;
    uint32_t pending_talker_position_ = 0;
    std::vector<DecoderWeights> talker_layers_;
    std::vector<DecoderWeights> predictor_layers_;
    std::vector<DecoderWeightsInt8> talker_layers_int8_;
    std::vector<DecoderWeightsInt8> predictor_layers_int8_;
    std::array<QuantizedTensor, kResidualCodebooks> predictor_heads_int8_{};
    QuantizedTensor codec_head_int8_;
    QuantizedTensor small_to_predictor_int8_;
    QuantizedTensor text_fc1_int8_;
    QuantizedTensor text_fc2_int8_;
    std::vector<LayerCache> talker_cache_;
    std::vector<LayerCache> predictor_cache_;

    const __nv_bfloat16* codec_embedding_ = nullptr;
    const __nv_bfloat16* text_embedding_ = nullptr;
    const __nv_bfloat16* text_fc1_ = nullptr;
    const __nv_bfloat16* text_fc1_bias_ = nullptr;
    const __nv_bfloat16* text_fc2_ = nullptr;
    const __nv_bfloat16* text_fc2_bias_ = nullptr;
    const __nv_bfloat16* talker_norm_ = nullptr;
    const __nv_bfloat16* codec_head_ = nullptr;
    const __nv_bfloat16* small_to_predictor_ = nullptr;
    const __nv_bfloat16* small_to_predictor_bias_ = nullptr;
    const __nv_bfloat16* predictor_norm_ = nullptr;
    std::array<const __nv_bfloat16*, kResidualCodebooks> predictor_embeddings_{};
    std::array<const __nv_bfloat16*, kResidualCodebooks> predictor_heads_{};

    DeviceBuffer hidden_;
    DeviceBuffer normalized_;
    DeviceBuffer query_;
    DeviceBuffer key_;
    DeviceBuffer value_;
    DeviceBuffer attention_;
    DeviceBuffer projection_;
    DeviceBuffer gate_;
    DeviceBuffer up_;
    DeviceBuffer logits_;
    DeviceBuffer text_output_;
    DeviceBuffer last_hidden_;
    DeviceBuffer packed_query_;
    DeviceBuffer packed_key_;
    DeviceBuffer packed_value_;
    DeviceBuffer packed_attention_;
    DeviceBuffer attention_scores_;
    DeviceBuffer sampled_token_;
    DeviceBuffer frame_tokens_;
    DeviceBuffer frame_codes_;
    DeviceBuffer semantic_history_;
    DeviceBuffer random_state_;
};

using TalkerModelBox = std::shared_ptr<TalkerModel>;

TalkerModelBox& model(Qwen3TtsModelHandle handle) {
    if (handle == nullptr) {
        throw std::runtime_error("model handle is null");
    }
    return *static_cast<TalkerModelBox*>(handle);
}

TalkerContext* session(Qwen3TtsSessionHandle handle) {
    if (handle == nullptr) {
        throw std::runtime_error("session handle is null");
    }
    return static_cast<TalkerContext*>(handle);
}

template <typename Operation>
int32_t protect(
    Operation&& operation,
    int32_t failure_status,
    char* error,
    size_t error_capacity
) {
    try {
        operation();
        write_error(error, error_capacity, "");
        return QWEN3_TTS_TALKER_STATUS_OK;
    } catch (const CudaFailure& exception) {
        write_error(error, error_capacity, exception.what());
        return QWEN3_TTS_TALKER_STATUS_CUDA;
    } catch (const std::bad_alloc& exception) {
        write_error(error, error_capacity, exception.what());
        return QWEN3_TTS_TALKER_STATUS_ALLOCATION;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return failure_status;
    }
}

}  // namespace

extern "C" QWEN3_TTS_API uint32_t qwen3_tts_talker_abi_version(void) {
    return QWEN3_TTS_TALKER_ABI_VERSION;
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_model_create(
    int32_t device_index,
    Qwen3TtsModelHandle* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "model output handle pointer is null");
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        *output = new TalkerModelBox(std::make_shared<TalkerModel>(device_index));
    }, QWEN3_TTS_TALKER_STATUS_MODEL, error, error_capacity);
}

extern "C" QWEN3_TTS_API void qwen3_tts_model_destroy(Qwen3TtsModelHandle handle) {
    delete static_cast<TalkerModelBox*>(handle);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_model_upload_tensor(
    Qwen3TtsModelHandle handle,
    const char* name,
    const void* bf16_data,
    uint64_t byte_size,
    int32_t rank,
    const uint64_t* shape,
    char* error,
    size_t error_capacity
) {
    return protect([&] {
        model(handle)->upload_tensor(name, bf16_data, byte_size, rank, shape);
    }, QWEN3_TTS_TALKER_STATUS_MODEL, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_model_finalize(
    Qwen3TtsModelHandle handle,
    Qwen3TtsModelMemory* memory,
    char* error,
    size_t error_capacity
) {
    if (memory == nullptr) {
        write_error(error, error_capacity, "talker memory output pointer is null");
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        *memory = model(handle)->finalize();
    }, QWEN3_TTS_TALKER_STATUS_MODEL, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_session_create(
    Qwen3TtsModelHandle model_handle,
    int32_t max_sequence_length,
    uint64_t random_seed,
    Qwen3TtsSessionHandle* output,
    Qwen3TtsSessionMemory* memory,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr || memory == nullptr) {
        write_error(error, error_capacity, "session output or memory pointer is null");
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        auto created = std::make_unique<TalkerContext>(
            model(model_handle),
            max_sequence_length,
            random_seed
        );
        *memory = created->session_memory();
        *output = created.release();
    }, QWEN3_TTS_TALKER_STATUS_STATE, error, error_capacity);
}

extern "C" QWEN3_TTS_API void qwen3_tts_session_destroy(Qwen3TtsSessionHandle handle) {
    delete static_cast<TalkerContext*>(handle);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_session_reset(
    Qwen3TtsSessionHandle handle,
    uint64_t random_seed,
    char* error,
    size_t error_capacity
) {
    return protect([&] {
        session(handle)->reset(random_seed);
    }, QWEN3_TTS_TALKER_STATUS_STATE, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_session_prefill(
    Qwen3TtsSessionHandle handle,
    const int32_t* text_token_ids,
    const int32_t* codec_token_ids,
    int32_t token_count,
    Qwen3TtsSamplingConfig sampling,
    Qwen3TtsTalkerPrefillResult* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "prefill output pointer is null");
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        *output = session(handle)->prefill(
            text_token_ids,
            codec_token_ids,
            token_count,
            sampling
        );
    }, QWEN3_TTS_TALKER_STATUS_STATE, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_session_next_frame(
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
) {
    if (output_codes == nullptr || output_code_capacity < kPredictorSequence
        || next_semantic_token == nullptr || frame_info == nullptr) {
        write_error(error, error_capacity, "codec-frame outputs are null or undersized");
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        const Qwen3TtsCodecFrameResult frame = session(handle)->next_frame(
            semantic_token,
            trailing_text_token_id,
            talker_sampling,
            predictor_sampling
        );
        std::copy_n(frame.codes, kPredictorSequence, output_codes);
        *next_semantic_token = frame.next_semantic_token;
        frame_info->talker_position = frame.talker_position;
        frame_info->ended_by_eos = frame.ended_by_eos;
        frame_info->predictor_gpu_milliseconds = frame.predictor_gpu_milliseconds;
        frame_info->talker_gpu_milliseconds = frame.talker_gpu_milliseconds;
    }, QWEN3_TTS_TALKER_STATUS_STATE, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_session_next_frame_begin_v2(
    Qwen3TtsSessionHandle handle,
    int32_t trailing_text_token_id,
    Qwen3TtsSamplingConfig talker_sampling,
    Qwen3TtsSamplingConfig predictor_sampling,
    Qwen3TtsDeviceFrameViewV2* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "device-frame output is null");
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    if (output->struct_size != sizeof(Qwen3TtsDeviceFrameViewV2)
        || output->reserved != 0) {
        write_error(
            error,
            error_capacity,
            "device-frame output has an unsupported size or nonzero reserved field"
        );
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        TalkerContext* context = session(handle);
        *output = context->begin_frame(
            context->current_semantic_token(),
            trailing_text_token_id,
            talker_sampling,
            predictor_sampling,
            false
        );
    }, QWEN3_TTS_TALKER_STATUS_STATE, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_session_next_frame_finish_v2(
    Qwen3TtsSessionHandle handle,
    uint64_t lease_id,
    Qwen3TtsCudaEventHandle consumer_done_event,
    uint16_t* next_semantic_token,
    Qwen3TtsCodecFrameInfo* frame_info,
    char* error,
    size_t error_capacity
) {
    if (next_semantic_token == nullptr || frame_info == nullptr) {
        write_error(error, error_capacity, "device-frame finish outputs are null");
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        const Qwen3TtsCodecFrameResult frame = session(handle)->finish_frame(
            lease_id,
            consumer_done_event
        );
        *next_semantic_token = frame.next_semantic_token;
        frame_info->talker_position = frame.talker_position;
        frame_info->ended_by_eos = frame.ended_by_eos;
        frame_info->predictor_gpu_milliseconds = frame.predictor_gpu_milliseconds;
        frame_info->talker_gpu_milliseconds = frame.talker_gpu_milliseconds;
    }, QWEN3_TTS_TALKER_STATUS_STATE, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_model_batch_next_frame_begin_v2(
    Qwen3TtsModelHandle model_handle,
    Qwen3TtsSessionHandle* sessions,
    int32_t session_count,
    const int32_t* trailing_text_token_ids,
    const Qwen3TtsSamplingConfig* talker_sampling,
    const Qwen3TtsSamplingConfig* predictor_sampling,
    Qwen3TtsDeviceFrameViewV2* outputs,
    char* error,
    size_t error_capacity
) {
    if (sessions == nullptr || session_count <= 0
        || session_count > QWEN3_TTS_MAX_BATCH_SESSIONS
        || trailing_text_token_ids == nullptr || talker_sampling == nullptr
        || predictor_sampling == nullptr || outputs == nullptr) {
        write_error(error, error_capacity, "batch frame arguments are null or out of range");
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    for (int32_t index = 0; index < session_count; ++index) {
        if (sessions[index] == nullptr
            || outputs[index].struct_size != sizeof(Qwen3TtsDeviceFrameViewV2)
            || outputs[index].reserved != 0) {
            write_error(
                error,
                error_capacity,
                "batch frame outputs have an unsupported size or nonzero reserved field"
            );
            return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
        }
    }
    return protect([&] {
        TalkerContext* contexts[QWEN3_TTS_MAX_BATCH_SESSIONS];
        int trailing[QWEN3_TTS_MAX_BATCH_SESSIONS];
        for (int32_t index = 0; index < session_count; ++index) {
            contexts[index] = session(sessions[index]);
            trailing[index] = trailing_text_token_ids[index];
        }
        TalkerContext::batch_begin_frames(
            model(model_handle).get(),
            contexts,
            session_count,
            trailing,
            talker_sampling,
            predictor_sampling,
            outputs
        );
    }, QWEN3_TTS_TALKER_STATUS_STATE, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_session_state_info(
    Qwen3TtsSessionHandle handle,
    Qwen3TtsTalkerStateInfo* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "session state output pointer is null");
        return QWEN3_TTS_TALKER_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        *output = session(handle)->state_info();
    }, QWEN3_TTS_TALKER_STATUS_STATE, error, error_capacity);
}
