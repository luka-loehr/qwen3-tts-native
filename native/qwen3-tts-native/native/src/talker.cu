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
#include <numeric>
#include <random>
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

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status)
        );
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
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

class TalkerContext {
public:
    TalkerContext(int device_index, int max_sequence_length, uint64_t seed)
        : device_index_(device_index),
          max_sequence_length_(max_sequence_length),
          prefill_gemm_capacity_(std::min(max_sequence_length, kPrefillGemmCapacity)),
          trace_enabled_(std::getenv("QWEN3_TTS_PARITY_TRACE") != nullptr),
          stage_dump_enabled_(std::getenv("QWEN3_TTS_STAGE_DUMP") != nullptr),
          random_(seed) {
        if (max_sequence_length < 16 || max_sequence_length > 8'192) {
            throw std::runtime_error("max sequence length must be in [16, 8192]");
        }
        check_cuda(cudaSetDevice(device_index_), "cudaSetDevice");
        try {
            check_cuda(
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags"
            );
            check_cublas(cublasCreate(&cublas_), "cublasCreate");
            check_cublas(cublasSetStream(cublas_, stream_), "cublasSetStream");
            check_cublas(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
            check_cuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
            check_cuda(cudaEventCreate(&stop_), "cudaEventCreate(stop)");

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
        } catch (...) {
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

    void upload_tensor(
        const char* name,
        const void* data,
        uint64_t byte_size,
        int rank,
        const uint64_t* shape
    ) {
        if (weights_finalized_) {
            throw std::runtime_error("weights are already finalized");
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
            if (shape[index] == 0 || elements > std::numeric_limits<uint64_t>::max() / shape[index]) {
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
                stream_
            ),
            "cudaMemcpyAsync(weight H2D)"
        );
        weight_bytes_ += byte_size;
        tensors_.emplace(tensor_name, std::move(tensor));
    }

    Qwen3TtsTalkerMemory finalize_weights() {
        if (weights_finalized_) {
            throw std::runtime_error("weights are already finalized");
        }
        if (tensors_.size() != 404) {
            throw std::runtime_error(
                "expected exactly 404 VoiceDesign tensors, found "
                    + std::to_string(tensors_.size())
            );
        }

        codec_embedding_ = require("talker.model.codec_embedding.weight", {kTalkerVocabulary, kTalkerHidden});
        text_embedding_ = require("talker.model.text_embedding.weight", {kTextVocabulary, kTalkerHidden});
        text_fc1_ = require("talker.text_projection.linear_fc1.weight", {kTalkerHidden, kTalkerHidden});
        text_fc1_bias_ = require("talker.text_projection.linear_fc1.bias", {kTalkerHidden});
        text_fc2_ = require("talker.text_projection.linear_fc2.weight", {kTalkerHidden, kTalkerHidden});
        text_fc2_bias_ = require("talker.text_projection.linear_fc2.bias", {kTalkerHidden});
        talker_norm_ = require("talker.model.norm.weight", {kTalkerHidden});
        codec_head_ = require("talker.codec_head.weight", {kTalkerVocabulary, kTalkerHidden});

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
        predictor_norm_ = require("talker.code_predictor.model.norm.weight", {kPredictorHidden});

        predictor_layers_.reserve(kPredictorLayers);
        for (int layer = 0; layer < kPredictorLayers; ++layer) {
            predictor_layers_.push_back(load_layer(
                "talker.code_predictor.model.layers." + std::to_string(layer),
                kPredictorDimensions
            ));
        }
        for (int group = 0; group < kResidualCodebooks; ++group) {
            predictor_embeddings_[group] = require(
                "talker.code_predictor.model.codec_embedding." + std::to_string(group) + ".weight",
                {kPredictorVocabulary, kTalkerHidden}
            );
            predictor_heads_[group] = require(
                "talker.code_predictor.lm_head." + std::to_string(group) + ".weight",
                {kPredictorVocabulary, kPredictorHidden}
            );
        }

        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(weight upload)");
        weights_finalized_ = true;
        return memory_stats();
    }

    void reset(uint64_t seed) {
        position_ = 0;
        generated_semantic_.clear();
        random_.seed(seed);
    }

    Qwen3TtsTalkerPrefillResult prefill(
        const int32_t* text_ids,
        const int32_t* codec_ids,
        int token_count,
        const Qwen3TtsSamplingConfig& sampling
    ) {
        ensure_ready();
        if (text_ids == nullptr || codec_ids == nullptr || token_count <= 0) {
            throw std::runtime_error("prefill requires non-empty text and codec ID arrays");
        }
        if (token_count > max_sequence_length_) {
            throw std::runtime_error("prompt exceeds the configured KV-cache capacity");
        }
        position_ = 0;
        generated_semantic_.clear();
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
        const int first = sample_logits(kTalkerVocabulary, sampling, true);
        generated_semantic_.push_back(first);
        Qwen3TtsTalkerPrefillResult result{};
        result.first_semantic_token = static_cast<uint16_t>(first);
        result.prompt_tokens = static_cast<uint32_t>(token_count);
        result.talker_gpu_milliseconds = total_gpu_ms;
        return result;
    }

    Qwen3TtsCodecFrameResult next_frame(
        uint16_t semantic_token,
        int trailing_text_token,
        const Qwen3TtsSamplingConfig& talker_sampling,
        const Qwen3TtsSamplingConfig& predictor_sampling
    ) {
        ensure_ready();
        if (semantic_token == kCodecEos) {
            throw std::runtime_error("EOS must not be expanded into a codec frame");
        }
        if (semantic_token >= kTalkerVocabulary) {
            throw std::runtime_error("semantic token is outside the talker vocabulary");
        }
        if (trailing_text_token < 0 || trailing_text_token >= kTextVocabulary) {
            throw std::runtime_error("next-frame text token is outside the text vocabulary");
        }
        if (position_ >= max_sequence_length_) {
            throw std::runtime_error("talker KV cache is full");
        }

        Qwen3TtsCodecFrameResult result{};
        result.codes[0] = semantic_token;
        check_cuda(cudaEventRecord(start_, stream_), "cudaEventRecord(predictor start)");

        run_predictor_position(last_hidden_.as<__nv_bfloat16>(), 0, -1);
        const __nv_bfloat16* semantic_embedding = embedding_row(
            codec_embedding_,
            semantic_token,
            kTalkerVocabulary,
            kTalkerHidden
        );
        run_predictor_position(semantic_embedding, 1, 0);
        result.codes[1] = static_cast<uint16_t>(
            sample_logits(kPredictorVocabulary, predictor_sampling, false)
        );

        for (int residual = 1; residual < kResidualCodebooks; ++residual) {
            const __nv_bfloat16* input = embedding_row(
                predictor_embeddings_[residual - 1],
                result.codes[residual],
                kPredictorVocabulary,
                kTalkerHidden
            );
            run_predictor_position(input, residual + 1, residual);
            result.codes[residual + 1] = static_cast<uint16_t>(
                sample_logits(kPredictorVocabulary, predictor_sampling, false)
            );
        }
        check_cuda(cudaEventRecord(stop_, stream_), "cudaEventRecord(predictor stop)");
        check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize(predictor)");
        check_cuda(
            cudaEventElapsedTime(&result.predictor_gpu_milliseconds, start_, stop_),
            "cudaEventElapsedTime(predictor)"
        );

        prepare_generated_embedding(result.codes, trailing_text_token);
        result.talker_position = static_cast<uint32_t>(position_);
        result.talker_gpu_milliseconds = run_talker_step(position_, true);
        ++position_;
        const int next_semantic = sample_logits(kTalkerVocabulary, talker_sampling, true);
        result.next_semantic_token = static_cast<uint16_t>(next_semantic);
        generated_semantic_.push_back(next_semantic);
        return result;
    }

private:
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
            require(prefix + ".input_layernorm.weight", {static_cast<uint64_t>(dimensions.hidden)}),
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

    void ensure_ready() const {
        if (!weights_finalized_) {
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
                CUBLAS_COMPUTE_32F_FAST_16BF,
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
                CUBLAS_COMPUTE_32F_FAST_16BF,
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
        gemv(text_fc1_, embedding, projection_.as<__nv_bfloat16>(), kTalkerHidden, kTalkerHidden);
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
        gemv(text_fc2_, projection_.as<__nv_bfloat16>(), output, kTalkerHidden, kTalkerHidden);
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
                CUBLAS_COMPUTE_32F_FAST_16BF,
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
                CUBLAS_COMPUTE_32F_FAST_16BF,
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
        std::vector<LayerCache>& caches,
        const ModelDimensions& dimensions,
        int position
    ) {
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const DecoderWeights& layer = layers[layer_index];
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
            gemv(
                layer.q_projection,
                normalized_.as<__nv_bfloat16>(),
                query_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.query_width
            );
            gemv(
                layer.k_projection,
                normalized_.as<__nv_bfloat16>(),
                key_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.key_value_width
            );
            gemv(
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
            gemv(
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
            gemv(
                layer.gate_projection,
                normalized_.as<__nv_bfloat16>(),
                gate_.as<__nv_bfloat16>(),
                dimensions.hidden,
                dimensions.intermediate
            );
            gemv(
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
            gemv(
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

    float run_talker_step(int position, bool emit_logits) {
        check_cuda(cudaEventRecord(start_, stream_), "cudaEventRecord(talker start)");
        run_decoder(talker_layers_, talker_cache_, kTalkerDimensions, position);
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
            gemv(
                codec_head_,
                normalized_.as<__nv_bfloat16>(),
                logits_.as<__nv_bfloat16>(),
                kTalkerHidden,
                kTalkerVocabulary
            );
            trace_logits();
        }
        check_cuda(cudaEventRecord(stop_, stream_), "cudaEventRecord(talker stop)");
        check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize(talker)");
        float milliseconds = 0.0f;
        check_cuda(
            cudaEventElapsedTime(&milliseconds, start_, stop_),
            "cudaEventElapsedTime(talker)"
        );
        return milliseconds;
    }

    void run_predictor_position(const __nv_bfloat16* input, int position, int head) {
        gemv(
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
        run_decoder(predictor_layers_, predictor_cache_, kPredictorDimensions, position);
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
            gemv(
                predictor_heads_[head],
                normalized_.as<__nv_bfloat16>(),
                logits_.as<__nv_bfloat16>(),
                kPredictorHidden,
                kPredictorVocabulary
            );
        }
    }

    void prepare_generated_embedding(const uint16_t* codes, int text_token) {
        const __nv_bfloat16* semantic = embedding_row(
            codec_embedding_,
            codes[0],
            kTalkerVocabulary,
            kTalkerHidden
        );
        check_cuda(
            cudaMemcpyAsync(
                hidden_.as<__nv_bfloat16>(),
                semantic,
                kTalkerHidden * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice,
                stream_
            ),
            "copy semantic embedding"
        );
        for (int residual = 0; residual < kResidualCodebooks; ++residual) {
            check_cuda(
                qwen3_tts::launch_add_in_place(
                    hidden_.as<__nv_bfloat16>(),
                    embedding_row(
                        predictor_embeddings_[residual],
                        codes[residual + 1],
                        kPredictorVocabulary,
                        kTalkerHidden
                    ),
                    kTalkerHidden,
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

    int sample_logits(
        int vocabulary,
        const Qwen3TtsSamplingConfig& config,
        bool talker
    ) {
        if (config.top_k < 0 || config.top_p <= 0.0f || config.top_p > 1.0f
            || config.temperature <= 0.0f || config.repetition_penalty <= 0.0f) {
            throw std::runtime_error("invalid sampling configuration");
        }
        std::vector<__nv_bfloat16> encoded(vocabulary);
        check_cuda(
            cudaMemcpyAsync(
                encoded.data(),
                logits_.as<__nv_bfloat16>(),
                static_cast<size_t>(vocabulary) * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToHost,
                stream_
            ),
            "copy logits to host"
        );
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(logits)");
        std::vector<float> logits(vocabulary);
        for (int token = 0; token < vocabulary; ++token) {
            logits[token] = __bfloat162float(encoded[token]);
        }

        if (talker) {
            for (int token = kPredictorVocabulary; token < kTalkerVocabulary; ++token) {
                if (token != kCodecEos) {
                    logits[token] = -std::numeric_limits<float>::infinity();
                }
            }
            for (const int token : generated_semantic_) {
                if (token >= 0 && token < vocabulary) {
                    logits[token] = logits[token] < 0.0f
                        ? logits[token] * config.repetition_penalty
                        : logits[token] / config.repetition_penalty;
                }
            }
        }

        if (config.do_sample == 0) {
            return static_cast<int>(
                std::max_element(logits.begin(), logits.end()) - logits.begin()
            );
        }

        std::vector<int> indices(vocabulary);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&] (int left, int right) {
            if (logits[left] == logits[right]) {
                return left < right;
            }
            return logits[left] > logits[right];
        });
        const int top_k = config.top_k == 0
            ? vocabulary
            : std::min(config.top_k, vocabulary);
        indices.resize(top_k);
        const float maximum = logits[indices.front()] / config.temperature;
        std::vector<double> probabilities(indices.size());
        double denominator = 0.0;
        for (size_t index = 0; index < indices.size(); ++index) {
            probabilities[index] = std::exp(
                static_cast<double>(logits[indices[index]] / config.temperature - maximum)
            );
            denominator += probabilities[index];
        }
        double cumulative = 0.0;
        size_t retained = probabilities.size();
        for (size_t index = 0; index < probabilities.size(); ++index) {
            cumulative += probabilities[index] / denominator;
            if (cumulative >= config.top_p) {
                retained = index + 1;
                break;
            }
        }
        indices.resize(retained);
        probabilities.resize(retained);
        std::discrete_distribution<size_t> distribution(
            probabilities.begin(),
            probabilities.end()
        );
        return indices[distribution(random_)];
    }

    Qwen3TtsTalkerMemory memory_stats() const {
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
            + packed_value_.bytes() + packed_attention_.bytes() + attention_scores_.bytes();
        Qwen3TtsTalkerMemory result{};
        result.weight_bytes = weight_bytes_;
        result.talker_kv_bytes = talker_kv;
        result.predictor_kv_bytes = predictor_kv;
        result.workspace_bytes = workspace;
        result.max_sequence_length = static_cast<uint32_t>(max_sequence_length_);
        result.tensor_count = static_cast<uint32_t>(tensors_.size());
        return result;
    }

    int device_index_;
    int max_sequence_length_;
    int prefill_gemm_capacity_;
    bool trace_enabled_;
    bool stage_dump_enabled_;
    int position_ = 0;
    bool trace_active_ = false;
    bool weights_finalized_ = false;
    uint64_t weight_bytes_ = 0;
    std::mt19937_64 random_;
    std::vector<int> generated_semantic_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    std::unordered_map<std::string, DeviceTensor> tensors_;
    std::vector<DecoderWeights> talker_layers_;
    std::vector<DecoderWeights> predictor_layers_;
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
};

TalkerContext* context(Qwen3TtsTalkerHandle handle) {
    if (handle == nullptr) {
        throw std::runtime_error("talker handle is null");
    }
    return static_cast<TalkerContext*>(handle);
}

template <typename Operation>
int32_t protect(Operation&& operation, char* error, size_t error_capacity) {
    try {
        operation();
        write_error(error, error_capacity, "");
        return 0;
    } catch (const std::exception& exception) {
        write_error(error, error_capacity, exception.what());
        return -1;
    }
}

}  // namespace

extern "C" QWEN3_TTS_API int32_t qwen3_tts_talker_create(
    int32_t device_index,
    int32_t max_sequence_length,
    uint64_t random_seed,
    Qwen3TtsTalkerHandle* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "talker output handle pointer is null");
        return -1;
    }
    return protect([&] {
        *output = new TalkerContext(device_index, max_sequence_length, random_seed);
    }, error, error_capacity);
}

extern "C" QWEN3_TTS_API void qwen3_tts_talker_destroy(Qwen3TtsTalkerHandle handle) {
    delete static_cast<TalkerContext*>(handle);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_talker_upload_tensor(
    Qwen3TtsTalkerHandle handle,
    const char* name,
    const void* bf16_data,
    uint64_t byte_size,
    int32_t rank,
    const uint64_t* shape,
    char* error,
    size_t error_capacity
) {
    return protect([&] {
        context(handle)->upload_tensor(name, bf16_data, byte_size, rank, shape);
    }, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_talker_finalize_weights(
    Qwen3TtsTalkerHandle handle,
    Qwen3TtsTalkerMemory* memory,
    char* error,
    size_t error_capacity
) {
    if (memory == nullptr) {
        write_error(error, error_capacity, "talker memory output pointer is null");
        return -1;
    }
    return protect([&] {
        *memory = context(handle)->finalize_weights();
    }, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_talker_reset(
    Qwen3TtsTalkerHandle handle,
    uint64_t random_seed,
    char* error,
    size_t error_capacity
) {
    return protect([&] {
        context(handle)->reset(random_seed);
    }, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_talker_prefill(
    Qwen3TtsTalkerHandle handle,
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
        return -1;
    }
    return protect([&] {
        *output = context(handle)->prefill(
            text_token_ids,
            codec_token_ids,
            token_count,
            sampling
        );
    }, error, error_capacity);
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_talker_next_frame(
    Qwen3TtsTalkerHandle handle,
    uint16_t semantic_token,
    int32_t trailing_text_token_id,
    Qwen3TtsSamplingConfig talker_sampling,
    Qwen3TtsSamplingConfig predictor_sampling,
    Qwen3TtsCodecFrameResult* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "codec-frame output pointer is null");
        return -1;
    }
    return protect([&] {
        *output = context(handle)->next_frame(
            semantic_token,
            trailing_text_token_id,
            talker_sampling,
            predictor_sampling
        );
    }, error, error_capacity);
}
