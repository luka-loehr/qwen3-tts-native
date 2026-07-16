#include "talker_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace qwen3_tts {
namespace {

constexpr int kThreads = 256;
constexpr int kMaximumVocabulary = 3'072;
constexpr int kPredictorVocabulary = 2'048;
constexpr int kCodebooks = 16;

__device__ bool history_contains(const int* history, int count, int token) {
    for (int index = 0; index < count; ++index) {
        if (history[index] == token) {
            return true;
        }
    }
    return false;
}

__device__ float adjusted_logit(
    const __nv_bfloat16* logits,
    int token,
    bool suppress_talker_reserved,
    int codec_eos_token,
    const int* semantic_history,
    int semantic_history_count,
    float repetition_penalty
) {
    if (suppress_talker_reserved
        && token >= kPredictorVocabulary
        && token != codec_eos_token) {
        return -__int_as_float(0x7f800000);
    }
    float value = __bfloat162float(logits[token]);
    if (suppress_talker_reserved
        && repetition_penalty != 1.0f
        && history_contains(semantic_history, semantic_history_count, token)) {
        value = value < 0.0f ? value * repetition_penalty : value / repetition_penalty;
    }
    return value;
}

__device__ uint64_t next_random(uint64_t* state) {
    uint64_t value = *state;
    if (value == 0) {
        value = 0x9e3779b97f4a7c15ULL;
    }
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * 0x2545f4914f6cdd1dULL;
}

__device__ float uniform_random(uint64_t* state) {
    constexpr double scale = 1.0 / 9'007'199'254'740'992.0;
    const uint64_t value = next_random(state);
    return static_cast<float>(static_cast<double>(value >> 11) * scale);
}

__global__ void sample_logits_kernel(
    const __nv_bfloat16* logits,
    int vocabulary,
    bool suppress_talker_reserved,
    int codec_eos_token,
    const int* semantic_history,
    int semantic_history_count,
    int do_sample,
    int top_k,
    float top_p,
    float temperature,
    float repetition_penalty,
    uint64_t* random_state,
    int* selected_token
) {
    __shared__ float candidates[kMaximumVocabulary];
    __shared__ int selected_indices[kMaximumVocabulary];
    __shared__ float selected_logits[kMaximumVocabulary];
    __shared__ float reduced_values[kThreads];
    __shared__ int reduced_indices[kThreads];

    for (int token = threadIdx.x; token < vocabulary; token += blockDim.x) {
        candidates[token] = adjusted_logit(
            logits,
            token,
            suppress_talker_reserved,
            codec_eos_token,
            semantic_history,
            semantic_history_count,
            repetition_penalty
        );
    }
    __syncthreads();

    const bool direct_categorical = do_sample != 0 && top_k == 0 && top_p >= 1.0f;
    const int selections = do_sample == 0
        ? 1
        : (top_k == 0 ? vocabulary : min(top_k, vocabulary));

    for (int rank = 0; rank < selections; ++rank) {
        float local_value = -__int_as_float(0x7f800000);
        int local_index = -1;
        for (int token = threadIdx.x; token < vocabulary; token += blockDim.x) {
            const float value = candidates[token];
            if (value > local_value
                || (value == local_value && (local_index < 0 || token < local_index))) {
                local_value = value;
                local_index = token;
            }
        }
        reduced_values[threadIdx.x] = local_value;
        reduced_indices[threadIdx.x] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                const float value = reduced_values[threadIdx.x + stride];
                const int index = reduced_indices[threadIdx.x + stride];
                if (value > reduced_values[threadIdx.x]
                    || (value == reduced_values[threadIdx.x]
                        && index >= 0
                        && (reduced_indices[threadIdx.x] < 0
                            || index < reduced_indices[threadIdx.x]))) {
                    reduced_values[threadIdx.x] = value;
                    reduced_indices[threadIdx.x] = index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected_indices[rank] = reduced_indices[0];
            selected_logits[rank] = reduced_values[0];
            if (!direct_categorical && reduced_indices[0] >= 0) {
                candidates[reduced_indices[0]] = -__int_as_float(0x7f800000);
            }
        }
        __syncthreads();
        if (direct_categorical) {
            break;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }
    if (do_sample == 0) {
        *selected_token = selected_indices[0];
        return;
    }

    if (direct_categorical) {
        const float maximum = selected_logits[0] / temperature;
        double total = 0.0;
        for (int token = 0; token < vocabulary; ++token) {
            total += exp(static_cast<double>(candidates[token] / temperature - maximum));
        }
        const double target = static_cast<double>(uniform_random(random_state)) * total;
        double cumulative = 0.0;
        int selected = selected_indices[0];
        for (int token = 0; token < vocabulary; ++token) {
            cumulative += exp(static_cast<double>(candidates[token] / temperature - maximum));
            if (cumulative >= target) {
                selected = token;
                break;
            }
        }
        *selected_token = selected;
        return;
    }

    const float maximum = selected_logits[0] / temperature;
    double denominator = 0.0;
    for (int rank = 0; rank < selections; ++rank) {
        denominator += exp(
            static_cast<double>(selected_logits[rank] / temperature - maximum)
        );
    }
    double cumulative = 0.0;
    int retained = selections;
    for (int rank = 0; rank < selections; ++rank) {
        cumulative += exp(
            static_cast<double>(selected_logits[rank] / temperature - maximum)
        ) / denominator;
        if (cumulative >= static_cast<double>(top_p)) {
            retained = rank + 1;
            break;
        }
    }
    double retained_total = 0.0;
    for (int rank = 0; rank < retained; ++rank) {
        retained_total += exp(
            static_cast<double>(selected_logits[rank] / temperature - maximum)
        );
    }
    const double target = static_cast<double>(uniform_random(random_state)) * retained_total;
    cumulative = 0.0;
    int selected = selected_indices[retained - 1];
    for (int rank = 0; rank < retained; ++rank) {
        cumulative += exp(
            static_cast<double>(selected_logits[rank] / temperature - maximum)
        );
        if (cumulative >= target) {
            selected = selected_indices[rank];
            break;
        }
    }
    *selected_token = selected;
}

__global__ void store_token_kernel(int* tokens, int index, int value) {
    if (threadIdx.x == 0) {
        tokens[index] = value;
    }
}

__global__ void store_sampled_token_kernel(
    int* tokens,
    int index,
    const int* sampled_token
) {
    if (threadIdx.x == 0) {
        tokens[index] = *sampled_token;
    }
}

__global__ void pack_frame_codes_kernel(const int* tokens, uint16_t* codes) {
    const int index = threadIdx.x;
    if (index < kCodebooks) {
        codes[index] = static_cast<uint16_t>(tokens[index]);
    }
}

__global__ void gather_embedding_kernel(
    const __nv_bfloat16* table,
    int vocabulary,
    int width,
    const int* token,
    __nv_bfloat16* output,
    bool add
) {
    const int row = *token;
    if (row < 0 || row >= vocabulary) {
        return;
    }
    const __nv_bfloat16* source = table + static_cast<size_t>(row) * width;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < width;
         index += blockDim.x * gridDim.x) {
        if (add) {
            output[index] = __float2bfloat16(
                __bfloat162float(output[index]) + __bfloat162float(source[index])
            );
        } else {
            output[index] = source[index];
        }
    }
}

int blocks_for(int width) {
    return (width + kThreads - 1) / kThreads;
}

}  // namespace

cudaError_t launch_sample_logits(
    const __nv_bfloat16* logits,
    int vocabulary,
    bool suppress_talker_reserved,
    int codec_eos_token,
    const int* semantic_history,
    int semantic_history_count,
    int do_sample,
    int top_k,
    float top_p,
    float temperature,
    float repetition_penalty,
    uint64_t* random_state,
    int* selected_token,
    cudaStream_t stream
) {
    if (vocabulary <= 0 || vocabulary > kMaximumVocabulary) {
        return cudaErrorInvalidValue;
    }
    sample_logits_kernel<<<1, kThreads, 0, stream>>>(
        logits,
        vocabulary,
        suppress_talker_reserved,
        codec_eos_token,
        semantic_history,
        semantic_history_count,
        do_sample,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        random_state,
        selected_token
    );
    return cudaGetLastError();
}

cudaError_t launch_store_token(
    int* tokens,
    int index,
    int value,
    cudaStream_t stream
) {
    store_token_kernel<<<1, 1, 0, stream>>>(tokens, index, value);
    return cudaGetLastError();
}

cudaError_t launch_store_sampled_token(
    int* tokens,
    int index,
    const int* sampled_token,
    cudaStream_t stream
) {
    store_sampled_token_kernel<<<1, 1, 0, stream>>>(tokens, index, sampled_token);
    return cudaGetLastError();
}

cudaError_t launch_pack_frame_codes(
    const int* tokens,
    uint16_t* codes,
    cudaStream_t stream
) {
    pack_frame_codes_kernel<<<1, kCodebooks, 0, stream>>>(tokens, codes);
    return cudaGetLastError();
}

cudaError_t launch_gather_embedding(
    const __nv_bfloat16* table,
    int vocabulary,
    int width,
    const int* token,
    __nv_bfloat16* output,
    cudaStream_t stream
) {
    gather_embedding_kernel<<<blocks_for(width), kThreads, 0, stream>>>(
        table,
        vocabulary,
        width,
        token,
        output,
        false
    );
    return cudaGetLastError();
}

cudaError_t launch_add_embedding(
    __nv_bfloat16* destination,
    const __nv_bfloat16* table,
    int vocabulary,
    int width,
    const int* token,
    cudaStream_t stream
) {
    gather_embedding_kernel<<<blocks_for(width), kThreads, 0, stream>>>(
        table,
        vocabulary,
        width,
        token,
        destination,
        true
    );
    return cudaGetLastError();
}

}  // namespace qwen3_tts
