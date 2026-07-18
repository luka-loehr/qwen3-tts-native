#include "talker_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace qwen3_tts {
namespace {

constexpr int kThreads = 256;
constexpr int kSampleThreads = 1'024;   // sampling kernel runs one block, one SM
constexpr int kCandidateCap = 512;      // survivors gathered before the small sort
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

// "a ranks earlier than b": higher value first, ties broken by lower index.
// IEEE == / > match the reference argmax predicate (so +0/-0 tie by index).
__device__ __forceinline__ bool sort_before(float av, int ai, float bv, int bi) {
    return (av > bv) || (av == bv && ai < bi);
}

// Monotonic float->uint32: higher float value maps to higher key. -0 is
// canonicalized to +0 so it ties with +0 in the (rare) exact-zero case,
// preserving the IEEE-equality semantics of the selection order.
__device__ __forceinline__ unsigned int order_key(float v) {
    unsigned int u = __float_as_uint(v);
    if (u == 0x80000000u) u = 0u;  // -0 -> +0
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// In-place bitonic sort of a padded power-of-two shared array by sort_before
// (value desc, ties by lower index). Block-wide; call with all threads.
__device__ void bitonic_sort_shared(float* v, int* idx, int m,
                                     int tid, int nthreads) {
    for (int k = 2; k <= m; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < m; i += nthreads) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const float vi = v[i];
                    const float vj = v[ixj];
                    const int ii = idx[i];
                    const int ij = idx[ixj];
                    const bool ascending = ((i & k) == 0);
                    const bool swap = ascending
                        ? sort_before(vj, ij, vi, ii)
                        : sort_before(vi, ii, vj, ij);
                    if (swap) {
                        v[i] = vj;
                        v[ixj] = vi;
                        idx[i] = ij;
                        idx[ixj] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Dynamic shared-memory footprint required by sample_logits_kernel.
size_t sample_shared_bytes(int vocabulary) {
    int n = 1;
    while (n < vocabulary) {
        n <<= 1;
    }
    return static_cast<size_t>(n) * sizeof(float)             // values
         + static_cast<size_t>(n) * sizeof(int)               // indices
         + static_cast<size_t>(vocabulary) * sizeof(double)   // weights
         + static_cast<size_t>(kSampleThreads) * sizeof(float)  // rval
         + static_cast<size_t>(kSampleThreads) * sizeof(int)    // ridx
         + static_cast<size_t>(256) * sizeof(unsigned int)      // hist
         + static_cast<size_t>(kCandidateCap) * sizeof(float)   // cval
         + static_cast<size_t>(kCandidateCap) * sizeof(int);    // cidx
}

// Optimized token sampler. Semantics are bit-identical to the original
// sequential-argmax implementation: it produces the top-`selections` logits
// in the same total order (value desc, ties by lower index) and then runs the
// unchanged softmax / top-p / single-draw tail with the same double-precision
// summation order and the same single RNG draw. The k sequential block-argmax
// passes are replaced by (a) a plain argmax for do_sample==0 and the
// direct-categorical path, (b) an exact radix-select of the k-th value
// threshold followed by a small bitonic sort of the survivors for bounded
// top_k, and (c) a full bitonic sort for the top_k==0 full-vocabulary case.
// exp() is evaluated in parallel but summed sequentially so the accumulation
// is bit-for-bit. Single launch, single block; safe to capture in CUDA graphs.
__global__ void sample_logits_kernel(
    const __nv_bfloat16* logits,
    int vocabulary,
    bool suppress_talker_reserved,
    int codec_eos_token,
    const int* semantic_history,
    const int* semantic_history_count_ptr,
    int semantic_history_count_fallback,
    int do_sample,
    int top_k,
    float top_p,
    float temperature,
    float repetition_penalty,
    uint64_t* random_state,
    int* selected_token
) {
    const float kNegInf = -__int_as_float(0x7f800000);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    int n = 1;
    while (n < vocabulary) {
        n <<= 1;
    }

    extern __shared__ unsigned char smem[];
    float* values = reinterpret_cast<float*>(smem);            // n floats
    int* indices = reinterpret_cast<int*>(values + n);         // n ints
    double* weights = reinterpret_cast<double*>(indices + n);  // vocabulary doubles
    float* rval = reinterpret_cast<float*>(weights + vocabulary);  // kSampleThreads
    int* ridx = reinterpret_cast<int*>(rval + kSampleThreads);     // kSampleThreads
    unsigned int* hist = reinterpret_cast<unsigned int*>(ridx + kSampleThreads);  // 256
    float* cval = reinterpret_cast<float*>(hist + 256);         // kCandidateCap
    int* cidx = reinterpret_cast<int*>(cval + kCandidateCap);   // kCandidateCap

    __shared__ int s_above;
    __shared__ unsigned int s_prefix;
    __shared__ int s_counter;
    __shared__ int s_fallback;

    const int semantic_history_count = semantic_history_count_ptr != nullptr
        ? *semantic_history_count_ptr
        : semantic_history_count_fallback;

    // Adjusted logits in index order; pad the tail with -inf and out-of-range,
    // strictly-larger indices so padding always sorts after every real token.
    for (int t = tid; t < n; t += nthreads) {
        if (t < vocabulary) {
            values[t] = adjusted_logit(
                logits, t, suppress_talker_reserved, codec_eos_token,
                semantic_history, semantic_history_count, repetition_penalty);
            indices[t] = t;
        } else {
            values[t] = kNegInf;
            indices[t] = vocabulary + t;
        }
    }
    __syncthreads();

    const bool direct_categorical = do_sample != 0 && top_k == 0 && top_p >= 1.0f;
    const int selections = do_sample == 0
        ? 1
        : (top_k == 0 ? vocabulary : min(top_k, vocabulary));

    // do_sample == 0 (argmax, no draw) and direct_categorical both need argmax.
    if (do_sample == 0 || direct_categorical) {
        float lv = kNegInf;
        int li = -1;
        for (int t = tid; t < vocabulary; t += nthreads) {
            const float v = values[t];
            if (v > lv || (v == lv && (li < 0 || t < li))) {
                lv = v;
                li = t;
            }
        }
        rval[tid] = lv;
        ridx[tid] = li;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                const float v = rval[tid + stride];
                const int idx = ridx[tid + stride];
                if (v > rval[tid]
                    || (v == rval[tid] && idx >= 0
                        && (ridx[tid] < 0 || idx < ridx[tid]))) {
                    rval[tid] = v;
                    ridx[tid] = idx;
                }
            }
            __syncthreads();
        }

        if (do_sample == 0) {
            if (tid == 0) {
                *selected_token = ridx[0];
            }
            return;
        }

        const float maximum = rval[0] / temperature;
        for (int t = tid; t < vocabulary; t += nthreads) {
            weights[t] = exp(static_cast<double>(values[t] / temperature - maximum));
        }
        __syncthreads();
        if (tid == 0) {
            double total = 0.0;
            for (int t = 0; t < vocabulary; ++t) {
                total += weights[t];
            }
            const double target =
                static_cast<double>(uniform_random(random_state)) * total;
            double cumulative = 0.0;
            int selected = ridx[0];
            for (int t = 0; t < vocabulary; ++t) {
                cumulative += weights[t];
                if (cumulative >= target) {
                    selected = t;
                    break;
                }
            }
            *selected_token = selected;
        }
        return;
    }

    // General top-k / top-p path: produce the top-`selections` candidates in
    // the same total order the original selection loop produced.
    float* sv;
    int* si;

    if (selections > kCandidateCap) {
        bitonic_sort_shared(values, indices, n, tid, nthreads);
        sv = values;
        si = indices;
    } else {
        if (tid == 0) {
            s_above = 0;
            s_prefix = 0u;
            s_fallback = 0;
        }
        __syncthreads();
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += nthreads) {
                hist[i] = 0u;
            }
            __syncthreads();
            const unsigned int pref = s_prefix;
            const int hs = shift + 8;
            for (int t = tid; t < vocabulary; t += nthreads) {
                const unsigned int ok = order_key(values[t]);
                const bool match = (hs >= 32) || ((ok >> hs) == (pref >> hs));
                if (match) {
                    atomicAdd(&hist[(ok >> shift) & 0xFFu], 1u);
                }
            }
            __syncthreads();
            if (tid == 0) {
                int cum = s_above;
                int chosen = 0;
                for (int b = 255; b >= 0; --b) {
                    if (cum + static_cast<int>(hist[b]) >= selections) {
                        chosen = b;
                        break;
                    }
                    cum += static_cast<int>(hist[b]);
                }
                s_above = cum;
                s_prefix = pref | (static_cast<unsigned int>(chosen) << shift);
            }
            __syncthreads();
        }
        const unsigned int okeyT = s_prefix;

        if (tid == 0) {
            s_counter = 0;
        }
        __syncthreads();
        for (int t = tid; t < vocabulary; t += nthreads) {
            if (order_key(values[t]) >= okeyT) {
                const int pos = atomicAdd(&s_counter, 1);
                if (pos < kCandidateCap) {
                    cval[pos] = values[t];
                    cidx[pos] = indices[t];
                } else {
                    s_fallback = 1;
                }
            }
        }
        __syncthreads();

        if (s_fallback) {
            bitonic_sort_shared(values, indices, n, tid, nthreads);
            sv = values;
            si = indices;
        } else {
            const int count = s_counter;
            int cpad = 1;
            while (cpad < count) cpad <<= 1;
            for (int i = tid + count; i < cpad; i += nthreads) {
                cval[i] = kNegInf;
                cidx[i] = vocabulary + i;
            }
            __syncthreads();
            bitonic_sort_shared(cval, cidx, cpad, tid, nthreads);
            sv = cval;
            si = cidx;
        }
    }

    const float maximum = sv[0] / temperature;
    for (int r = tid; r < selections; r += nthreads) {
        weights[r] = exp(static_cast<double>(sv[r] / temperature - maximum));
    }
    __syncthreads();

    if (tid == 0) {
        double denominator = 0.0;
        for (int r = 0; r < selections; ++r) {
            denominator += weights[r];
        }
        double cumulative = 0.0;
        int retained = selections;
        for (int r = 0; r < selections; ++r) {
            cumulative += weights[r] / denominator;
            if (cumulative >= static_cast<double>(top_p)) {
                retained = r + 1;
                break;
            }
        }
        double retained_total = 0.0;
        for (int r = 0; r < retained; ++r) {
            retained_total += weights[r];
        }
        const double target =
            static_cast<double>(uniform_random(random_state)) * retained_total;
        cumulative = 0.0;
        int selected = si[retained - 1];
        for (int r = 0; r < retained; ++r) {
            cumulative += weights[r];
            if (cumulative >= target) {
                selected = si[r];
                break;
            }
        }
        *selected_token = selected;
    }
}

// Raise the sampler's dynamic shared-memory ceiling once (idempotent; not a
// stream operation, so it is safe alongside CUDA-graph capture).
void ensure_sample_shared_limit() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        cudaFuncSetAttribute(
            sample_logits_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            96 * 1024);
    });
}

__global__ void store_token_kernel(int* tokens, int index, int value) {
    if (threadIdx.x == 0) {
        tokens[index] = value;
    }
}

__global__ void store_sampled_token_kernel(
    int* tokens,
    const int* index_ptr,
    int index_fallback,
    const int* sampled_token
) {
    if (threadIdx.x == 0) {
        const int index = index_ptr != nullptr ? *index_ptr : index_fallback;
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
    ensure_sample_shared_limit();
    const size_t shared_bytes = sample_shared_bytes(vocabulary);
    sample_logits_kernel<<<1, kSampleThreads, shared_bytes, stream>>>(
        logits,
        vocabulary,
        suppress_talker_reserved,
        codec_eos_token,
        semantic_history,
        nullptr,
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

cudaError_t launch_sample_logits_at(
    const __nv_bfloat16* logits,
    int vocabulary,
    bool suppress_talker_reserved,
    int codec_eos_token,
    const int* semantic_history,
    const int* semantic_history_count,
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
    ensure_sample_shared_limit();
    const size_t shared_bytes = sample_shared_bytes(vocabulary);
    sample_logits_kernel<<<1, kSampleThreads, shared_bytes, stream>>>(
        logits,
        vocabulary,
        suppress_talker_reserved,
        codec_eos_token,
        semantic_history,
        semantic_history_count,
        0,
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
    store_sampled_token_kernel<<<1, 1, 0, stream>>>(tokens, nullptr, index, sampled_token);
    return cudaGetLastError();
}

cudaError_t launch_store_sampled_token_at(
    int* tokens,
    const int* index,
    const int* sampled_token,
    cudaStream_t stream
) {
    store_sampled_token_kernel<<<1, 1, 0, stream>>>(tokens, index, 0, sampled_token);
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
