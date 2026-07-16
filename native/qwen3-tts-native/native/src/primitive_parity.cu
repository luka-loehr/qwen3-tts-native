#include "qwen3_tts_native.h"
#include "talker_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) {
        const cudaError_t status = cudaMalloc(&pointer_, bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMalloc failed: ") + cudaGetErrorString(status)
            );
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    template <typename T>
    T* as() {
        return static_cast<T*>(pointer_);
    }

private:
    void* pointer_ = nullptr;
};

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status)
        );
    }
}

void write_error(char* destination, size_t capacity, const char* message) {
    if (destination != nullptr && capacity != 0) {
        std::snprintf(destination, capacity, "%s", message);
    }
}

std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& values) {
    std::vector<__nv_bfloat16> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(), [] (float value) {
        return __float2bfloat16(value);
    });
    return converted;
}

std::vector<float> from_bf16(const std::vector<__nv_bfloat16>& values) {
    std::vector<float> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(), [] (__nv_bfloat16 value) {
        return __bfloat162float(value);
    });
    return converted;
}

float maximum_error(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("parity vector size mismatch");
    }
    float maximum = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    return maximum;
}

template <typename T>
void upload(DeviceBuffer& destination, const std::vector<T>& source) {
    check(
        cudaMemcpy(
            destination.as<T>(),
            source.data(),
            source.size() * sizeof(T),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy(H2D)"
    );
}

template <typename T>
std::vector<T> download(DeviceBuffer& source, size_t elements) {
    std::vector<T> output(elements);
    check(
        cudaMemcpy(
            output.data(),
            source.as<T>(),
            elements * sizeof(T),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy(D2H)"
    );
    return output;
}

float validate_rms_norm(cudaStream_t stream) {
    constexpr int width = 128;
    std::vector<float> input(width);
    std::vector<float> weight(width);
    for (int index = 0; index < width; ++index) {
        input[index] = std::sin(index * 0.071f) * 1.7f;
        weight[index] = 0.75f + std::cos(index * 0.037f) * 0.2f;
    }
    const auto input_bf16 = to_bf16(input);
    const auto weight_bf16 = to_bf16(weight);
    input = from_bf16(input_bf16);
    weight = from_bf16(weight_bf16);

    float square_sum = 0.0f;
    for (const float value : input) {
        square_sum += value * value;
    }
    const float inverse_rms = 1.0f / std::sqrt(square_sum / width + 1.0e-6f);
    std::vector<float> expected(width);
    for (int index = 0; index < width; ++index) {
        expected[index] = __bfloat162float(
            __float2bfloat16(input[index] * inverse_rms * weight[index])
        );
    }

    DeviceBuffer device_input(width * sizeof(__nv_bfloat16));
    DeviceBuffer device_weight(width * sizeof(__nv_bfloat16));
    DeviceBuffer device_output(width * sizeof(__nv_bfloat16));
    upload(device_input, input_bf16);
    upload(device_weight, weight_bf16);
    check(
        qwen3_tts::launch_rms_norm(
            device_input.as<__nv_bfloat16>(),
            device_weight.as<__nv_bfloat16>(),
            device_output.as<__nv_bfloat16>(),
            width,
            1.0e-6f,
            stream
        ),
        "launch_rms_norm"
    );
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(rms_norm)");
    return maximum_error(from_bf16(download<__nv_bfloat16>(device_output, width)), expected);
}

float validate_rope(cudaStream_t stream) {
    constexpr int heads = 2;
    constexpr int dimension = 128;
    constexpr int half = dimension / 2;
    constexpr int position = 37;
    constexpr float theta = 1.0e6f;
    std::vector<float> values(heads * dimension);
    for (size_t index = 0; index < values.size(); ++index) {
        values[index] = std::sin(static_cast<float>(index) * 0.019f);
    }
    auto values_bf16 = to_bf16(values);
    values = from_bf16(values_bf16);
    std::vector<float> expected(values.size());
    for (int head = 0; head < heads; ++head) {
        for (int index = 0; index < half; ++index) {
            const float exponent = static_cast<float>(2 * index) / dimension;
            const float angle = position * std::pow(theta, -exponent);
            const float first = values[head * dimension + index];
            const float second = values[head * dimension + index + half];
            expected[head * dimension + index] = __bfloat162float(
                __float2bfloat16(first * std::cos(angle) - second * std::sin(angle))
            );
            expected[head * dimension + index + half] = __bfloat162float(
                __float2bfloat16(second * std::cos(angle) + first * std::sin(angle))
            );
        }
    }

    DeviceBuffer device(values_bf16.size() * sizeof(__nv_bfloat16));
    upload(device, values_bf16);
    check(
        qwen3_tts::launch_rope(
            device.as<__nv_bfloat16>(),
            heads,
            dimension,
            position,
            theta,
            stream
        ),
        "launch_rope"
    );
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(rope)");
    return maximum_error(
        from_bf16(download<__nv_bfloat16>(device, values_bf16.size())),
        expected
    );
}

float validate_attention(cudaStream_t stream) {
    constexpr int query_heads = 4;
    constexpr int key_value_heads = 2;
    constexpr int dimension = 32;
    constexpr int sequence = 7;
    std::vector<float> query(query_heads * dimension);
    std::vector<float> keys(sequence * key_value_heads * dimension);
    std::vector<float> values(sequence * key_value_heads * dimension);
    for (size_t index = 0; index < query.size(); ++index) {
        query[index] = std::sin(static_cast<float>(index) * 0.031f) * 0.4f;
    }
    for (size_t index = 0; index < keys.size(); ++index) {
        keys[index] = std::cos(static_cast<float>(index) * 0.023f) * 0.3f;
        values[index] = std::sin(static_cast<float>(index) * 0.017f) * 0.5f;
    }
    const auto query_bf16 = to_bf16(query);
    const auto key_bf16 = to_bf16(keys);
    const auto value_bf16 = to_bf16(values);
    query = from_bf16(query_bf16);
    keys = from_bf16(key_bf16);
    values = from_bf16(value_bf16);

    std::vector<float> expected(query.size());
    std::vector<float> scores(sequence);
    for (int query_head = 0; query_head < query_heads; ++query_head) {
        const int key_value_head = query_head / (query_heads / key_value_heads);
        float maximum = -INFINITY;
        for (int position = 0; position < sequence; ++position) {
            float dot = 0.0f;
            for (int index = 0; index < dimension; ++index) {
                dot += query[query_head * dimension + index]
                    * keys[(position * key_value_heads + key_value_head) * dimension + index];
            }
            scores[position] = dot / std::sqrt(static_cast<float>(dimension));
            maximum = std::max(maximum, scores[position]);
        }
        float denominator = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int index = 0; index < dimension; ++index) {
            float accumulated = 0.0f;
            for (int position = 0; position < sequence; ++position) {
                accumulated += scores[position] / denominator
                    * values[(position * key_value_heads + key_value_head) * dimension + index];
            }
            expected[query_head * dimension + index] =
                __bfloat162float(__float2bfloat16(accumulated));
        }
    }

    DeviceBuffer device_query(query_bf16.size() * sizeof(__nv_bfloat16));
    DeviceBuffer device_keys(key_bf16.size() * sizeof(__nv_bfloat16));
    DeviceBuffer device_values(value_bf16.size() * sizeof(__nv_bfloat16));
    DeviceBuffer device_output(query_bf16.size() * sizeof(__nv_bfloat16));
    upload(device_query, query_bf16);
    upload(device_keys, key_bf16);
    upload(device_values, value_bf16);
    check(
        qwen3_tts::launch_causal_gqa_attention(
            device_query.as<__nv_bfloat16>(),
            device_keys.as<__nv_bfloat16>(),
            device_values.as<__nv_bfloat16>(),
            device_output.as<__nv_bfloat16>(),
            query_heads,
            key_value_heads,
            dimension,
            sequence,
            stream
        ),
        "launch_causal_gqa_attention"
    );
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(attention)");
    return maximum_error(
        from_bf16(download<__nv_bfloat16>(device_output, query_bf16.size())),
        expected
    );
}

float validate_silu_gate(cudaStream_t stream) {
    constexpr int width = 257;
    std::vector<float> gate(width);
    std::vector<float> up(width);
    for (int index = 0; index < width; ++index) {
        gate[index] = std::sin(index * 0.043f) * 2.0f;
        up[index] = std::cos(index * 0.029f);
    }
    auto gate_bf16 = to_bf16(gate);
    const auto up_bf16 = to_bf16(up);
    gate = from_bf16(gate_bf16);
    up = from_bf16(up_bf16);
    std::vector<float> expected(width);
    for (int index = 0; index < width; ++index) {
        const float activated = gate[index] / (1.0f + std::exp(-gate[index]));
        expected[index] = __bfloat162float(__float2bfloat16(activated * up[index]));
    }

    DeviceBuffer device_gate(gate_bf16.size() * sizeof(__nv_bfloat16));
    DeviceBuffer device_up(up_bf16.size() * sizeof(__nv_bfloat16));
    upload(device_gate, gate_bf16);
    upload(device_up, up_bf16);
    check(
        qwen3_tts::launch_silu_gate(
            device_gate.as<__nv_bfloat16>(),
            device_up.as<__nv_bfloat16>(),
            width,
            stream
        ),
        "launch_silu_gate"
    );
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(silu_gate)");
    return maximum_error(
        from_bf16(download<__nv_bfloat16>(device_gate, gate_bf16.size())),
        expected
    );
}

}  // namespace

extern "C" QWEN3_TTS_API int32_t qwen3_tts_validate_transformer_primitives(
    int32_t device_index,
    Qwen3TtsPrimitiveParity* output,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr) {
        write_error(error, error_capacity, "primitive parity output pointer is null");
        return -1;
    }
    cudaStream_t stream = nullptr;
    try {
        check(cudaSetDevice(device_index), "cudaSetDevice");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        Qwen3TtsPrimitiveParity result{};
        result.rms_norm_max_absolute_error = validate_rms_norm(stream);
        result.rope_max_absolute_error = validate_rope(stream);
        result.attention_max_absolute_error = validate_attention(stream);
        result.silu_gate_max_absolute_error = validate_silu_gate(stream);
        check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        stream = nullptr;

        constexpr float tolerance = 0.016f;
        if (result.rms_norm_max_absolute_error > tolerance
            || result.rope_max_absolute_error > tolerance
            || result.attention_max_absolute_error > tolerance
            || result.silu_gate_max_absolute_error > tolerance) {
            write_error(error, error_capacity, "a transformer primitive exceeded BF16 parity tolerance");
            return -2;
        }
        *output = result;
        write_error(error, error_capacity, "");
        return 0;
    } catch (const std::exception& exception) {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        write_error(error, error_capacity, exception.what());
        return -3;
    }
}
