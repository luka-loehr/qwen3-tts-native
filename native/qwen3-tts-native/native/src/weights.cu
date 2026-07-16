#include "qwen3_tts_native.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>

struct Qwen3TtsDeviceBuffer {
    int32_t device_index;
    uint8_t* device_data;
    uint8_t* pinned_staging;
    cudaStream_t stream;
    uint64_t capacity_bytes;
    uint64_t staging_bytes;
    uint64_t uploaded_bytes;
    uint64_t upload_calls;
    double allocation_microseconds;
    double upload_microseconds;
    uint64_t free_before_bytes;
    uint64_t free_after_allocation_bytes;
};

namespace {

constexpr uint64_t kMaximumStagingBytes = 256ULL * 1024ULL * 1024ULL;

void write_error(char* destination, size_t capacity, const char* message) {
    if (destination == nullptr || capacity == 0) {
        return;
    }
    std::snprintf(destination, capacity, "%s", message);
}

int32_t cuda_failure(
    cudaError_t status,
    const char* operation,
    char* error,
    size_t error_capacity
) {
    char message[512];
    std::snprintf(
        message,
        sizeof(message),
        "%s failed: %s",
        operation,
        cudaGetErrorString(status)
    );
    write_error(error, error_capacity, message);
    return static_cast<int32_t>(status);
}

void release_buffer(Qwen3TtsDeviceBuffer* buffer) {
    if (buffer == nullptr) {
        return;
    }
    cudaSetDevice(buffer->device_index);
    if (buffer->stream != nullptr) {
        cudaStreamSynchronize(buffer->stream);
    }
    if (buffer->pinned_staging != nullptr) {
        cudaFreeHost(buffer->pinned_staging);
    }
    if (buffer->device_data != nullptr) {
        cudaFree(buffer->device_data);
    }
    if (buffer->stream != nullptr) {
        cudaStreamDestroy(buffer->stream);
    }
    delete buffer;
}

void fill_metrics(
    const Qwen3TtsDeviceBuffer* buffer,
    Qwen3TtsWeightUploadMetrics* output
) {
    if (output == nullptr || buffer == nullptr) {
        return;
    }
    std::memset(output, 0, sizeof(*output));
    output->device_index = buffer->device_index;
    output->allocation_bytes = buffer->capacity_bytes;
    output->pinned_staging_bytes = buffer->staging_bytes;
    output->uploaded_bytes = buffer->uploaded_bytes;
    output->upload_calls = buffer->upload_calls;
    output->free_before_bytes = buffer->free_before_bytes;
    output->free_after_allocation_bytes = buffer->free_after_allocation_bytes;
    output->allocation_microseconds =
        static_cast<float>(buffer->allocation_microseconds);
    output->upload_microseconds = static_cast<float>(buffer->upload_microseconds);
}

}  // namespace

extern "C" QWEN3_TTS_API int32_t qwen3_tts_device_buffer_create(
    int32_t device_index,
    uint64_t capacity_bytes,
    uint64_t staging_bytes,
    Qwen3TtsDeviceBuffer** output,
    Qwen3TtsWeightUploadMetrics* metrics,
    char* error,
    size_t error_capacity
) {
    if (output == nullptr || metrics == nullptr) {
        write_error(error, error_capacity, "output buffer or metrics pointer is null");
        return -1;
    }
    *output = nullptr;
    std::memset(metrics, 0, sizeof(*metrics));
    if (capacity_bytes == 0 ||
        capacity_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        write_error(error, error_capacity, "device capacity is zero or exceeds size_t");
        return -2;
    }
    if (staging_bytes == 0 || staging_bytes > kMaximumStagingBytes ||
        staging_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        write_error(
            error,
            error_capacity,
            "staging capacity must be in [1, 256 MiB] and fit size_t"
        );
        return -3;
    }

    cudaError_t status = cudaSetDevice(device_index);
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaSetDevice", error, error_capacity);
    }

    size_t free_before = 0;
    size_t total = 0;
    status = cudaMemGetInfo(&free_before, &total);
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaMemGetInfo(before)", error, error_capacity);
    }

    auto* buffer = new (std::nothrow) Qwen3TtsDeviceBuffer{};
    if (buffer == nullptr) {
        write_error(error, error_capacity, "failed to allocate device-buffer metadata");
        return -4;
    }
    buffer->device_index = device_index;
    buffer->capacity_bytes = capacity_bytes;
    buffer->staging_bytes = staging_bytes;
    buffer->free_before_bytes = static_cast<uint64_t>(free_before);

    const auto started = std::chrono::steady_clock::now();
    status = cudaStreamCreateWithFlags(&buffer->stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        release_buffer(buffer);
        return cuda_failure(status, "cudaStreamCreateWithFlags", error, error_capacity);
    }
    status = cudaMalloc(
        reinterpret_cast<void**>(&buffer->device_data),
        static_cast<size_t>(capacity_bytes)
    );
    if (status != cudaSuccess) {
        release_buffer(buffer);
        return cuda_failure(status, "cudaMalloc(weight arena)", error, error_capacity);
    }
    status = cudaHostAlloc(
        reinterpret_cast<void**>(&buffer->pinned_staging),
        static_cast<size_t>(staging_bytes),
        cudaHostAllocPortable
    );
    if (status != cudaSuccess) {
        release_buffer(buffer);
        return cuda_failure(status, "cudaHostAlloc(staging)", error, error_capacity);
    }

    size_t free_after = 0;
    status = cudaMemGetInfo(&free_after, &total);
    if (status != cudaSuccess) {
        release_buffer(buffer);
        return cuda_failure(status, "cudaMemGetInfo(after)", error, error_capacity);
    }
    const auto finished = std::chrono::steady_clock::now();
    buffer->allocation_microseconds =
        std::chrono::duration<double, std::micro>(finished - started).count();
    buffer->free_after_allocation_bytes = static_cast<uint64_t>(free_after);
    fill_metrics(buffer, metrics);
    *output = buffer;
    write_error(error, error_capacity, "");
    return 0;
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_device_buffer_upload(
    Qwen3TtsDeviceBuffer* buffer,
    uint64_t offset_bytes,
    const void* source,
    uint64_t bytes,
    char* error,
    size_t error_capacity
) {
    if (buffer == nullptr) {
        write_error(error, error_capacity, "device buffer is null");
        return -1;
    }
    if (bytes != 0 && source == nullptr) {
        write_error(error, error_capacity, "upload source is null");
        return -2;
    }
    if (offset_bytes != buffer->uploaded_bytes) {
        write_error(error, error_capacity, "uploads must be contiguous and sequential");
        return -3;
    }
    if (bytes > buffer->capacity_bytes - offset_bytes) {
        write_error(error, error_capacity, "upload exceeds device-buffer capacity");
        return -4;
    }
    cudaError_t status = cudaSetDevice(buffer->device_index);
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaSetDevice", error, error_capacity);
    }

    const auto started = std::chrono::steady_clock::now();
    const auto* input = static_cast<const uint8_t*>(source);
    uint64_t copied = 0;
    while (copied < bytes) {
        const uint64_t chunk =
            std::min(buffer->staging_bytes, bytes - copied);
        std::memcpy(
            buffer->pinned_staging,
            input + static_cast<size_t>(copied),
            static_cast<size_t>(chunk)
        );
        status = cudaMemcpyAsync(
            buffer->device_data + static_cast<size_t>(offset_bytes + copied),
            buffer->pinned_staging,
            static_cast<size_t>(chunk),
            cudaMemcpyHostToDevice,
            buffer->stream
        );
        if (status != cudaSuccess) {
            return cuda_failure(status, "cudaMemcpyAsync(weight chunk)", error, error_capacity);
        }
        status = cudaStreamSynchronize(buffer->stream);
        if (status != cudaSuccess) {
            return cuda_failure(status, "cudaStreamSynchronize(upload)", error, error_capacity);
        }
        copied += chunk;
        buffer->upload_calls += 1;
    }
    const auto finished = std::chrono::steady_clock::now();
    buffer->upload_microseconds +=
        std::chrono::duration<double, std::micro>(finished - started).count();
    buffer->uploaded_bytes += bytes;
    write_error(error, error_capacity, "");
    return 0;
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_device_buffer_finish(
    Qwen3TtsDeviceBuffer* buffer,
    Qwen3TtsWeightUploadMetrics* metrics,
    char* error,
    size_t error_capacity
) {
    if (buffer == nullptr || metrics == nullptr) {
        write_error(error, error_capacity, "device buffer or metrics pointer is null");
        return -1;
    }
    if (buffer->uploaded_bytes != buffer->capacity_bytes) {
        write_error(error, error_capacity, "device buffer is not completely uploaded");
        return -2;
    }
    const cudaError_t status = cudaStreamSynchronize(buffer->stream);
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaStreamSynchronize(finish)", error, error_capacity);
    }
    fill_metrics(buffer, metrics);
    write_error(error, error_capacity, "");
    return 0;
}

extern "C" QWEN3_TTS_API int32_t qwen3_tts_device_buffer_read(
    Qwen3TtsDeviceBuffer* buffer,
    uint64_t offset_bytes,
    void* destination,
    uint64_t bytes,
    char* error,
    size_t error_capacity
) {
    if (buffer == nullptr || (bytes != 0 && destination == nullptr)) {
        write_error(error, error_capacity, "device buffer or read destination is null");
        return -1;
    }
    if (offset_bytes > buffer->uploaded_bytes ||
        bytes > buffer->uploaded_bytes - offset_bytes) {
        write_error(error, error_capacity, "read exceeds uploaded device-buffer range");
        return -2;
    }
    cudaError_t status = cudaSetDevice(buffer->device_index);
    if (status != cudaSuccess) {
        return cuda_failure(status, "cudaSetDevice", error, error_capacity);
    }

    auto* output = static_cast<uint8_t*>(destination);
    uint64_t copied = 0;
    while (copied < bytes) {
        const uint64_t chunk =
            std::min(buffer->staging_bytes, bytes - copied);
        status = cudaMemcpyAsync(
            buffer->pinned_staging,
            buffer->device_data + static_cast<size_t>(offset_bytes + copied),
            static_cast<size_t>(chunk),
            cudaMemcpyDeviceToHost,
            buffer->stream
        );
        if (status != cudaSuccess) {
            return cuda_failure(status, "cudaMemcpyAsync(readback chunk)", error, error_capacity);
        }
        status = cudaStreamSynchronize(buffer->stream);
        if (status != cudaSuccess) {
            return cuda_failure(status, "cudaStreamSynchronize(readback)", error, error_capacity);
        }
        std::memcpy(
            output + static_cast<size_t>(copied),
            buffer->pinned_staging,
            static_cast<size_t>(chunk)
        );
        copied += chunk;
    }
    write_error(error, error_capacity, "");
    return 0;
}

extern "C" QWEN3_TTS_API const void* qwen3_tts_device_buffer_data(
    const Qwen3TtsDeviceBuffer* buffer
) {
    return buffer == nullptr ? nullptr : buffer->device_data;
}

extern "C" QWEN3_TTS_API void qwen3_tts_device_buffer_destroy(
    Qwen3TtsDeviceBuffer* buffer
) {
    release_buffer(buffer);
}
