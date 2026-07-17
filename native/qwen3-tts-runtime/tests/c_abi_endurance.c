#define _POSIX_C_SOURCE 200809L

#include "qwen3_tts_runtime.h"

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    ERROR_CAPACITY = 1024,
    WARMUP_REQUESTS = 3,
    MEASURED_REQUESTS = 200,
    NATURAL_EOS_GUARD_FRAMES = 512,
    PCM_SENTINEL = 0x5a5a,
    PCM_CAPACITY = QWEN3_TTS_RUNTIME_MAX_PACKET_FRAMES
        * QWEN3_TTS_RUNTIME_SAMPLES_PER_CODEC_FRAME,
};

typedef struct RequestResult {
    int passed;
    uint32_t index;
    uint32_t finish_reason;
    double ttfa_ms;
    double wall_ms;
    double audio_ms;
    double rtf;
    uint64_t packets;
    uint64_t frames;
    uint64_t samples;
    Qwen3TtsRequestMetricsV1 metrics;
    char error[ERROR_CAPACITY];
} RequestResult;

typedef struct Summary {
    uint32_t completed;
    double total_wall_ms;
    double total_audio_ms;
    double ttfa_p50_ms;
    double ttfa_p95_ms;
    double ttfa_p99_ms;
    double ttfa_max_ms;
    double request_rtf_p50;
    double request_rtf_p95;
    double request_rtf_p99;
    double request_rtf_max;
    double aggregate_rtf;
    uint64_t peak_request_device_bytes;
    uint64_t peak_request_host_bytes;
} Summary;

static const uint8_t TEXT[] =
    "Guten Morgen. Dies ist ein ruhiger Test der nativen Sprachausgabe.";
static const uint8_t INSTRUCTION[] =
    "A calm, relaxed adult male voice. Natural pace, low energy, clear articulation.";

static double monotonic_milliseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0.0;
    }
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
}

static Qwen3TtsEngineConfigV1 engine_config(void) {
    Qwen3TtsEngineConfigV1 config;
    memset(&config, 0, sizeof(config));
    config.struct_size = (uint32_t)sizeof(config);
    config.device_index = 0;
    config.max_concurrent_requests = 1U;
    config.packet_frames = QWEN3_TTS_RUNTIME_MAX_PACKET_FRAMES;
    config.pcm_ring_slots = 3U;
    config.max_text_bytes = 64U * 1024U;
    config.max_instruct_bytes = 16U * 1024U;
    return config;
}

static Qwen3TtsRequestInputV1 request_input(uint64_t seed) {
    Qwen3TtsRequestInputV1 input;
    memset(&input, 0, sizeof(input));
    input.struct_size = (uint32_t)sizeof(input);
    input.language = QWEN3_TTS_LANGUAGE_GERMAN;
    input.text_utf8 = TEXT;
    input.text_bytes = sizeof(TEXT) - 1U;
    input.instruct_utf8 = INSTRUCTION;
    input.instruct_bytes = sizeof(INSTRUCTION) - 1U;
    input.generation.struct_size = (uint32_t)sizeof(input.generation);
    input.generation.max_codec_frames = NATURAL_EOS_GUARD_FRAMES;
    input.generation.seed = seed;
    input.generation.temperature = 0.9F;
    input.generation.top_p = 1.0F;
    input.generation.repetition_penalty = 1.05F;
    input.generation.top_k = 50U;
    input.generation.do_sample = 1U;
    input.generation.predictor_temperature = 0.9F;
    input.generation.predictor_top_p = 1.0F;
    input.generation.predictor_top_k = 50U;
    input.generation.predictor_do_sample = 1U;
    return input;
}

static void set_error(
    RequestResult* result,
    const char* operation,
    int32_t status,
    const char* error
) {
    snprintf(
        result->error,
        sizeof(result->error),
        "%s status %" PRId32 ": %.900s",
        operation,
        status,
        error
    );
}

static RequestResult run_request(Qwen3TtsEngineV1* engine, uint32_t index, uint64_t seed) {
    RequestResult result;
    memset(&result, 0, sizeof(result));
    result.index = index;

    char error[ERROR_CAPACITY] = {0};
    const Qwen3TtsRequestInputV1 input = request_input(seed);
    Qwen3TtsRequestV1* request = NULL;
    const double started = monotonic_milliseconds();
    int32_t status = qwen3_tts_request_start_v1(
        engine,
        &input,
        &request,
        error,
        sizeof(error)
    );
    if (status != QWEN3_TTS_RUNTIME_OK) {
        set_error(&result, "request_start", status, error);
        return result;
    }

    int16_t pcm[PCM_CAPACITY];
    uint64_t expected_sequence = 0U;
    uint64_t expected_frame = 0U;
    uint64_t expected_sample = 0U;
    int saw_final = 0;
    for (;;) {
        for (size_t index = 0; index < PCM_CAPACITY; ++index) {
            pcm[index] = PCM_SENTINEL;
        }
        Qwen3TtsAudioPacketV1 packet;
        memset(&packet, 0, sizeof(packet));
        status = qwen3_tts_request_poll_v1(
            request,
            30000U,
            pcm,
            PCM_CAPACITY,
            &packet,
            error,
            sizeof(error)
        );
        if (status == QWEN3_TTS_RUNTIME_WOULD_BLOCK) {
            continue;
        }
        if (status == QWEN3_TTS_RUNTIME_END_OF_STREAM) {
            break;
        }
        if (status != QWEN3_TTS_RUNTIME_OK) {
            set_error(&result, "request_poll", status, error);
            qwen3_tts_request_destroy_v1(request, error, sizeof(error));
            return result;
        }
        if (saw_final) {
            snprintf(result.error, sizeof(result.error), "packet arrived after final packet");
            qwen3_tts_request_destroy_v1(request, error, sizeof(error));
            return result;
        }
        if (expected_sequence == 0U) {
            result.ttfa_ms = monotonic_milliseconds() - started;
            if (packet.codec_frames != 1U) {
                snprintf(result.error, sizeof(result.error), "first packet was not one frame");
                qwen3_tts_request_destroy_v1(request, error, sizeof(error));
                return result;
            }
        }
        if (packet.sequence != expected_sequence
            || packet.first_codec_frame != expected_frame
            || packet.first_sample != expected_sample
            || packet.codec_frames == 0U
            || packet.codec_frames > QWEN3_TTS_RUNTIME_MAX_PACKET_FRAMES
            || packet.sample_count
                != packet.codec_frames * QWEN3_TTS_RUNTIME_SAMPLES_PER_CODEC_FRAME
            || packet.sample_rate != QWEN3_TTS_RUNTIME_SAMPLE_RATE
            || packet.channels != QWEN3_TTS_RUNTIME_CHANNELS) {
            snprintf(result.error, sizeof(result.error), "packet continuity invariant failed");
            qwen3_tts_request_destroy_v1(request, error, sizeof(error));
            return result;
        }
        for (size_t index = packet.sample_count; index < PCM_CAPACITY; ++index) {
            if (pcm[index] != PCM_SENTINEL) {
                snprintf(result.error, sizeof(result.error), "PCM tail overwrite detected");
                qwen3_tts_request_destroy_v1(request, error, sizeof(error));
                return result;
            }
        }
        ++expected_sequence;
        expected_frame += packet.codec_frames;
        expected_sample += packet.sample_count;
        if (packet.is_final != 0U) {
            saw_final = 1;
        }
    }

    status = qwen3_tts_request_finish_reason_v1(
        request,
        &result.finish_reason,
        error,
        sizeof(error)
    );
    if (status != QWEN3_TTS_RUNTIME_OK) {
        set_error(&result, "request_finish_reason", status, error);
        qwen3_tts_request_destroy_v1(request, error, sizeof(error));
        return result;
    }
    status = qwen3_tts_request_metrics_v1(request, &result.metrics, error, sizeof(error));
    if (status != QWEN3_TTS_RUNTIME_OK) {
        set_error(&result, "request_metrics", status, error);
        qwen3_tts_request_destroy_v1(request, error, sizeof(error));
        return result;
    }
    status = qwen3_tts_request_destroy_v1(request, error, sizeof(error));
    if (status != QWEN3_TTS_RUNTIME_OK) {
        set_error(&result, "request_destroy", status, error);
        return result;
    }

    result.wall_ms = monotonic_milliseconds() - started;
    result.packets = expected_sequence;
    result.frames = expected_frame;
    result.samples = expected_sample;
    result.audio_ms = (double)expected_sample * 1000.0
        / (double)QWEN3_TTS_RUNTIME_SAMPLE_RATE;
    result.rtf = result.wall_ms / result.audio_ms;
    if (!saw_final
        || result.finish_reason != QWEN3_TTS_FINISH_REASON_CODEC_EOS
        || expected_frame == 0U
        || expected_frame >= NATURAL_EOS_GUARD_FRAMES
        || result.metrics.generated_codec_frames != expected_frame
        || result.metrics.emitted_samples != expected_sample
        || result.metrics.emitted_packets != expected_sequence
        || result.metrics.first_audio_microseconds == 0U) {
        snprintf(
            result.error,
            sizeof(result.error),
            "natural EOS, terminal stream, or metrics invariant failed"
        );
        return result;
    }
    result.passed = 1;
    return result;
}

static int compare_double(const void* left, const void* right) {
    const double a = *(const double*)left;
    const double b = *(const double*)right;
    return (a > b) - (a < b);
}

static double percentile(double* values, size_t count, size_t numerator) {
    qsort(values, count, sizeof(*values), compare_double);
    size_t rank = (numerator * count + 99U) / 100U;
    if (rank == 0U) {
        rank = 1U;
    }
    return values[rank - 1U];
}

static uint64_t peak_rss_kib(void) {
    FILE* status = fopen("/proc/self/status", "r");
    if (status == NULL) {
        return 0U;
    }
    char line[256];
    uint64_t value = 0U;
    while (fgets(line, sizeof(line), status) != NULL) {
        if (sscanf(line, "VmHWM: %" SCNu64 " kB", &value) == 1) {
            break;
        }
    }
    fclose(status);
    return value;
}

static Summary summarize(const RequestResult* results) {
    double ttfa[MEASURED_REQUESTS];
    double rtf[MEASURED_REQUESTS];
    Summary summary;
    memset(&summary, 0, sizeof(summary));
    summary.completed = MEASURED_REQUESTS;
    for (uint32_t index = 0; index < MEASURED_REQUESTS; ++index) {
        ttfa[index] = results[index].ttfa_ms;
        rtf[index] = results[index].rtf;
        summary.total_wall_ms += results[index].wall_ms;
        summary.total_audio_ms += results[index].audio_ms;
        if (results[index].metrics.peak_request_device_bytes
            > summary.peak_request_device_bytes) {
            summary.peak_request_device_bytes = results[index].metrics.peak_request_device_bytes;
        }
        if (results[index].metrics.peak_request_host_bytes > summary.peak_request_host_bytes) {
            summary.peak_request_host_bytes = results[index].metrics.peak_request_host_bytes;
        }
    }
    double ttfa_p50[MEASURED_REQUESTS];
    double ttfa_p95[MEASURED_REQUESTS];
    double ttfa_p99[MEASURED_REQUESTS];
    double rtf_p50[MEASURED_REQUESTS];
    double rtf_p95[MEASURED_REQUESTS];
    double rtf_p99[MEASURED_REQUESTS];
    memcpy(ttfa_p50, ttfa, sizeof(ttfa));
    memcpy(ttfa_p95, ttfa, sizeof(ttfa));
    memcpy(ttfa_p99, ttfa, sizeof(ttfa));
    memcpy(rtf_p50, rtf, sizeof(rtf));
    memcpy(rtf_p95, rtf, sizeof(rtf));
    memcpy(rtf_p99, rtf, sizeof(rtf));
    summary.ttfa_p50_ms = percentile(ttfa_p50, MEASURED_REQUESTS, 50U);
    summary.ttfa_p95_ms = percentile(ttfa_p95, MEASURED_REQUESTS, 95U);
    summary.ttfa_p99_ms = percentile(ttfa_p99, MEASURED_REQUESTS, 99U);
    summary.request_rtf_p50 = percentile(rtf_p50, MEASURED_REQUESTS, 50U);
    summary.request_rtf_p95 = percentile(rtf_p95, MEASURED_REQUESTS, 95U);
    summary.request_rtf_p99 = percentile(rtf_p99, MEASURED_REQUESTS, 99U);
    summary.ttfa_max_ms = ttfa[0];
    summary.request_rtf_max = rtf[0];
    for (uint32_t index = 1; index < MEASURED_REQUESTS; ++index) {
        if (ttfa[index] > summary.ttfa_max_ms) {
            summary.ttfa_max_ms = ttfa[index];
        }
        if (rtf[index] > summary.request_rtf_max) {
            summary.request_rtf_max = rtf[index];
        }
    }
    summary.aggregate_rtf = summary.total_wall_ms / summary.total_audio_ms;
    return summary;
}

static void write_report(
    const char* report_path,
    double model_load_ms,
    uint64_t rss_kib,
    const Summary* summary,
    const RequestResult* results
) {
    FILE* report = fopen(report_path, "w");
    if (report == NULL) {
        perror("open report");
        exit(EXIT_FAILURE);
    }
    fprintf(
        report,
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"operation\": \"qwen3-tts-public-c-abi-natural-eos-endurance\",\n"
        "  \"qualifying_run\": true,\n"
        "  \"model\": \"Qwen3-TTS-12Hz-1.7B-VoiceDesign\",\n"
        "  \"concurrency\": 1,\n"
        "  \"model_load_ms_excluded\": %.6f,\n"
        "  \"warmup_requests\": %u,\n"
        "  \"measured_requests\": %u,\n"
        "  \"completed_requests\": %u,\n"
        "  \"failed_requests\": 0,\n"
        "  \"codec_eos_requests\": %u,\n"
        "  \"max_codec_frames_requests\": 0,\n"
        "  \"natural_eos_guard_frames\": %u,\n"
        "  \"peak_process_rss_kib\": %" PRIu64 ",\n"
        "  \"all_packet_metric_and_finish_reason_invariants_passed\": true,\n"
        "  \"total_wall_ms\": %.6f,\n"
        "  \"total_audio_ms\": %.6f,\n"
        "  \"ttfa_ms\": {\"p50\": %.6f, \"p95\": %.6f, \"p99\": %.6f, \"max\": %.6f},\n"
        "  \"request_rtf\": {\"p50\": %.6f, \"p95\": %.6f, \"p99\": %.6f, \"max\": %.6f},\n"
        "  \"aggregate_rtf\": %.6f,\n"
        "  \"peak_request_device_bytes\": %" PRIu64 ",\n"
        "  \"peak_request_host_bytes\": %" PRIu64 ",\n"
        "  \"requests\": [\n",
        model_load_ms,
        WARMUP_REQUESTS,
        MEASURED_REQUESTS,
        summary->completed,
        summary->completed,
        NATURAL_EOS_GUARD_FRAMES,
        rss_kib,
        summary->total_wall_ms,
        summary->total_audio_ms,
        summary->ttfa_p50_ms,
        summary->ttfa_p95_ms,
        summary->ttfa_p99_ms,
        summary->ttfa_max_ms,
        summary->request_rtf_p50,
        summary->request_rtf_p95,
        summary->request_rtf_p99,
        summary->request_rtf_max,
        summary->aggregate_rtf,
        summary->peak_request_device_bytes,
        summary->peak_request_host_bytes
    );
    for (uint32_t index = 0; index < MEASURED_REQUESTS; ++index) {
        fprintf(
            report,
            "    {\"index\": %u, \"seed\": %u, \"finish_reason\": \"codec_eos\", "
            "\"ttfa_ms\": %.6f, \"wall_ms\": %.6f, \"audio_ms\": %.6f, "
            "\"rtf\": %.6f, \"packets\": %" PRIu64 ", \"codec_frames\": %" PRIu64 ", "
            "\"samples\": %" PRIu64 ", \"metric_first_audio_us\": %" PRIu64 ", "
            "\"metric_prefill_us\": %" PRIu64 ", \"talker_gpu_us\": %.6f, "
            "\"codec_gpu_us\": %.6f}%s\n",
            index,
            42000U + index,
            results[index].ttfa_ms,
            results[index].wall_ms,
            results[index].audio_ms,
            results[index].rtf,
            results[index].packets,
            results[index].frames,
            results[index].samples,
            results[index].metrics.first_audio_microseconds,
            results[index].metrics.prefill_microseconds,
            results[index].metrics.talker_gpu_microseconds,
            results[index].metrics.codec_gpu_microseconds,
            index + 1U == MEASURED_REQUESTS ? "" : ","
        );
    }
    fprintf(report, "  ]\n}\n");
    if (fclose(report) != 0) {
        perror("close report");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_ROOT REPORT_JSON\n", argv[0]);
        return EXIT_FAILURE;
    }
    char error[ERROR_CAPACITY] = {0};
    Qwen3TtsEngineConfigV1 config = engine_config();
    Qwen3TtsEngineV1* engine = NULL;
    const double load_started = monotonic_milliseconds();
    const int32_t status = qwen3_tts_engine_create_v1(
        (const uint8_t*)argv[1],
        strlen(argv[1]),
        &config,
        &engine,
        error,
        sizeof(error)
    );
    if (status != QWEN3_TTS_RUNTIME_OK) {
        fprintf(stderr, "engine_create status %" PRId32 ": %s\n", status, error);
        return EXIT_FAILURE;
    }
    const double model_load_ms = monotonic_milliseconds() - load_started;

    fprintf(stderr, "warming %u full natural-EOS requests at B1\n", WARMUP_REQUESTS);
    for (uint32_t index = 0; index < WARMUP_REQUESTS; ++index) {
        const RequestResult warm = run_request(engine, index, 41000U + index);
        if (!warm.passed) {
            fprintf(stderr, "warm request %u failed: %s\n", index, warm.error);
            return EXIT_FAILURE;
        }
    }

    RequestResult* results = calloc(MEASURED_REQUESTS, sizeof(*results));
    if (results == NULL) {
        fprintf(stderr, "result allocation failed\n");
        return EXIT_FAILURE;
    }
    for (uint32_t index = 0; index < MEASURED_REQUESTS; ++index) {
        results[index] = run_request(engine, index, 42000U + index);
        if (!results[index].passed) {
            fprintf(stderr, "request %u failed: %s\n", index, results[index].error);
            free(results);
            return EXIT_FAILURE;
        }
        if ((index + 1U) % 10U == 0U || index + 1U == MEASURED_REQUESTS) {
            fprintf(
                stderr,
                "B1 natural EOS: %u/%u requests complete\n",
                index + 1U,
                MEASURED_REQUESTS
            );
            fflush(stderr);
        }
    }

    const Summary summary = summarize(results);
    write_report(argv[2], model_load_ms, peak_rss_kib(), &summary, results);
    const int32_t destroy_status = qwen3_tts_engine_destroy_v1(
        engine,
        error,
        sizeof(error)
    );
    if (destroy_status != QWEN3_TTS_RUNTIME_OK) {
        fprintf(stderr, "engine_destroy status %" PRId32 ": %s\n", destroy_status, error);
        free(results);
        return EXIT_FAILURE;
    }
    printf(
        "%u/%u full requests reached natural Codec EOS; TTFA p95 %.3f ms, "
        "request RTF p50 %.3f, aggregate RTF %.3f\n",
        summary.completed,
        MEASURED_REQUESTS,
        summary.ttfa_p95_ms,
        summary.request_rtf_p50,
        summary.aggregate_rtf
    );
    free(results);
    return EXIT_SUCCESS;
}
