#define _POSIX_C_SOURCE 200809L

#include "qwen3_tts_runtime.h"

#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    ERROR_CAPACITY = 1024,
    MAX_BATCH = QWEN3_TTS_RUNTIME_MAX_CONCURRENT_REQUESTS,
    WARM_REQUESTS = 24,
    MEASURED_REQUESTS = 200,
    BENCHMARK_FRAMES = 4,
    PCM_CAPACITY = QWEN3_TTS_RUNTIME_MAX_PACKET_FRAMES
        * QWEN3_TTS_RUNTIME_SAMPLES_PER_CODEC_FRAME,
};

typedef struct RequestResult {
    int passed;
    uint32_t index;
    uint32_t concurrency;
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

typedef struct StartBarrier {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    uint32_t participants;
    uint32_t waiting;
    uint32_t generation;
} StartBarrier;

typedef struct ThreadInput {
    Qwen3TtsEngineV1* engine;
    StartBarrier* barrier;
    RequestResult* result;
    uint32_t index;
    uint32_t concurrency;
} ThreadInput;

typedef struct LevelSummary {
    uint32_t concurrency;
    uint32_t completed;
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
} LevelSummary;

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

static int start_barrier_init(StartBarrier* barrier, uint32_t participants) {
    memset(barrier, 0, sizeof(*barrier));
    barrier->participants = participants;
    if (pthread_mutex_init(&barrier->mutex, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&barrier->condition, NULL) != 0) {
        pthread_mutex_destroy(&barrier->mutex);
        return -1;
    }
    return 0;
}

static int start_barrier_wait(StartBarrier* barrier) {
    if (pthread_mutex_lock(&barrier->mutex) != 0) {
        return -1;
    }
    const uint32_t generation = barrier->generation;
    ++barrier->waiting;
    if (barrier->waiting == barrier->participants) {
        barrier->waiting = 0U;
        ++barrier->generation;
        const int broadcast_status = pthread_cond_broadcast(&barrier->condition);
        const int unlock_status = pthread_mutex_unlock(&barrier->mutex);
        return broadcast_status == 0 && unlock_status == 0 ? 0 : -1;
    }
    while (generation == barrier->generation) {
        if (pthread_cond_wait(&barrier->condition, &barrier->mutex) != 0) {
            pthread_mutex_unlock(&barrier->mutex);
            return -1;
        }
    }
    return pthread_mutex_unlock(&barrier->mutex) == 0 ? 0 : -1;
}

static void start_barrier_destroy(StartBarrier* barrier) {
    pthread_cond_destroy(&barrier->condition);
    pthread_mutex_destroy(&barrier->mutex);
}

static Qwen3TtsEngineConfigV1 engine_config(void) {
    Qwen3TtsEngineConfigV1 config;
    memset(&config, 0, sizeof(config));
    config.struct_size = (uint32_t)sizeof(config);
    config.device_index = 0;
    config.max_concurrent_requests = MAX_BATCH;
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
    input.generation.max_codec_frames = BENCHMARK_FRAMES;
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

static void set_error(RequestResult* result, const char* operation, int32_t status, const char* error) {
    snprintf(
        result->error,
        sizeof(result->error),
        "%s status %" PRId32 ": %.900s",
        operation,
        status,
        error
    );
}

static void* run_request(void* opaque) {
    ThreadInput* thread_input = (ThreadInput*)opaque;
    RequestResult* result = thread_input->result;
    memset(result, 0, sizeof(*result));
    result->index = thread_input->index;
    result->concurrency = thread_input->concurrency;

    if (start_barrier_wait(thread_input->barrier) != 0) {
        snprintf(result->error, sizeof(result->error), "start barrier wait failed");
        return NULL;
    }

    char error[ERROR_CAPACITY] = {0};
    const Qwen3TtsRequestInputV1 input = request_input(1000U + thread_input->index);
    Qwen3TtsRequestV1* request = NULL;
    const double started = monotonic_milliseconds();
    int32_t status = qwen3_tts_request_start_v1(
        thread_input->engine,
        &input,
        &request,
        error,
        sizeof(error)
    );
    if (status != QWEN3_TTS_RUNTIME_OK) {
        set_error(result, "request_start", status, error);
        return NULL;
    }

    int16_t pcm[PCM_CAPACITY];
    uint64_t expected_sequence = 0U;
    uint64_t expected_frame = 0U;
    uint64_t expected_sample = 0U;
    int saw_final = 0;
    for (;;) {
        for (size_t index = 0; index < PCM_CAPACITY; ++index) {
            pcm[index] = INT16_MIN;
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
            set_error(result, "request_poll", status, error);
            qwen3_tts_request_destroy_v1(request, error, sizeof(error));
            return NULL;
        }
        if (expected_sequence == 0U) {
            result->ttfa_ms = monotonic_milliseconds() - started;
            if (packet.codec_frames != 1U) {
                snprintf(result->error, sizeof(result->error), "first packet was not one frame");
                qwen3_tts_request_destroy_v1(request, error, sizeof(error));
                return NULL;
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
            snprintf(result->error, sizeof(result->error), "packet continuity invariant failed");
            qwen3_tts_request_destroy_v1(request, error, sizeof(error));
            return NULL;
        }
        for (size_t index = packet.sample_count; index < PCM_CAPACITY; ++index) {
            if (pcm[index] != INT16_MIN) {
                snprintf(result->error, sizeof(result->error), "PCM tail overwrite detected");
                qwen3_tts_request_destroy_v1(request, error, sizeof(error));
                return NULL;
            }
        }
        ++expected_sequence;
        expected_frame += packet.codec_frames;
        expected_sample += packet.sample_count;
        saw_final = packet.is_final != 0U;
    }

    status = qwen3_tts_request_metrics_v1(request, &result->metrics, error, sizeof(error));
    if (status != QWEN3_TTS_RUNTIME_OK) {
        set_error(result, "request_metrics", status, error);
        qwen3_tts_request_destroy_v1(request, error, sizeof(error));
        return NULL;
    }
    status = qwen3_tts_request_destroy_v1(request, error, sizeof(error));
    if (status != QWEN3_TTS_RUNTIME_OK) {
        set_error(result, "request_destroy", status, error);
        return NULL;
    }

    result->wall_ms = monotonic_milliseconds() - started;
    result->packets = expected_sequence;
    result->frames = expected_frame;
    result->samples = expected_sample;
    result->audio_ms = (double)expected_sample * 1000.0
        / (double)QWEN3_TTS_RUNTIME_SAMPLE_RATE;
    result->rtf = result->wall_ms / result->audio_ms;
    if (!saw_final
        || expected_frame != BENCHMARK_FRAMES
        || result->metrics.generated_codec_frames != expected_frame
        || result->metrics.emitted_samples != expected_sample
        || result->metrics.emitted_packets != expected_sequence
        || result->metrics.first_audio_microseconds == 0U) {
        snprintf(result->error, sizeof(result->error), "terminal stream or metrics invariant failed");
        return NULL;
    }
    result->passed = 1;
    return NULL;
}

static double run_group(
    Qwen3TtsEngineV1* engine,
    uint32_t concurrency,
    uint32_t first_index,
    RequestResult* results
) {
    StartBarrier barrier;
    if (start_barrier_init(&barrier, concurrency + 1U) != 0) {
        fprintf(stderr, "start barrier initialization failed\n");
        exit(EXIT_FAILURE);
    }
    pthread_t threads[MAX_BATCH];
    ThreadInput inputs[MAX_BATCH];
    for (uint32_t slot = 0; slot < concurrency; ++slot) {
        inputs[slot].engine = engine;
        inputs[slot].barrier = &barrier;
        inputs[slot].result = &results[slot];
        inputs[slot].index = first_index + slot;
        inputs[slot].concurrency = concurrency;
        if (pthread_create(&threads[slot], NULL, run_request, &inputs[slot]) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            exit(EXIT_FAILURE);
        }
    }
    const double group_started = monotonic_milliseconds();
    if (start_barrier_wait(&barrier) != 0) {
        fprintf(stderr, "main start barrier wait failed\n");
        exit(EXIT_FAILURE);
    }
    for (uint32_t slot = 0; slot < concurrency; ++slot) {
        if (pthread_join(threads[slot], NULL) != 0) {
            fprintf(stderr, "pthread_join failed\n");
            exit(EXIT_FAILURE);
        }
    }
    const double group_wall_ms = monotonic_milliseconds() - group_started;
    start_barrier_destroy(&barrier);
    return group_wall_ms;
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

static LevelSummary run_level(
    Qwen3TtsEngineV1* engine,
    uint32_t concurrency,
    RequestResult* results
) {
    double total_group_wall_ms = 0.0;
    uint32_t completed = 0U;
    while (completed < MEASURED_REQUESTS) {
        uint32_t group = concurrency;
        if (group > MEASURED_REQUESTS - completed) {
            group = MEASURED_REQUESTS - completed;
        }
        total_group_wall_ms += run_group(
            engine,
            group,
            completed,
            &results[completed]
        );
        completed += group;
        if (completed % 25U == 0U || completed == MEASURED_REQUESTS) {
            fprintf(stderr, "B%u: %u/%u requests complete\n", concurrency, completed, MEASURED_REQUESTS);
            fflush(stderr);
        }
    }

    double ttfa[MEASURED_REQUESTS];
    double rtf[MEASURED_REQUESTS];
    uint64_t peak_device = 0U;
    uint64_t peak_host = 0U;
    double total_audio_ms = 0.0;
    for (uint32_t index = 0; index < MEASURED_REQUESTS; ++index) {
        if (!results[index].passed) {
            fprintf(stderr, "B%u request %u failed: %s\n", concurrency, index, results[index].error);
            exit(EXIT_FAILURE);
        }
        ttfa[index] = results[index].ttfa_ms;
        rtf[index] = results[index].rtf;
        total_audio_ms += results[index].audio_ms;
        if (results[index].metrics.peak_request_device_bytes > peak_device) {
            peak_device = results[index].metrics.peak_request_device_bytes;
        }
        if (results[index].metrics.peak_request_host_bytes > peak_host) {
            peak_host = results[index].metrics.peak_request_host_bytes;
        }
    }
    double ttfa_p50_values[MEASURED_REQUESTS];
    double ttfa_p95_values[MEASURED_REQUESTS];
    double ttfa_p99_values[MEASURED_REQUESTS];
    double rtf_p50_values[MEASURED_REQUESTS];
    double rtf_p95_values[MEASURED_REQUESTS];
    double rtf_p99_values[MEASURED_REQUESTS];
    memcpy(ttfa_p50_values, ttfa, sizeof(ttfa));
    memcpy(ttfa_p95_values, ttfa, sizeof(ttfa));
    memcpy(ttfa_p99_values, ttfa, sizeof(ttfa));
    memcpy(rtf_p50_values, rtf, sizeof(rtf));
    memcpy(rtf_p95_values, rtf, sizeof(rtf));
    memcpy(rtf_p99_values, rtf, sizeof(rtf));
    double ttfa_max = ttfa[0];
    double rtf_max = rtf[0];
    for (uint32_t index = 1; index < MEASURED_REQUESTS; ++index) {
        if (ttfa[index] > ttfa_max) {
            ttfa_max = ttfa[index];
        }
        if (rtf[index] > rtf_max) {
            rtf_max = rtf[index];
        }
    }
    LevelSummary summary = {
        .concurrency = concurrency,
        .completed = MEASURED_REQUESTS,
        .ttfa_p50_ms = percentile(ttfa_p50_values, MEASURED_REQUESTS, 50U),
        .ttfa_p95_ms = percentile(ttfa_p95_values, MEASURED_REQUESTS, 95U),
        .ttfa_p99_ms = percentile(ttfa_p99_values, MEASURED_REQUESTS, 99U),
        .ttfa_max_ms = ttfa_max,
        .request_rtf_p50 = percentile(rtf_p50_values, MEASURED_REQUESTS, 50U),
        .request_rtf_p95 = percentile(rtf_p95_values, MEASURED_REQUESTS, 95U),
        .request_rtf_p99 = percentile(rtf_p99_values, MEASURED_REQUESTS, 99U),
        .request_rtf_max = rtf_max,
        .aggregate_rtf = total_group_wall_ms / total_audio_ms,
        .peak_request_device_bytes = peak_device,
        .peak_request_host_bytes = peak_host,
    };
    return summary;
}

static void write_level(
    FILE* report,
    const LevelSummary* summary,
    const RequestResult* results,
    int trailing_comma
) {
    fprintf(
        report,
        "    {\n"
        "      \"concurrency\": %u,\n"
        "      \"measured_requests\": %u,\n"
        "      \"completed_requests\": %u,\n"
        "      \"ttfa_ms\": {\"p50\": %.6f, \"p95\": %.6f, \"p99\": %.6f, \"max\": %.6f},\n"
        "      \"request_rtf\": {\"p50\": %.6f, \"p95\": %.6f, \"p99\": %.6f, \"max\": %.6f},\n"
        "      \"aggregate_rtf\": %.6f,\n"
        "      \"peak_request_device_bytes\": %" PRIu64 ",\n"
        "      \"peak_request_host_bytes\": %" PRIu64 ",\n"
        "      \"requests\": [\n",
        summary->concurrency,
        MEASURED_REQUESTS,
        summary->completed,
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
            "        {\"index\": %u, \"ttfa_ms\": %.6f, \"wall_ms\": %.6f, "
            "\"audio_ms\": %.6f, \"rtf\": %.6f, \"packets\": %" PRIu64 ", "
            "\"codec_frames\": %" PRIu64 ", \"samples\": %" PRIu64 ", "
            "\"metric_first_audio_us\": %" PRIu64 "}%s\n",
            index,
            results[index].ttfa_ms,
            results[index].wall_ms,
            results[index].audio_ms,
            results[index].rtf,
            results[index].packets,
            results[index].frames,
            results[index].samples,
            results[index].metrics.first_audio_microseconds,
            index + 1U == MEASURED_REQUESTS ? "" : ","
        );
    }
    fprintf(report, "      ]\n    }%s\n", trailing_comma ? "," : "");
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_ROOT REPORT_JSON\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* model_root = argv[1];
    const char* report_path = argv[2];
    char error[ERROR_CAPACITY] = {0};
    Qwen3TtsEngineConfigV1 config = engine_config();
    Qwen3TtsEngineV1* engine = NULL;
    const double load_started = monotonic_milliseconds();
    const int32_t status = qwen3_tts_engine_create_v1(
        (const uint8_t*)model_root,
        strlen(model_root),
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

    fprintf(stderr, "warming %u requests at B6\n", WARM_REQUESTS);
    RequestResult warm_results[MAX_BATCH];
    for (uint32_t completed = 0; completed < WARM_REQUESTS; completed += MAX_BATCH) {
        run_group(engine, MAX_BATCH, 100000U + completed, warm_results);
        for (uint32_t slot = 0; slot < MAX_BATCH; ++slot) {
            if (!warm_results[slot].passed) {
                fprintf(stderr, "warm request failed: %s\n", warm_results[slot].error);
                return EXIT_FAILURE;
            }
        }
    }

    RequestResult* b1 = calloc(MEASURED_REQUESTS, sizeof(*b1));
    RequestResult* b3 = calloc(MEASURED_REQUESTS, sizeof(*b3));
    RequestResult* b6 = calloc(MEASURED_REQUESTS, sizeof(*b6));
    if (b1 == NULL || b3 == NULL || b6 == NULL) {
        fprintf(stderr, "result allocation failed\n");
        return EXIT_FAILURE;
    }
    const LevelSummary b1_summary = run_level(engine, 1U, b1);
    const LevelSummary b3_summary = run_level(engine, 3U, b3);
    const LevelSummary b6_summary = run_level(engine, 6U, b6);
    const uint64_t rss_kib = peak_rss_kib();

    FILE* report = fopen(report_path, "w");
    if (report == NULL) {
        perror("open report");
        return EXIT_FAILURE;
    }
    fprintf(
        report,
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"operation\": \"qwen3-tts-public-c-abi-benchmark\",\n"
        "  \"qualifying_run\": true,\n"
        "  \"model\": \"Qwen3-TTS-12Hz-1.7B-VoiceDesign\",\n"
        "  \"model_load_ms_excluded\": %.6f,\n"
        "  \"warmup_requests\": %u,\n"
        "  \"requests_per_concurrency\": %u,\n"
        "  \"codec_frames_per_request\": %u,\n"
        "  \"audio_ms_per_request\": 320.0,\n"
        "  \"peak_process_rss_kib\": %" PRIu64 ",\n"
        "  \"all_packet_and_metric_invariants_passed\": true,\n"
        "  \"levels\": [\n",
        model_load_ms,
        WARM_REQUESTS,
        MEASURED_REQUESTS,
        BENCHMARK_FRAMES,
        rss_kib
    );
    write_level(report, &b1_summary, b1, 1);
    write_level(report, &b3_summary, b3, 1);
    write_level(report, &b6_summary, b6, 0);
    fprintf(report, "  ]\n}\n");
    if (fclose(report) != 0) {
        perror("close report");
        return EXIT_FAILURE;
    }

    const int32_t destroy_status = qwen3_tts_engine_destroy_v1(engine, error, sizeof(error));
    if (destroy_status != QWEN3_TTS_RUNTIME_OK) {
        fprintf(stderr, "engine_destroy status %" PRId32 ": %s\n", destroy_status, error);
        return EXIT_FAILURE;
    }
    printf(
        "B1 TTFA p95 %.3f ms, request RTF p50 %.3f, aggregate RTF %.3f\n"
        "B3 TTFA p95 %.3f ms, request RTF p50 %.3f, aggregate RTF %.3f\n"
        "B6 TTFA p95 %.3f ms, request RTF p50 %.3f, aggregate RTF %.3f\n",
        b1_summary.ttfa_p95_ms,
        b1_summary.request_rtf_p50,
        b1_summary.aggregate_rtf,
        b3_summary.ttfa_p95_ms,
        b3_summary.request_rtf_p50,
        b3_summary.aggregate_rtf,
        b6_summary.ttfa_p95_ms,
        b6_summary.request_rtf_p50,
        b6_summary.aggregate_rtf
    );
    free(b1);
    free(b3);
    free(b6);
    return EXIT_SUCCESS;
}
