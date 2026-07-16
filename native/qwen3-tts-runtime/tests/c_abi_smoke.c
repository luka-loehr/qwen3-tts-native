#define _POSIX_C_SOURCE 200809L

#include "qwen3_tts_runtime.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Static_assert(sizeof(Qwen3TtsEngineConfigV1) == 96, "engine config ABI drift");
_Static_assert(sizeof(Qwen3TtsGenerationConfigV1) == 120, "generation config ABI drift");
_Static_assert(sizeof(Qwen3TtsRequestInputV1) == 160, "request input ABI drift");
_Static_assert(sizeof(Qwen3TtsAudioPacketV1) == 72, "audio packet ABI drift");
_Static_assert(sizeof(Qwen3TtsRequestMetricsV1) == 96, "metrics ABI drift");

enum {
    ERROR_CAPACITY = 1024,
    PCM_CAPACITY = QWEN3_TTS_RUNTIME_MAX_PACKET_FRAMES
        * QWEN3_TTS_RUNTIME_SAMPLES_PER_CODEC_FRAME,
};

static double monotonic_milliseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0.0;
    }
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
}

static void fail_status(const char* operation, int32_t status, const char* error) {
    fprintf(stderr, "%s failed with status %" PRId32 ": %s\n", operation, status, error);
    exit(EXIT_FAILURE);
}

static void require_status(
    const char* operation,
    int32_t actual,
    int32_t expected,
    const char* error
) {
    if (actual != expected) {
        fprintf(
            stderr,
            "%s returned status %" PRId32 ", expected %" PRId32 ": %s\n",
            operation,
            actual,
            expected,
            error
        );
        exit(EXIT_FAILURE);
    }
}

static void write_u16_le(FILE* file, uint16_t value) {
    const uint8_t bytes[2] = {
        (uint8_t)(value & 0xffU),
        (uint8_t)((value >> 8U) & 0xffU),
    };
    if (fwrite(bytes, sizeof(bytes), 1, file) != 1) {
        perror("write WAV u16");
        exit(EXIT_FAILURE);
    }
}

static void write_u32_le(FILE* file, uint32_t value) {
    const uint8_t bytes[4] = {
        (uint8_t)(value & 0xffU),
        (uint8_t)((value >> 8U) & 0xffU),
        (uint8_t)((value >> 16U) & 0xffU),
        (uint8_t)((value >> 24U) & 0xffU),
    };
    if (fwrite(bytes, sizeof(bytes), 1, file) != 1) {
        perror("write WAV u32");
        exit(EXIT_FAILURE);
    }
}

static void write_wav_header(FILE* file, uint32_t samples) {
    const uint32_t data_bytes = samples * (uint32_t)sizeof(int16_t);
    const uint32_t byte_rate = QWEN3_TTS_RUNTIME_SAMPLE_RATE * (uint32_t)sizeof(int16_t);
    rewind(file);
    if (fwrite("RIFF", 4, 1, file) != 1) {
        perror("write WAV RIFF");
        exit(EXIT_FAILURE);
    }
    write_u32_le(file, 36U + data_bytes);
    if (fwrite("WAVEfmt ", 8, 1, file) != 1) {
        perror("write WAV format");
        exit(EXIT_FAILURE);
    }
    write_u32_le(file, 16U);
    write_u16_le(file, 1U);
    write_u16_le(file, QWEN3_TTS_RUNTIME_CHANNELS);
    write_u32_le(file, QWEN3_TTS_RUNTIME_SAMPLE_RATE);
    write_u32_le(file, byte_rate);
    write_u16_le(file, (uint16_t)sizeof(int16_t));
    write_u16_le(file, 16U);
    if (fwrite("data", 4, 1, file) != 1) {
        perror("write WAV data");
        exit(EXIT_FAILURE);
    }
    write_u32_le(file, data_bytes);
}

static Qwen3TtsEngineConfigV1 engine_config(void) {
    Qwen3TtsEngineConfigV1 config;
    memset(&config, 0, sizeof(config));
    config.struct_size = (uint32_t)sizeof(config);
    config.device_index = 0;
    config.max_concurrent_requests = QWEN3_TTS_RUNTIME_MAX_CONCURRENT_REQUESTS;
    config.packet_frames = QWEN3_TTS_RUNTIME_MAX_PACKET_FRAMES;
    config.pcm_ring_slots = 3U;
    config.max_text_bytes = 64U * 1024U;
    config.max_instruct_bytes = 16U * 1024U;
    return config;
}

static Qwen3TtsGenerationConfigV1 generation_config(uint32_t max_frames, uint64_t seed) {
    Qwen3TtsGenerationConfigV1 generation;
    memset(&generation, 0, sizeof(generation));
    generation.struct_size = (uint32_t)sizeof(generation);
    generation.max_codec_frames = max_frames;
    generation.seed = seed;
    generation.temperature = 0.9F;
    generation.top_p = 1.0F;
    generation.repetition_penalty = 1.05F;
    generation.top_k = 50U;
    generation.do_sample = 1U;
    generation.predictor_temperature = 0.9F;
    generation.predictor_top_p = 1.0F;
    generation.predictor_top_k = 50U;
    generation.predictor_do_sample = 1U;
    return generation;
}

static Qwen3TtsRequestInputV1 request_input(
    const uint8_t* text,
    size_t text_bytes,
    const uint8_t* instruction,
    size_t instruction_bytes,
    uint32_t max_frames,
    uint64_t seed
) {
    Qwen3TtsRequestInputV1 input;
    memset(&input, 0, sizeof(input));
    input.struct_size = (uint32_t)sizeof(input);
    input.language = QWEN3_TTS_LANGUAGE_GERMAN;
    input.text_utf8 = text;
    input.text_bytes = text_bytes;
    input.instruct_utf8 = instruction;
    input.instruct_bytes = instruction_bytes;
    input.generation = generation_config(max_frames, seed);
    return input;
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s MODEL_ROOT OUTPUT_WAV [MAX_CODEC_FRAMES]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* model_root = argv[1];
    const char* wav_path = argv[2];
    const uint32_t max_frames = argc == 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 40U;
    if (max_frames == 0U || max_frames > QWEN3_TTS_RUNTIME_MAX_CODEC_FRAMES) {
        fprintf(stderr, "MAX_CODEC_FRAMES is outside the public runtime limit\n");
        return EXIT_FAILURE;
    }

    char error[ERROR_CAPACITY] = {0};
    if (qwen3_tts_runtime_abi_version_v1() != QWEN3_TTS_RUNTIME_ABI_VERSION_V1) {
        fprintf(stderr, "runtime ABI version mismatch\n");
        return EXIT_FAILURE;
    }

    Qwen3TtsEngineConfigV1 config = engine_config();
    Qwen3TtsEngineV1* engine = (Qwen3TtsEngineV1*)(uintptr_t)1U;
    require_status(
        "null engine output",
        qwen3_tts_engine_create_v1(
            (const uint8_t*)model_root,
            strlen(model_root),
            &config,
            NULL,
            error,
            sizeof(error)
        ),
        QWEN3_TTS_RUNTIME_INVALID_ARGUMENT,
        error
    );

    Qwen3TtsEngineConfigV1 invalid_config = config;
    invalid_config.struct_size = 0U;
    require_status(
        "invalid engine struct size",
        qwen3_tts_engine_create_v1(
            (const uint8_t*)model_root,
            strlen(model_root),
            &invalid_config,
            &engine,
            error,
            sizeof(error)
        ),
        QWEN3_TTS_RUNTIME_INVALID_ARGUMENT,
        error
    );
    if (engine != NULL) {
        fprintf(stderr, "engine output was not cleared after invalid input\n");
        return EXIT_FAILURE;
    }

    const double model_load_started = monotonic_milliseconds();
    const int32_t create_status = qwen3_tts_engine_create_v1(
        (const uint8_t*)model_root,
        strlen(model_root),
        &config,
        &engine,
        error,
        sizeof(error)
    );
    if (create_status != QWEN3_TTS_RUNTIME_OK) {
        fail_status("engine create", create_status, error);
    }
    const double model_load_milliseconds = monotonic_milliseconds() - model_load_started;

    const uint8_t invalid_utf8[] = {0xffU};
    const uint8_t instruction[] =
        "A calm, relaxed adult male voice. Natural pace, low energy, clear articulation.";
    Qwen3TtsRequestInputV1 invalid_input = request_input(
        invalid_utf8,
        sizeof(invalid_utf8),
        instruction,
        sizeof(instruction) - 1U,
        max_frames,
        7U
    );
    Qwen3TtsRequestV1* invalid_request = (Qwen3TtsRequestV1*)(uintptr_t)1U;
    require_status(
        "invalid UTF-8 request",
        qwen3_tts_request_start_v1(
            engine,
            &invalid_input,
            &invalid_request,
            error,
            sizeof(error)
        ),
        QWEN3_TTS_RUNTIME_INVALID_UTF8,
        error
    );
    if (invalid_request != NULL) {
        fprintf(stderr, "request output was not cleared after invalid UTF-8\n");
        return EXIT_FAILURE;
    }

    const uint8_t warm_text[] = "Kurzer Abbruchtest.";
    Qwen3TtsRequestInputV1 warm_input = request_input(
        warm_text,
        sizeof(warm_text) - 1U,
        instruction,
        sizeof(instruction) - 1U,
        16U,
        11U
    );
    Qwen3TtsRequestV1* warm_request = NULL;
    require_status(
        "warm request start",
        qwen3_tts_request_start_v1(
            engine,
            &warm_input,
            &warm_request,
            error,
            sizeof(error)
        ),
        QWEN3_TTS_RUNTIME_OK,
        error
    );
    require_status(
        "warm request cancel",
        qwen3_tts_request_cancel_v1(warm_request, error, sizeof(error)),
        QWEN3_TTS_RUNTIME_OK,
        error
    );
    require_status(
        "warm request destroy",
        qwen3_tts_request_destroy_v1(warm_request, error, sizeof(error)),
        QWEN3_TTS_RUNTIME_OK,
        error
    );

    const uint8_t text[] =
        "Guten Morgen. Dies ist ein ruhiger Test der nativen, fortlaufenden Sprachausgabe.";
    Qwen3TtsRequestInputV1 input = request_input(
        text,
        sizeof(text) - 1U,
        instruction,
        sizeof(instruction) - 1U,
        max_frames,
        23U
    );
    Qwen3TtsRequestV1* request = NULL;
    const double request_started = monotonic_milliseconds();
    require_status(
        "request start",
        qwen3_tts_request_start_v1(engine, &input, &request, error, sizeof(error)),
        QWEN3_TTS_RUNTIME_OK,
        error
    );

    struct timespec settle = {.tv_sec = 0, .tv_nsec = 300000000L};
    while (nanosleep(&settle, &settle) != 0 && errno == EINTR) {
    }

    int16_t pcm[PCM_CAPACITY];
    Qwen3TtsAudioPacketV1 packet;
    memset(&packet, 0x7f, sizeof(packet));
    require_status(
        "undersized PCM preflight",
        qwen3_tts_request_poll_v1(
            request,
            0U,
            pcm,
            PCM_CAPACITY - 1U,
            &packet,
            error,
            sizeof(error)
        ),
        QWEN3_TTS_RUNTIME_INVALID_ARGUMENT,
        error
    );
    if (packet.codec_frames != 0U || packet.sample_count != 0U) {
        fprintf(stderr, "packet output was not cleared on preflight failure\n");
        return EXIT_FAILURE;
    }

    require_status(
        "engine destroy with live request",
        qwen3_tts_engine_destroy_v1(engine, error, sizeof(error)),
        QWEN3_TTS_RUNTIME_OK,
        error
    );
    engine = NULL;

    FILE* wav = fopen(wav_path, "wb+");
    if (wav == NULL) {
        perror("open WAV");
        return EXIT_FAILURE;
    }
    write_wav_header(wav, 0U);

    uint64_t expected_sequence = 0U;
    uint64_t expected_frame = 0U;
    uint64_t expected_sample = 0U;
    uint64_t nonzero_samples = 0U;
    int16_t minimum = INT16_MAX;
    int16_t maximum = INT16_MIN;
    double first_audio_milliseconds = 0.0;
    int saw_final = 0;

    for (;;) {
        for (size_t index = 0; index < PCM_CAPACITY; ++index) {
            pcm[index] = INT16_MIN;
        }
        memset(&packet, 0, sizeof(packet));
        const int32_t status = qwen3_tts_request_poll_v1(
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
            fclose(wav);
            fail_status("request poll", status, error);
        }
        if (expected_sequence == 0U) {
            first_audio_milliseconds = monotonic_milliseconds() - request_started;
            if (packet.codec_frames != 1U) {
                fprintf(stderr, "first packet did not contain exactly one codec frame\n");
                return EXIT_FAILURE;
            }
        }
        if (packet.sequence != expected_sequence
            || packet.first_codec_frame != expected_frame
            || packet.first_sample != expected_sample
            || packet.codec_frames == 0U
            || packet.codec_frames > config.packet_frames
            || packet.sample_count
                != packet.codec_frames * QWEN3_TTS_RUNTIME_SAMPLES_PER_CODEC_FRAME
            || packet.sample_rate != QWEN3_TTS_RUNTIME_SAMPLE_RATE
            || packet.channels != QWEN3_TTS_RUNTIME_CHANNELS) {
            fprintf(stderr, "packet continuity or format invariant failed\n");
            return EXIT_FAILURE;
        }
        for (size_t index = packet.sample_count; index < PCM_CAPACITY; ++index) {
            if (pcm[index] != INT16_MIN) {
                fprintf(stderr, "poll wrote beyond packet.sample_count\n");
                return EXIT_FAILURE;
            }
        }
        for (size_t index = 0; index < packet.sample_count; ++index) {
            if (pcm[index] != 0) {
                ++nonzero_samples;
            }
            if (pcm[index] < minimum) {
                minimum = pcm[index];
            }
            if (pcm[index] > maximum) {
                maximum = pcm[index];
            }
        }
        if (fwrite(pcm, sizeof(int16_t), packet.sample_count, wav) != packet.sample_count) {
            perror("write PCM");
            return EXIT_FAILURE;
        }
        ++expected_sequence;
        expected_frame += packet.codec_frames;
        expected_sample += packet.sample_count;
        saw_final = packet.is_final != 0U;
        if (saw_final) {
            continue;
        }
    }
    if (!saw_final || expected_frame == 0U || nonzero_samples == 0U) {
        fprintf(stderr, "runtime did not produce a final non-silent stream\n");
        return EXIT_FAILURE;
    }

    Qwen3TtsRequestMetricsV1 metrics;
    memset(&metrics, 0, sizeof(metrics));
    require_status(
        "request metrics",
        qwen3_tts_request_metrics_v1(request, &metrics, error, sizeof(error)),
        QWEN3_TTS_RUNTIME_OK,
        error
    );
    if (metrics.generated_codec_frames != expected_frame
        || metrics.emitted_samples != expected_sample
        || metrics.emitted_packets != expected_sequence
        || metrics.first_audio_microseconds == 0U) {
        fprintf(stderr, "request metrics disagree with delivered packets\n");
        return EXIT_FAILURE;
    }

    write_wav_header(wav, (uint32_t)expected_sample);
    if (fclose(wav) != 0) {
        perror("close WAV");
        return EXIT_FAILURE;
    }
    require_status(
        "request destroy",
        qwen3_tts_request_destroy_v1(request, error, sizeof(error)),
        QWEN3_TTS_RUNTIME_OK,
        error
    );

    const double wall_milliseconds = monotonic_milliseconds() - request_started;
    const double audio_milliseconds = (double)expected_sample * 1000.0
        / (double)QWEN3_TTS_RUNTIME_SAMPLE_RATE;
    printf(
        "{\"status\":\"pass\",\"model_load_ms\":%.3f,"
        "\"ttfa_ms\":%.3f,\"wall_ms\":%.3f,\"audio_ms\":%.3f,"
        "\"rtf\":%.6f,\"packets\":%" PRIu64 ",\"codec_frames\":%" PRIu64 ","
        "\"samples\":%" PRIu64 ",\"nonzero_samples\":%" PRIu64 ","
        "\"pcm_min\":%" PRId16 ",\"pcm_max\":%" PRId16 ","
        "\"metric_first_audio_us\":%" PRIu64 ","
        "\"peak_request_device_bytes\":%" PRIu64 ","
        "\"peak_request_host_bytes\":%" PRIu64 "}\n",
        model_load_milliseconds,
        first_audio_milliseconds,
        wall_milliseconds,
        audio_milliseconds,
        wall_milliseconds / audio_milliseconds,
        expected_sequence,
        expected_frame,
        expected_sample,
        nonzero_samples,
        minimum,
        maximum,
        metrics.first_audio_microseconds,
        metrics.peak_request_device_bytes,
        metrics.peak_request_host_bytes
    );
    return EXIT_SUCCESS;
}
