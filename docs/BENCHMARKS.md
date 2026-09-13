# Performance

All results in this document were measured on one NVIDIA DGX Spark. Read the
[benchmark protocol](../benchmarks/README.md) before comparing or republishing
any result: model loading, artifact hashing, fixed-length scheduling,
natural-EOS generation, HTTP transport, container overhead, energy, and
subjective listening results are separate measurements.

## At a glance

Single-stream profile B1, native versus stock SGLang:

|  | native | stock SGLang |
| --- | ---: | ---: |
| Time to first audio, p95 | **94–96 ms** | 2.69–2.70 s |
| Peak GPU unified memory | **5.68 GB** | 108.90 GB |
| Aggregate RTF (compute ÷ audio, lower is better) | 0.80 | **0.50** |

Native's measured advantages are progressive time to first audio (~28×) and
~19× lower peak memory; stock SGLang keeps the better aggregate throughput in
every profile.

## Controlled native-versus-stock-SGLang comparison

The final schema-1.2 study completed two alternating-order rounds of B1, B3,
and B6 on one NVIDIA DGX Spark. All 2,600 measured requests succeeded after 24
warm-ups per cell, with no competing CUDA process. The ranges below span the
two accepted rounds:

| Profile | Native aggregate RTF | Stock aggregate RTF | Native TTFA p95 | Stock TTFA p95 |
| ---: | ---: | ---: | ---: | ---: |
| B1 | 0.800–0.803 | 0.497–0.499 | 93.89–95.58 ms | 2,691.51–2,703.91 ms |
| B3 | 0.641–0.642 | 0.186–0.197 | 215.55–216.81 ms | 2,720.93–2,814.63 ms |
| B6 | 0.617–0.618 | 0.102–0.112 | 405.38–406.04 ms | 2,873.49–3,145.88 ms |

Native peaked at 5.68 GB of observed GPU unified memory; stock SGLang peaked
at 108.90 GB. Stock SGLang achieved better aggregate throughput in every
profile. Native's measured advantages were progressive time to first audio
and approximately 19.2 times lower peak GPU unified-memory use. Stock delivery
was completion-buffered and exposed no authoritative EOS metadata, so its
TTFA and completion semantics are reported explicitly rather than treated as
equivalent to native progressive delivery.

See the
[benchmark report](../reports/output/qwen3-tts-native-vs-sglang-stock-dgx-spark-2026-07-17-428307c-report.pdf),
[research paper](../research/paper/qwen3-tts-native-paper.pdf), and
[benchmark protocol](../benchmarks/README.md) for methodology, energy results,
and limitations. The report and paper derive from the `v0.1.0` release
evidence bundle.

## Historical native baselines

The remaining results are checked JSON evidence from direct native runs. They
are pre-release baselines and do not replace the controlled comparison above.

### Full natural-end-of-sequence endurance

The native C ABI completed 200 measured single-stream requests after three
warm-ups. Every request reached natural codec EOS; none failed or hit the
512-frame emergency guard.

| Measurement | Result |
| --- | ---: |
| Completed requests | 200 / 200 |
| TTFA p50 / p95 / p99 | 74.01 / 76.95 / 79.82 ms |
| Request RTF p50 / p95 / p99 | 0.733 / 0.740 / 0.743 |
| Aggregate RTF | 0.734 |
| Generated audio | 926.72 s |
| Peak process RSS | 4,045,112 KiB |
| Peak additional device allocation per request | 141,285,524 bytes |

An RTF below 1.0 means synthesis completed faster than the generated audio's
playback duration. Evidence:
[`native-runtime-natural-eos-endurance-a6bc32e.json`](../benchmarks/results/native-runtime-natural-eos-endurance-a6bc32e.json).

### Official-language qualification

The multilingual native C-ABI run completed all 24 corpus entries covering all
ten explicit languages plus `Auto`. Every request streamed progressively,
preserved exact PCM copy bounds, and ended at natural codec EOS.

| Measurement | Result |
| --- | ---: |
| Completed corpus entries | 24 / 24 |
| TTFA p95 | 78.47 ms |
| Request RTF p50 / p95 | 0.745 / 0.763 |
| Aggregate RTF | 0.751 |
| Generated audio | 200.24 s |

Evidence:
[`native-multilingual-natural-eos-ff061b6.json`](../benchmarks/results/native-multilingual-natural-eos-ff061b6.json).
The saved WAV corpus is listening evidence, not an automated claim about
pronunciation, naturalness, or instruction adherence.

### Warmed HTTP server qualification

A direct native server run became ready after a full pipeline warm-up in
10.261 seconds. A German progressive request delivered its first audio in
77.884 ms and generated 4.72 seconds of audio at RTF 0.710. An Italian `Auto`
request returned a valid, unclipped 24 kHz PCM WAV at RTF 0.707. SIGTERM closed
the process and loopback port in under one second.

That run used a warm host filesystem cache and coexisted with an already
running SGLang service, so it is not a performance comparison. Evidence:
[`native-server-startup-warmup-ce46acb.json`](../benchmarks/results/native-server-startup-warmup-ce46acb.json).

### Fixed-length concurrency throughput

An earlier C-ABI scheduler qualification measured 200 fixed 320 ms requests at
each concurrency level. It is a packet-delivery and throughput test, not a
natural-EOS or audio-quality corpus.

| Concurrency | Completed | TTFA p95 | Request RTF p50 | Aggregate RTF |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 200 / 200 | 78.24 ms | 0.765 | 0.767 |
| 3 | 200 / 200 | 186.42 ms | 1.800 | 0.601 |
| 6 | 200 / 200 | 364.62 ms | 3.557 | 0.594 |

At B3 and B6, aggregate throughput was faster than real time while an
individual request was not. Evidence:
[`native-runtime-public-c-abi-qualification.json`](../benchmarks/results/native-runtime-public-c-abi-qualification.json).
