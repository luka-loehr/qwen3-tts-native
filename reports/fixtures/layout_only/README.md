# TEST FIXTURE - NOT BENCHMARK EVIDENCE

Every value in this directory is synthetic and exists only to exercise report
layout and validation branches. It does not describe measured Qwen3-TTS,
Native, SGLang, DGX Spark, latency, memory, power, energy, reliability, or image
size.

This fixture deliberately remains on normalized schema version `1.0`. That
version is restricted to test fixtures; production evidence must use the direct
Rust client-run schema version `1.1`.

The generator rejects this fixture unless `--allow-test-fixture` is passed. It
also refuses to place a fixture report in `reports/output/` and prints
`TEST FIXTURE - NOT BENCHMARK EVIDENCE` on every generated page.
