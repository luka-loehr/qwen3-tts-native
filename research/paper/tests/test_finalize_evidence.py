from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "finalize_evidence.py"
SPEC = importlib.util.spec_from_file_location("paper_finalize_evidence", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load finalizer: {MODULE_PATH}")
finalizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = finalizer
SPEC.loader.exec_module(finalizer)

GIB = 1024**3
NATIVE_COMMIT = "1" * 40
STOCK_COMMIT = "2" * 40
NATIVE_IMAGE = "sha256:" + "3" * 64
STOCK_IMAGE = "sha256:" + "4" * 64
NATIVE_ARTIFACT_EVIDENCE = "5" * 64
STOCK_ARTIFACT_EVIDENCE = "6" * 64
NATIVE_ARTIFACT_MANIFEST = "7" * 64
MANIFEST_SHA = "8" * 64
WORKLOAD_SHA = "9" * 64
NATIVE_REGISTRY = "sha256:" + "a" * 64
STOCK_REGISTRY = "sha256:" + "b" * 64


def implementation(engine: str, registry_state: str) -> dict[str, object]:
    native = engine == "native"
    parameter_count = 1_700_000_000 if native else 1_750_000_000
    artifact = {
        "repository": finalizer.EXPECTED_MODEL_REPOSITORY,
        "revision": finalizer.EXPECTED_MODEL_REVISION,
        "variant": "1.7B VoiceDesign",
        "parameter_count": parameter_count,
        "precision": ["bfloat16"],
        "manifest_sha256": NATIVE_ARTIFACT_MANIFEST if native else None,
        "weight_files": [
            {
                "path": "model.safetensors",
                "sha256": "c" * 64,
                "bytes": 3 * GIB if native else 4 * GIB,
                "parameter_count": parameter_count,
                "precision": "bfloat16",
            }
        ],
        "evidence": {
            "path": f"artifacts/{engine}/model-artifact.json",
            "sha256": (NATIVE_ARTIFACT_EVIDENCE if native else STOCK_ARTIFACT_EVIDENCE),
        },
    }
    result: dict[str, object] = {
        "id": engine,
        "role": engine,
        "name": "Native" if native else "Stock SGLang",
        "version": "test",
        "source_commit": NATIVE_COMMIT if native else STOCK_COMMIT,
        "source_url": f"https://example.invalid/{engine}",
        "local_image": {
            "reference": f"example/{engine}:candidate",
            "id": NATIVE_IMAGE if native else STOCK_IMAGE,
            "unpacked_size_bytes": 6 * GIB if native else 7 * GIB,
        },
        "model_artifact": artifact,
        "api_protocol": "HTTP/1.1 raw PCM16",
        "streaming_semantics": "progressive" if native else "buffered",
        "runtime_components": ["Rust", "CUDA"] if native else ["SGLang-Omni"],
    }
    if registry_state != "absent":
        registry: dict[str, object] = {
            "reference": f"example/{engine}@digest",
            "manifest_digest": NATIVE_REGISTRY if native else STOCK_REGISTRY,
            "evidence": {
                "path": f"artifacts/{engine}/registry-image.json",
                "sha256": "d" * 64,
            },
        }
        if registry_state == "sized":
            registry["compressed_size_bytes"] = 2 * GIB if native else 3 * GIB
        result["registry_image"] = registry
    return result


def manifest(registry_state: str = "absent") -> dict[str, object]:
    return {
        "schema_version": "1.2",
        "evidence_kind": "production",
        "report": {
            "benchmark_id": "paper-finalizer-test",
            "title": "Paper finalizer deterministic test manifest",
            "generated_at": "2026-07-17T12:00:00Z",
            "authors": ["Luka Loehr"],
        },
        "system": {
            "host_model": finalizer.EXPECTED_HOST_MODEL,
            "hostname_alias": "spark-benchmark-host",
            "os": "Ubuntu 24.04 LTS",
            "kernel": "6.11.0-test",
            "architecture": "aarch64",
            "cpu": "NVIDIA Grace",
            "accelerator": "NVIDIA GB10 Blackwell",
            "driver_version": "580.00",
            "cuda_version": "13.0",
            "physical_unified_memory_bytes": 128_000_000_000,
            "power_measurement_source": "nvidia-smi board power.draw",
            "notes": ["Controlled test host."],
        },
        "model": {
            "repository": finalizer.EXPECTED_MODEL_REPOSITORY,
            "revision": finalizer.EXPECTED_MODEL_REVISION,
            "variant": "1.7B VoiceDesign",
        },
        "workload": {
            "corpus_sha256": WORKLOAD_SHA,
            "ordered_seeds": list(range(42, 52)),
            "sample_rate_hz": 24_000,
            "channels": 1,
            "sample_format": "pcm_s16le",
            "response_mode": "streaming",
            "warmup_requests_per_run": 24,
            "minimum_measured_requests_per_profile": 200,
            "minimum_rounds_per_subject": 2,
            "profiles": [
                {"id": "B1", "concurrency": 1, "repetitions_per_request": 20},
                {"id": "B3", "concurrency": 3, "repetitions_per_request": 21},
                {"id": "B6", "concurrency": 6, "repetitions_per_request": 24},
            ],
            "language_policy": "Use the explicit language in each workload row.",
        },
        "implementations": [
            implementation("native", registry_state),
            implementation("sglang", registry_state),
        ],
        "methodology": {
            "clock_source": "Client monotonic clock",
            "ttfa_definition": "Elapsed request time to first playable PCM byte.",
            "rtf_definition": "Scenario wall time divided by generated audio time.",
            "throughput_definition": "Successful requests per scenario wall time.",
            "memory_definition": "Peak process RSS and GPU-visible unified memory.",
            "power_definition": "NVIDIA board power sampled every 100 milliseconds.",
            "energy_definition": "Integrated board energy minus idle baseline energy.",
            "sampling_interval_ms": 100,
            "run_order": "Round one Native then SGLang; round two reversed.",
            "statistical_method": "Two complete rounds with raw-request percentiles.",
            "environment_controls": ["No competing CUDA processes."],
        },
        "limitations": ["Stock SGLang exposes unknown EOS metadata."],
    }


def bundle(registry_state: str = "absent") -> SimpleNamespace:
    summaries: dict[tuple[str, str, int], dict[str, object]] = {}
    resources: dict[tuple[str, str, int], dict[str, object]] = {}
    profile_index = {"B1": 1, "B3": 3, "B6": 6}
    for round_number in (1, 2):
        for engine_index, engine in enumerate(("native", "sglang")):
            for profile, concurrency in profile_index.items():
                offset = round_number * 10 + engine_index * 100 + concurrency
                successes = 200 + concurrency
                summaries[(engine, profile, round_number)] = {
                    "successful_requests": successes,
                    "ttfa_ms": {
                        "p50": 10.123456789 + offset,
                        "p95": 20.123456789 + offset,
                    },
                    "aggregate_rtf": 0.123456789 + offset / 1000,
                    "summed_request_wall_rtf": 9.9 + offset,
                    "total_audio_seconds": 120.0,
                }
                resources[(engine, profile, round_number)] = {
                    "process_rss_peak_bytes": (1 + concurrency) * GIB,
                    "gpu_unified_memory_peak_bytes": (2 + concurrency) * GIB,
                    "average_power_w": 45.125 + offset,
                    "energy_j": 240.0 + offset,
                }
    return SimpleNamespace(
        manifest=manifest(registry_state),
        manifest_sha256=MANIFEST_SHA,
        run_summaries=summaries,
        run_resources=resources,
    )


def aggregates() -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for engine_index, engine in enumerate(("native", "sglang")):
        result[engine] = {}
        for profile, concurrency in (("B1", 1), ("B3", 3), ("B6", 6)):
            success = 2 * (200 + concurrency)
            result[engine][profile] = {
                "total": success,
                "success": success,
                "natural_eos": success if engine == "native" else 0,
                "eos_unknown": success if engine == "sglang" else 0,
                "ttfa_p95": 30.125 + engine_index * 100 + concurrency,
                "aggregate_rtf": 0.25 + engine_index + concurrency / 100,
            }
    return result


class FinalizeEvidenceTests(unittest.TestCase):
    def test_exact_rows_formulas_order_and_source_precision(self) -> None:
        outputs = finalizer.build_outputs(bundle(), aggregates())
        tex = outputs.tex.decode("ascii")
        self.assertIn(
            "Native & 1 & B1 & 1 & 201 & 21.123 & 31.123 & 0.1345",
            tex,
        )
        self.assertIn("Native & 1 & B1 & 2.000 & 3.000 & 56.125 & 125.500", tex)
        self.assertNotIn("20.9", tex)

        native_lines = outputs.native_dat.decode("ascii").splitlines()
        stock_lines = outputs.sglang_dat.decode("ascii").splitlines()
        self.assertEqual(
            native_lines[0],
            "concurrency round ttfa_p95_ms aggregate_rtf",
        )
        self.assertEqual(
            [line.split()[:2] for line in native_lines[1:]],
            [["1", "1"], ["3", "1"], ["6", "1"], ["1", "2"], ["3", "2"], ["6", "2"]],
        )
        self.assertEqual(native_lines[1], "1 1 31.123456789 0.134456789")
        self.assertEqual(stock_lines[1], "1 1 131.123456789 0.234456789")
        self.assertEqual(len(native_lines), 7)
        self.assertEqual(len(stock_lines), 7)

    def test_registry_absent_digest_only_and_sized(self) -> None:
        absent = finalizer.build_outputs(bundle("absent"), aggregates()).tex.decode(
            "ascii"
        )
        self.assertIn(r"\newcommand{\NativeImageDigest}{N/A}", absent)
        self.assertIn("Native & 6.000 & N/A & 1700000000 & 3.000", absent)

        digest_only = finalizer.build_outputs(
            bundle("digest-only"), aggregates()
        ).tex.decode("ascii")
        self.assertIn(r"\newcommand{\NativeImageDigest}{\code{", digest_only)
        self.assertIn("Native & 6.000 & N/A & 1700000000 & 3.000", digest_only)

        sized = finalizer.build_outputs(bundle("sized"), aggregates()).tex.decode(
            "ascii"
        )
        self.assertIn("Native & 6.000 & 2.000 & 1700000000 & 3.000", sized)
        self.assertIn("Stock SGLang & 7.000 & 3.000 & 1750000000 & 4.000", sized)

    def test_artifact_evidence_digest_is_not_manifest_digest(self) -> None:
        tex = finalizer.build_outputs(bundle(), aggregates()).tex.decode("ascii")
        self.assertIn(NATIVE_ARTIFACT_EVIDENCE[:8], tex)
        self.assertNotIn(NATIVE_ARTIFACT_MANIFEST[:8], tex)
        self.assertIn(r"\newcommand{\StockModelArtifactEvidenceSha}", tex)

    def test_rejects_missing_and_extra_run_cells(self) -> None:
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                candidate = bundle()
                if mutation == "missing":
                    del candidate.run_summaries[("native", "B1", 1)]
                else:
                    candidate.run_resources[("native", "B1", 3)] = copy.deepcopy(
                        candidate.run_resources[("native", "B1", 1)]
                    )
                with self.assertRaisesRegex(
                    finalizer.FinalizationError, "exactly twelve run"
                ):
                    finalizer.build_outputs(candidate, aggregates())

    def test_rejects_paper_protocol_mismatch(self) -> None:
        mutations = (
            ("system", "host_model", "Another host"),
            ("system", "architecture", "x86_64"),
            ("system", "accelerator", "Another GPU"),
            ("model", "repository", "another/model"),
            ("model", "revision", "0" * 40),
            ("workload", "sample_rate_hz", 48_000),
            ("workload", "warmup_requests_per_run", 25),
            ("methodology", "sampling_interval_ms", 200),
        )
        for section, field, value in mutations:
            with self.subTest(section=section, field=field):
                candidate = bundle()
                candidate.manifest[section][field] = value
                with self.assertRaisesRegex(
                    finalizer.FinalizationError, "paper (target|protocol)"
                ):
                    finalizer.build_outputs(candidate, aggregates())

    def test_output_is_deterministic_and_rejects_unsafe_tex_token(self) -> None:
        candidate = bundle("sized")
        first = finalizer.build_outputs(candidate, aggregates())
        second = finalizer.build_outputs(copy.deepcopy(candidate), aggregates())
        self.assertEqual(first, second)
        candidate.manifest["report"]["generated_at"] = "2026-07-17%unsafe"
        with self.assertRaisesRegex(finalizer.FinalizationError, "unsafe identity"):
            finalizer.build_outputs(candidate, aggregates())

    def test_failure_before_write_preserves_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._paper_root(Path(directory))
            originals = {
                "evidence_placeholders.tex": b"old tex\n",
                "native-runs.dat": b"old native\n",
                "sglang-runs.dat": b"old stock\n",
            }
            for name, payload in originals.items():
                (root / "data" / name).write_bytes(payload)
            invalid = bundle()
            del invalid.run_summaries[("native", "B1", 1)]
            with mock.patch.object(
                finalizer, "_validated_bundle", return_value=(invalid, aggregates())
            ):
                with self.assertRaises(finalizer.FinalizationError):
                    finalizer.finalize(root / "manifest.json", root)
            for name, payload in originals.items():
                self.assertEqual((root / "data" / name).read_bytes(), payload)

    def test_atomic_write_replaces_managed_files_and_rejects_symlink(self) -> None:
        outputs = finalizer.build_outputs(bundle(), aggregates())
        with tempfile.TemporaryDirectory() as directory:
            root = self._paper_root(Path(directory))
            finalizer.write_outputs(outputs, root)
            for name, payload in outputs.by_name().items():
                self.assertEqual((root / "data" / name).read_bytes(), payload)
            self.assertEqual(list((root / "data").glob(".*.tmp")), [])
            target = root / "outside.dat"
            target.write_bytes(b"outside\n")
            plot = root / "data" / "native-runs.dat"
            plot.unlink()
            plot.symlink_to(target)
            with self.assertRaisesRegex(finalizer.FinalizationError, "symlink"):
                finalizer.write_outputs(outputs, root)
            self.assertEqual(target.read_bytes(), b"outside\n")

    @staticmethod
    def _paper_root(root: Path) -> Path:
        (root / "data").mkdir()
        (root / "main.tex").write_text("paper\n", encoding="ascii")
        (root / "Makefile").write_text("all:\n", encoding="ascii")
        (root / "data" / "evidence_placeholders.tex").write_text(
            "pending\n", encoding="ascii"
        )
        return root


if __name__ == "__main__":
    unittest.main()
