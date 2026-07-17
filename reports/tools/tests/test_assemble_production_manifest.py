from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_DIR / "assemble_production_manifest.py"
SPEC = importlib.util.spec_from_file_location(
    "assemble_production_manifest", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
assembler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assembler
SPEC.loader.exec_module(assembler)
REPORT_GENERATOR_PATH = TOOLS_DIR.parent / "generate_report.py"
REPORT_SPEC = importlib.util.spec_from_file_location(
    "generate_report_for_manifest_assembler_tests", REPORT_GENERATOR_PATH
)
assert REPORT_SPEC is not None and REPORT_SPEC.loader is not None
report_generator = importlib.util.module_from_spec(REPORT_SPEC)
sys.modules[REPORT_SPEC.name] = report_generator
REPORT_SPEC.loader.exec_module(report_generator)


NATIVE_DIGEST = "sha256:" + "a" * 64
SGLANG_DIGEST = "sha256:" + "b" * 64
TOOLING_COMMIT = "c" * 40
MODEL_MANIFEST_DIGEST = "d" * 64
NATIVE_IMAGE_SIZE = 5_000_000_000
SGLANG_IMAGE_SIZE = 29_000_000_000


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )


def refresh_checksums(run_dir: Path) -> None:
    records = []
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name != "SHA256SUMS":
            payload = path.read_bytes()
            records.append(
                f"{sha256(payload)}  {path.relative_to(run_dir).as_posix()}\n"
            )
    (run_dir / "SHA256SUMS").write_text("".join(records), encoding="utf-8")


class EvidenceFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.runs = self.evidence / "runs"
        self.workload = self.evidence / "workload" / "workload.jsonl"
        self.config = root / "production-metadata.json"
        self.output = self.evidence / "manifest.json"
        self.evidence.mkdir(parents=True)
        workload_records = [
            {
                "id": "multilingual-001",
                "text": "A calm production benchmark sentence.",
                "voice_description": "A calm adult male voice with measured pacing.",
                "language": "English",
                "seed": 42,
                "max_duration_seconds": 20.48,
                "sampling": {
                    "strategy": "sample",
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 50,
                    "repetition_penalty": 1.05,
                    "predictor": {
                        "strategy": "sample",
                        "temperature": 0.9,
                        "top_p": 1.0,
                        "top_k": 50,
                    },
                },
                "stream": True,
            }
        ]
        write_jsonl(self.workload, workload_records)
        workload_payload = self.workload.read_bytes()
        self.config_value = self._config_value(sha256(workload_payload))
        write_json(self.config, self.config_value)
        for round_number in (1, 2):
            for engine in ("native", "sglang"):
                for profile in ("B1", "B3", "B6"):
                    self._create_run(engine, profile, round_number, workload_payload)

    def _config_value(self, workload_digest: str) -> dict[str, Any]:
        model = {
            "repository": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "revision": "5ecdb1a0123456789abcdef0123456789abcdef0",
            "variant": "1.7B VoiceDesign",
        }

        def implementation(engine: str, digest: str) -> dict[str, Any]:
            native = engine == "native"
            weights = [
                {
                    "path": "model.safetensors",
                    "sha256": "e" * 64,
                    "bytes": 2_000_000_000,
                    "parameter_count": 1_000_000_000,
                    "precision": "bfloat16",
                },
                {
                    "path": "speech_tokenizer/model.safetensors",
                    "sha256": ("f" if native else "9") * 64,
                    "bytes": 200_000_000 if native else 600_000_000,
                    "parameter_count": 100_000_000 if native else 150_000_000,
                    "precision": "bfloat16" if native else "float32",
                },
            ]
            artifact: dict[str, Any] = {
                **model,
                "parameter_count": sum(item["parameter_count"] for item in weights),
                "precision": sorted({item["precision"] for item in weights}),
                "manifest_sha256": MODEL_MANIFEST_DIGEST if native else None,
                "weight_files": weights,
            }
            artifact_payload = {
                "schema_version": "qwen3-tts-model-artifact/v1",
                "implementation_id": engine,
                "local_image_id": digest,
                **artifact,
                "source": {
                    "kind": "container_image" if native else "read_only_bind_mount",
                    "container_path": "/opt/qwen3-tts/model"
                    if native
                    else "/models/hf-repository",
                    "read_only": True,
                    **(
                        {}
                        if native
                        else {
                            "host_path": "/srv/fixture/hf-cache",
                            "snapshot_path": "snapshots/5ecdb1a0",
                            "revision_ref_path": "refs/main",
                        }
                    ),
                },
            }
            artifact_path = self.evidence / "artifacts" / engine / "model-artifact.json"
            write_json(artifact_path, artifact_payload)
            artifact["evidence"] = {
                "path": artifact_path.relative_to(self.evidence).as_posix(),
                "sha256": sha256(artifact_path.read_bytes()),
            }
            return {
                "id": engine,
                "role": engine,
                "name": "Qwen3 TTS Native" if engine == "native" else "SGLang Omni",
                "version": "0.1.0",
                "source_commit": "1" * 40 if engine == "native" else "2" * 40,
                "source_url": f"https://example.invalid/{engine}",
                "local_image": {
                    "reference": f"fixture/{engine}:candidate",
                    "id": digest,
                    "unpacked_size_bytes": NATIVE_IMAGE_SIZE
                    if native
                    else SGLANG_IMAGE_SIZE,
                },
                "model_artifact": artifact,
                "api_protocol": "HTTP/1.1 raw PCM16",
                "streaming_semantics": "progressive"
                if engine == "native"
                else "buffered",
                "runtime_components": ["Rust", "CUDA"]
                if engine == "native"
                else ["SGLang-Omni"],
            }

        return {
            "report": {
                "benchmark_id": "spark-native-vs-sglang-2026-07-17",
                "title": "Native Qwen3 TTS versus stock SGLang on DGX Spark",
                "generated_at": "2026-07-17T12:00:00Z",
                "authors": ["Luka Loehr"],
            },
            "system": {
                "host_model": "NVIDIA DGX Spark",
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
                "notes": ["Controlled single-GPU benchmark host."],
            },
            "model": model,
            "workload": {
                "corpus_sha256": workload_digest,
                "ordered_seeds": [42],
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
                implementation("native", NATIVE_DIGEST),
                implementation("sglang", SGLANG_DIGEST),
            ],
            "methodology": {
                "clock_source": "Client monotonic clock",
                "ttfa_definition": "Elapsed request time to the first playable PCM byte.",
                "rtf_definition": "Measured scenario wall time divided by generated audio time.",
                "throughput_definition": "Successful requests divided by measured scenario wall time.",
                "memory_definition": "Peak measured process RSS and NVIDIA unified memory.",
                "power_definition": "NVIDIA board power sampled every one hundred milliseconds.",
                "energy_definition": "Trapezoidal board energy minus measured idle baseline energy.",
                "sampling_interval_ms": 100,
                "run_order": "Round one Native then SGLang; round two reversed.",
                "statistical_method": "Two complete rounds with raw-request percentiles.",
                "environment_controls": [
                    "No competing CUDA processes from idle baseline through measured end."
                ],
            },
            "limitations": [
                "Stock SGLang exposes completion-buffered audio and unknown EOS metadata."
            ],
        }

    def run_dir(self, engine: str, profile: str, round_number: int) -> Path:
        return self.runs / f"round-{round_number:02d}" / engine / profile

    def _create_run(
        self,
        engine: str,
        profile: str,
        round_number: int,
        workload_payload: bytes,
    ) -> None:
        run_dir = self.run_dir(engine, profile, round_number)
        evidence_prefix = run_dir.relative_to(self.evidence).as_posix()
        client_payload = b"qwen3-tts-http-bench-fixture\n"
        image_digest = NATIVE_DIGEST if engine == "native" else SGLANG_DIGEST
        implementation = self.config_value["implementations"][
            0 if engine == "native" else 1
        ]
        request_count = 200
        invocation = {
            "schema_version": "qwen3-tts-qualifying-run/v1",
            "engine": engine,
            "profile": profile,
            "round": round_number,
            "container": {"name": f"fixture-{engine}", "id": "5" * 64},
            "image": {
                "reference": implementation["local_image"]["reference"],
                "resolved_id": image_digest,
            },
            "client": {
                "path": "input/qwen3-tts-http-bench",
                "sha256": sha256(client_payload),
            },
            "workload": {
                "path": "input/workload.jsonl",
                "sha256": sha256(workload_payload),
            },
            "evidence_prefix": evidence_prefix,
            "request": {
                "endpoint": "http://127.0.0.1:8080/v1/voice-design/speech",
                "requests": request_count,
                "warmups": 24,
                "timeout_seconds": 600,
                "sglang_model": self.config_value["model"]["repository"]
                if engine == "sglang"
                else None,
            },
            "telemetry": {
                "idle_baseline_seconds": 15,
                "configured_sample_interval_ms": 100,
                "maximum_qualifying_observed_gap_ms": 200,
                "gpu_index": 0,
            },
            "tooling_repository": {
                "commit": TOOLING_COMMIT,
                "tracked_files_clean": True,
            },
        }
        write_json(run_dir / "provenance/invocation.json", invocation)
        provenance = {
            "run-qualifying-benchmark.sh": "#!/bin/sh\n",
            "capture-spark-telemetry.sh": "#!/bin/sh\n",
            "lib/process-rss-sampler.sh": "#!/usr/bin/env bash\n",
            "reduce-spark-run.sh": "#!/bin/sh\n",
            "image-inspect.json": json.dumps(
                [
                    {
                        "Id": image_digest,
                        "RepoTags": [implementation["local_image"]["reference"]],
                        "RepoDigests": [],
                        "Size": implementation["local_image"]["unpacked_size_bytes"],
                    }
                ],
                indent=2,
            )
            + "\n",
            "container-inspect.sanitized.json": "[]\n",
            "server-log-window.json": json.dumps(
                {
                    "schema_version": "qwen3-tts-server-log-window/v1",
                    "container": {"name": f"fixture-{engine}", "id": "5" * 64},
                    "since_unix_seconds": 1_752_710_400,
                    "until_unix_seconds": 1_752_710_401,
                },
                indent=2,
            )
            + "\n",
            "server.log": "",
            "client-version.txt": "fixture-client 1.0\n",
            "uname.txt": "Linux spark fixture\n",
            "nvidia-smi-list.txt": "GPU 0: fixture\n",
            "nvidia-smi-query.txt": "fixture query\n",
            "docker-version.txt": "fixture docker\n",
            "repository-status.txt": "## detached at fixture\n",
        }
        for name, contents in provenance.items():
            path = run_dir / "provenance" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "qwen3-tts-http-bench").write_bytes(client_payload)
        (input_dir / "workload.jsonl").write_bytes(workload_payload)

        summary = {
            "schema_version": "qwen3-tts-http-bench/v1",
            "backend": "native" if engine == "native" else "sglang-omni",
            "sglang_model": self.config_value["model"]["repository"]
            if engine == "sglang"
            else None,
            "concurrency": profile,
            "warmups": 24,
            "planned_requests": request_count,
            "completed_requests": request_count,
            "successful_requests": request_count,
        }
        write_json(run_dir / "client/summary.json", summary)
        requests = []
        for index in range(request_count):
            record = {
                "schema_version": "qwen3-tts-http-bench/v1",
                "request_index": index,
                "success": True,
                "finish_reason": "stop" if engine == "native" else None,
                "natural_eos": True if engine == "native" else None,
                "length_limited": False if engine == "native" else None,
                "samples": 48_000,
                "audio_seconds": 2.0,
            }
            requests.append(record)
        write_jsonl(run_dir / "client/requests.jsonl", requests)
        write_jsonl(
            run_dir / "client/packets.jsonl",
            [{"schema_version": "qwen3-tts-http-bench/v1", "fixture": True}],
        )

        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_payloads = {
            "gpu.csv": "wall_time_unix_ns,power_w\n1,50\n",
            "system.csv": "wall_time_unix_ns,memory\n1,1000\n",
            "process-rss.csv": "wall_time_unix_ns,rss\n1,1000\n",
            "process-rss-total.csv": "wall_time_unix_ns,rss\n1,1000\n",
            "gpu-processes.csv": "wall_time_unix_ns,memory\n1,4096\n",
            "gpu-process-summary.csv": "wall_time_unix_ns,competing\n1,0\n",
            "phase-events.jsonl": '{"event":"measured_end"}\n',
            "run.txt": "exit_status=0\n",
            "command.stdout": "",
            "command.stderr": "",
        }
        for name, contents in raw_payloads.items():
            (raw_dir / name).write_text(contents, encoding="utf-8")

        telemetry_paths = [
            f"{evidence_prefix}/{relative}"
            for relative in assembler.TELEMETRY_RELATIVE_PATHS
        ]
        resource = {
            "engine_id": engine,
            "profile_id": profile,
            "round": round_number,
            "process_rss_peak_bytes": 1_000_000,
            "gpu_unified_memory_peak_bytes": 4_294_967_296,
            "average_power_w": 80.0,
            "peak_power_w": 100.0,
            "energy_j": 120.0,
            "sampling_interval_ms": 100,
            "competing_cuda_processes": 0,
            "telemetry_evidence_paths": telemetry_paths,
        }
        write_json(run_dir / "run-resource.json", resource)
        source_files = []
        for relative in assembler.AUDIT_SOURCE_PATHS:
            payload = (run_dir / relative).read_bytes()
            source_files.append(
                {"path": relative, "sha256": sha256(payload), "bytes": len(payload)}
            )
        write_json(
            run_dir / "resource-audit.json",
            {
                "schema_version": "qwen3-tts-spark-resource-audit/v1",
                "engine_id": engine,
                "profile_id": profile,
                "round": round_number,
                "phase_boundaries": {
                    "idle_start_wall_time_unix_ns": "1000000000",
                    "idle_end_wall_time_unix_ns": "16000000000",
                    "measured_start_wall_time_unix_ns": "17000000000",
                    "measured_end_wall_time_unix_ns": "19000000000",
                    "idle_duration_seconds": 15.0,
                    "measured_monotonic_duration_seconds": 2.0,
                    "measured_wall_duration_seconds": 2.0,
                },
                "sampling": {
                    "configured_interval_ms": 100,
                    "maximum_allowed_observed_gap_ms": 200,
                    "idle_power_samples": 150,
                    "measured_power_samples": 20,
                    "measured_process_rss_samples": 20,
                    "measured_gpu_process_samples": 20,
                    "measured_system_samples": 20,
                },
                "power": {
                    "source": "NVIDIA board power.draw",
                    "integration": "linear boundary interpolation and trapezoidal integration",
                    "idle_average_power_w": 20.0,
                    "idle_peak_power_w": 25.0,
                    "idle_gross_energy_j": 300.0,
                    "measured_average_power_w": 80.0,
                    "measured_peak_power_w": 100.0,
                    "measured_gross_energy_j": 160.0,
                    "measured_idle_adjusted_energy_j": 120.0,
                    "idle_adjustment": "max(0, measured gross energy - idle mean power * measured wall-clock duration)",
                },
                "memory": {
                    "process_rss_definition": "peak measured-window sum of VmRSS for every extant PID in the target container cgroup",
                    "process_rss_peak_bytes": 1_000_000,
                    "gpu_unified_memory_definition": "peak measured-window sum of NVIDIA used_memory for target-container compute PIDs",
                    "gpu_unified_memory_peak_bytes": 4_294_967_296,
                    "cgroup_memory_definition": "peak measured-window memory.current; distinct from process RSS",
                    "cgroup_memory_current_peak_bytes": 1_500_000,
                    "host_mem_available_min_kib": 100_000,
                    "host_swap_free_start_kib": 200_000,
                    "host_swap_free_end_kib": 200_000,
                },
                "source_files": source_files,
            },
        )
        refresh_checksums(run_dir)

    def assemble(self) -> dict[str, Any]:
        return assembler.assemble_manifest(
            self.config, self.workload, self.runs, self.output
        )

    def add_registry_evidence(self, engine: str) -> None:
        implementation = next(
            item
            for item in self.config_value["implementations"]
            if item["id"] == engine
        )
        registry = {
            "reference": f"ghcr.io/example/{engine}",
            "manifest_digest": "sha256:" + "7" * 64,
            "compressed_size_bytes": 4_000_000_000,
        }
        payload = {
            "schema_version": "qwen3-tts-registry-image/v1",
            "implementation_id": engine,
            "local_image_id": implementation["local_image"]["id"],
            **registry,
        }
        path = self.evidence / "artifacts" / engine / "registry-image.json"
        write_json(path, payload)
        registry["evidence"] = {
            "path": path.relative_to(self.evidence).as_posix(),
            "sha256": sha256(path.read_bytes()),
        }
        implementation["registry_image"] = registry
        write_json(self.config, self.config_value)


class AssembleProductionManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = EvidenceFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_assembles_exact_production_matrix_create_new(self) -> None:
        manifest = self.fixture.assemble()
        self.assertEqual(manifest["schema_version"], "1.2")
        self.assertEqual(manifest["evidence_kind"], "production")
        self.assertEqual(len(manifest["run_resources"]), 12)
        self.assertEqual(
            {
                (item["engine_id"], item["profile_id"], item["round"])
                for item in manifest["run_resources"]
            },
            assembler.EXPECTED_RUN_KEYS,
        )
        self.assertEqual(manifest, json.loads(self.fixture.output.read_text()))
        self.assertIs(report_generator._validate_manifest(manifest), manifest)
        artifacts = {
            item["id"]: item["model_artifact"] for item in manifest["implementations"]
        }
        self.assertNotEqual(
            artifacts["native"]["parameter_count"],
            artifacts["sglang"]["parameter_count"],
        )
        self.assertNotEqual(
            artifacts["native"]["precision"], artifacts["sglang"]["precision"]
        )
        self.assertIsNotNone(artifacts["native"]["manifest_sha256"])
        self.assertIsNone(artifacts["sglang"]["manifest_sha256"])
        descriptors = manifest["evidence_files"]
        self.assertEqual(sum(item["role"] == "workload" for item in descriptors), 1)
        self.assertEqual(
            sum(item["role"] == "client_summary" for item in descriptors), 12
        )
        self.assertTrue(
            any(item["path"].endswith("run-resource.json") for item in descriptors)
        )
        self.assertFalse(any(item["bytes"] == 0 for item in descriptors))
        with self.assertRaisesRegex(assembler.AssemblyError, "refusing to overwrite"):
            self.fixture.assemble()

    def test_rejects_missing_and_extra_runs(self) -> None:
        shutil.rmtree(self.fixture.run_dir("native", "B1", 1))
        with self.assertRaisesRegex(assembler.AssemblyError, "exactly 12"):
            self.fixture.assemble()
        self.assertFalse(self.fixture.output.exists())

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = EvidenceFixture(Path(self.temporary.name))
        shutil.copytree(
            self.fixture.run_dir("native", "B1", 2),
            self.fixture.run_dir("native", "B1", 3),
        )
        with self.assertRaisesRegex(assembler.AssemblyError, "exactly 12"):
            self.fixture.assemble()

    def test_rejects_unowned_failed_run_artifacts(self) -> None:
        failed = self.fixture.runs / "round-01/native/B1.failed.20260717T000000Z.12345"
        failed.mkdir(parents=True)
        (failed / "partial.log").write_text("failed\n", encoding="utf-8")
        with self.assertRaisesRegex(
            assembler.AssemblyError,
            "unexpected directory outside a qualifying run",
        ):
            self.fixture.assemble()
        self.assertFalse(self.fixture.output.exists())

    def test_rejects_checksum_mutation(self) -> None:
        summary = self.fixture.run_dir("native", "B1", 1) / "client/summary.json"
        summary.write_text(summary.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(assembler.AssemblyError, "digest mismatch"):
            self.fixture.assemble()

    def test_rejects_missing_process_rss_sampler_provenance(self) -> None:
        run_dir = self.fixture.run_dir("sglang", "B1", 1)
        (run_dir / "provenance/lib/process-rss-sampler.sh").unlink()
        refresh_checksums(run_dir)
        with self.assertRaisesRegex(
            assembler.AssemblyError,
            "missing qualifying-run files.*process-rss-sampler[.]sh",
        ):
            self.fixture.assemble()

    def test_rejects_missing_server_log_provenance(self) -> None:
        run_dir = self.fixture.run_dir("native", "B1", 1)
        (run_dir / "provenance/server.log").unlink()
        refresh_checksums(run_dir)
        with self.assertRaisesRegex(
            assembler.AssemblyError,
            "missing qualifying-run files.*provenance/server.log",
        ):
            self.fixture.assemble()

    def test_rejects_server_log_window_for_another_container(self) -> None:
        run_dir = self.fixture.run_dir("sglang", "B1", 1)
        window_path = run_dir / "provenance/server-log-window.json"
        window = json.loads(window_path.read_text(encoding="utf-8"))
        window["container"]["id"] = "6" * 64
        write_json(window_path, window)
        refresh_checksums(run_dir)
        with self.assertRaisesRegex(
            assembler.AssemblyError,
            "server-log-window container differs from invocation",
        ):
            self.fixture.assemble()

    def test_rejects_internally_stale_resource_audit_digest(self) -> None:
        run_dir = self.fixture.run_dir("native", "B1", 1)
        audit_path = run_dir / "resource-audit.json"
        audit = json.loads(audit_path.read_text())
        audit["source_files"][0]["sha256"] = "0" * 64
        write_json(audit_path, audit)
        refresh_checksums(run_dir)
        with self.assertRaisesRegex(
            assembler.AssemblyError, "digest or byte count mismatch"
        ):
            self.fixture.assemble()

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_rejects_symlink_anywhere_in_run_tree(self) -> None:
        run_dir = self.fixture.run_dir("native", "B1", 1)
        os.symlink(run_dir / "raw/gpu.csv", run_dir / "raw/alias.csv")
        with self.assertRaisesRegex(assembler.AssemblyError, "symlinks are forbidden"):
            self.fixture.assemble()

    def test_rejects_engine_profile_round_identity_mismatch(self) -> None:
        run_dir = self.fixture.run_dir("native", "B3", 1)
        resource_path = run_dir / "run-resource.json"
        resource = json.loads(resource_path.read_text())
        resource["profile_id"] = "B6"
        write_json(resource_path, resource)
        refresh_checksums(run_dir)
        with self.assertRaisesRegex(assembler.AssemblyError, "identities differ"):
            self.fixture.assemble()

    def test_rejects_noncanonical_workload_duration(self) -> None:
        workload = json.loads(self.fixture.workload.read_text().splitlines()[0])
        workload["max_duration_seconds"] = 20.47
        write_jsonl(self.fixture.workload, [workload])
        with self.assertRaisesRegex(assembler.AssemblyError, "exactly 20.48"):
            self.fixture.assemble()

    def test_rejects_sglang_255_frame_boundary(self) -> None:
        run_dir = self.fixture.run_dir("sglang", "B1", 1)
        requests_path = run_dir / "client/requests.jsonl"
        requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
        requests[0]["samples"] = 255 * 1_920
        requests[0]["audio_seconds"] = 20.4
        write_jsonl(requests_path, requests)
        refresh_checksums(run_dir)
        with self.assertRaisesRegex(assembler.AssemblyError, "255-frame boundary"):
            self.fixture.assemble()

    def test_rejects_configured_image_digest_mismatch(self) -> None:
        config = json.loads(self.fixture.config.read_text())
        config["implementations"][0]["local_image"]["id"] = "sha256:" + "9" * 64
        write_json(self.fixture.config, config)
        with self.assertRaisesRegex(
            assembler.AssemblyError, "local_image_id|local Docker image ID"
        ):
            self.fixture.assemble()

    def test_rejects_ordered_seed_drift(self) -> None:
        config = json.loads(self.fixture.config.read_text())
        config["workload"]["ordered_seeds"] = [43]
        write_json(self.fixture.config, config)
        with self.assertRaisesRegex(assembler.AssemblyError, "ordered_seeds"):
            self.fixture.assemble()

    def test_rejects_missing_stock_artifact_evidence(self) -> None:
        stock = self.fixture.evidence / "artifacts/sglang/model-artifact.json"
        stock.unlink()
        with self.assertRaisesRegex(
            assembler.AssemblyError, "digest-bound evidence is unavailable"
        ):
            self.fixture.assemble()

    def test_rejects_stock_artifact_digest_drift(self) -> None:
        stock = self.fixture.evidence / "artifacts/sglang/model-artifact.json"
        payload = json.loads(stock.read_text())
        payload["weight_files"][1]["sha256"] = "8" * 64
        write_json(stock, payload)
        with self.assertRaisesRegex(assembler.AssemblyError, "declared .* observed"):
            self.fixture.assemble()

    def test_rejects_artifact_parameter_sum_mismatch(self) -> None:
        config = json.loads(self.fixture.config.read_text())
        config["implementations"][0]["model_artifact"]["parameter_count"] += 1
        write_json(self.fixture.config, config)
        with self.assertRaisesRegex(
            assembler.AssemblyError, "parameter_count|digest-bound artifact evidence"
        ):
            self.fixture.assemble()

    def test_rejects_local_image_size_mismatch(self) -> None:
        config = json.loads(self.fixture.config.read_text())
        config["implementations"][1]["local_image"]["unpacked_size_bytes"] += 1
        write_json(self.fixture.config, config)
        with self.assertRaisesRegex(
            assembler.AssemblyError, "image-inspect Size|local unpacked size"
        ):
            self.fixture.assemble()

    def test_optional_registry_metadata_is_separate_and_evidence_bound(self) -> None:
        self.fixture.add_registry_evidence("native")
        manifest = self.fixture.assemble()
        native = next(
            item for item in manifest["implementations"] if item["id"] == "native"
        )
        self.assertNotEqual(
            native["local_image"]["id"], native["registry_image"]["manifest_digest"]
        )
        self.assertNotEqual(
            native["local_image"]["unpacked_size_bytes"],
            native["registry_image"]["compressed_size_bytes"],
        )
        self.assertEqual(
            sum(
                item["role"] == "registry_metadata"
                for item in manifest["evidence_files"]
            ),
            1,
        )

    def test_rejects_registry_metadata_drift(self) -> None:
        self.fixture.add_registry_evidence("native")
        config = json.loads(self.fixture.config.read_text())
        config["implementations"][0]["registry_image"]["compressed_size_bytes"] += 1
        write_json(self.fixture.config, config)
        with self.assertRaisesRegex(
            assembler.AssemblyError, "compressed_size_bytes.*digest-bound"
        ):
            self.fixture.assemble()


if __name__ == "__main__":
    unittest.main()
