from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPORTS_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPORTS_DIR / "fixtures" / "layout_only"
MODULE_PATH = REPORTS_DIR / "generate_report.py"
SPEC = importlib.util.spec_from_file_location("generate_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generate_report
SPEC.loader.exec_module(generate_report)


def update_descriptor(manifest_path: Path, relative_path: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = (manifest_path.parent / relative_path).read_bytes()
    for descriptor in manifest["evidence_files"]:
        if descriptor["path"] == relative_path:
            descriptor["bytes"] = len(payload)
            descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
            if descriptor["role"] == "requests":
                manifest["workload"]["corpus_sha256"] = descriptor["sha256"]
            break
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(quantile: float) -> float:
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )


def descriptor(role: str, path: Path, base: Path, **identity: Any) -> dict[str, Any]:
    payload = path.read_bytes()
    result = {
        "role": role,
        **identity,
        "path": path.relative_to(base).as_posix(),
        "format": path.suffix[1:],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    return result


def create_client_bundle(base: Path) -> Path:
    workload = [
        {
            "id": "fixture-001",
            "text": "A calm benchmark sentence.",
            "voice_description": "A calm adult voice.",
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
        },
        {
            "id": "fixture-002",
            "text": "Eine ruhige Benchmark-Zeile.",
            "voice_description": "Eine ruhige erwachsene Stimme.",
            "language": "German",
            "seed": 43,
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
        },
    ]
    workload_path = base / "workload.jsonl"
    write_jsonl(workload_path, workload)
    evidence = [descriptor("workload", workload_path, base)]
    resources = []
    profiles = (("B1", 1), ("B3", 3), ("B6", 6))
    for engine in ("native", "sglang"):
        for profile_id, width in profiles:
            run_dir = base / "runs" / engine / profile_id / "round-1"
            request_rows: list[dict[str, Any]] = []
            packet_rows: list[dict[str, Any]] = []
            for index, entry in enumerate(workload):
                normalized = {
                    "contract": "qwen3-tts-native-sglang-common/v1",
                    "seed": entry["seed"],
                    "talker": {
                        "strategy": "sample",
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "top_k": 50,
                        "repetition_penalty": 1.05,
                    },
                    "predictor": {
                        "strategy": "sample",
                        "temperature": 0.9,
                        "top_p": 1.0,
                        "top_k": 50,
                    },
                }
                native = engine == "native"
                ttfa_ms = (100.0 if native else 700.0) + index
                wall_ms = (500.0 if native else 800.0) + index
                request_rows.append(
                    {
                        "schema_version": "qwen3-tts-http-bench/v1",
                        "request_index": index,
                        "workload_id": entry["id"],
                        "backend": "native" if native else "sglang-omni",
                        "text_sha256": sha256_text(entry["text"]),
                        "voice_description_sha256": sha256_text(
                            entry["voice_description"]
                        ),
                        "request_body_sha256": sha256_text(f"{engine}-{index}"),
                        "normalized_sampling": normalized,
                        "normalized_sampling_sha256": canonical_sha256(normalized),
                        "sampling_parity_qualifying": True,
                        "sampling_parity_non_qualifying_reasons": [],
                        "language": entry["language"],
                        "streaming": True,
                        "success": True,
                        "http_status": 200,
                        "server_request_id": f"{engine}-{profile_id}-{index}",
                        "server_seed": None,
                        "ttfa_ms": ttfa_ms,
                        "wall_ms": wall_ms,
                        "sample_rate_hz": 24000,
                        "samples": 24000,
                        "audio_sha256": sha256_text(
                            f"audio-{engine}-{profile_id}-{index}"
                        ),
                        "audio_seconds": 1.0,
                        "rtf": wall_ms / 1000.0,
                        "response_bytes": 50000,
                        "packet_count": 2 if native else 1,
                        "continuity_valid": True,
                        "final_flag_seen": True if native else None,
                        "finish_reason": "stop" if native else None,
                        "natural_eos": True if native else None,
                        "length_limited": False if native else None,
                        "end_metrics": {} if native else None,
                        "failure": None,
                    }
                )
                packet_count = 2 if native else 1
                for sequence in range(packet_count):
                    payload_bytes = 24000 if native else 48000
                    arrival_ms = ttfa_ms + sequence * 100.0
                    packet_rows.append(
                        {
                            "schema_version": "qwen3-tts-http-bench/v1",
                            "request_index": index,
                            "workload_id": entry["id"],
                            "backend": "native" if native else "sglang-omni",
                            "kind": "native_audio_packet"
                            if native
                            else "raw_pcm_transport_arrival",
                            "sequence": sequence,
                            "arrival_ms": arrival_ms,
                            "inter_arrival_ms": None if sequence == 0 else 100.0,
                            "payload_bytes": payload_bytes,
                            "payload_sha256": sha256_text(
                                f"packet-{engine}-{profile_id}-{index}-{sequence}"
                            ),
                            "byte_offset": sequence * payload_bytes,
                            "first_codec_frame": sequence * 6 if native else None,
                            "first_sample": sequence * 12000 if native else None,
                            "sample_count": 12000 if native else None,
                            "codec_frames": 6 if native else None,
                            "is_first": sequence == 0,
                            "is_final": sequence == packet_count - 1
                            if native
                            else None,
                        }
                    )
            wall_values = [row["wall_ms"] for row in request_rows]
            ttfa_values = [row["ttfa_ms"] for row in request_rows]
            rtf_values = [row["rtf"] for row in request_rows]
            benchmark_wall_seconds = 1.2 if engine == "native" else 2.0
            summary = {
                "schema_version": "qwen3-tts-http-bench/v1",
                "endpoint": "http://127.0.0.1:8080/v1/voice-design/speech"
                if engine == "native"
                else "http://127.0.0.1:8000/v1/audio/speech",
                "backend": "native" if engine == "native" else "sglang-omni",
                "sglang_model": None
                if engine == "native"
                else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                "concurrency": profile_id,
                "synchronized_batch_width": width,
                "warmups": 2,
                "planned_requests": 2,
                "completed_requests": 2,
                "successful_requests": 2,
                "failed_requests": 0,
                "natural_eos_requests": 2 if engine == "native" else 0,
                "length_limited_requests": 0,
                "eos_unknown_requests": 0 if engine == "native" else 2,
                "sampling_parity_qualifying_requests": 2,
                "sampling_parity_non_qualifying_requests": 0,
                "normalized_sampling_sha256s": sorted(
                    {row["normalized_sampling_sha256"] for row in request_rows}
                ),
                "benchmark_wall_seconds": benchmark_wall_seconds,
                "attempted_requests_per_second": 2 / benchmark_wall_seconds,
                "throughput_requests_per_second": 2 / benchmark_wall_seconds,
                "total_audio_seconds": 2.0,
                "aggregate_rtf": benchmark_wall_seconds / 2.0,
                "summed_request_wall_rtf": sum(wall_values) / 1000.0 / 2.0,
                "ttfa_ms": distribution(ttfa_values),
                "wall_ms": distribution(wall_values),
                "request_rtf": distribution(rtf_values),
            }
            requests_path = run_dir / "requests.jsonl"
            packets_path = run_dir / "packets.jsonl"
            summary_path = run_dir / "summary.json"
            telemetry_path = run_dir / "gpu.csv"
            write_jsonl(requests_path, request_rows)
            write_jsonl(packets_path, packet_rows)
            write_json(summary_path, summary)
            telemetry_path.write_text("timestamp,power_w\n0,100\n", encoding="utf-8")
            identity = {"engine_id": engine, "profile_id": profile_id, "round": 1}
            evidence.extend(
                [
                    descriptor("client_requests", requests_path, base, **identity),
                    descriptor("client_packets", packets_path, base, **identity),
                    descriptor("client_summary", summary_path, base, **identity),
                    descriptor("raw", telemetry_path, base, **identity),
                ]
            )
            resources.append(
                {
                    **identity,
                    "process_rss_peak_bytes": 1_000_000_000
                    if engine == "native"
                    else 4_000_000_000,
                    "gpu_unified_memory_peak_bytes": 5_000_000_000
                    if engine == "native"
                    else 15_000_000_000,
                    "average_power_w": 100.0 if engine == "native" else 150.0,
                    "peak_power_w": 120.0 if engine == "native" else 180.0,
                    "energy_j": 120.0 if engine == "native" else 300.0,
                    "sampling_interval_ms": 200,
                    "competing_cuda_processes": 0,
                    "telemetry_evidence_paths": [
                        telemetry_path.relative_to(base).as_posix()
                    ],
                }
            )
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.1"
    manifest["workload"] = {
        "corpus_sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
        "seed": 42,
        "sample_rate_hz": 24000,
        "channels": 1,
        "sample_format": "pcm_s16le",
        "response_mode": "streaming",
        "warmup_requests_per_engine": 2,
        "minimum_measured_requests_per_profile": 2,
        "minimum_rounds_per_subject": 1,
        "profiles": [
            {"id": profile_id, "concurrency": width, "repetitions_per_request": 1}
            for profile_id, width in profiles
        ],
        "language_policy": "Synthetic bilingual direct client fixture",
    }
    manifest["model"]["repository"] = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    for implementation in manifest["implementations"]:
        implementation["model_repository"] = manifest["model"]["repository"]
    manifest["evidence_files"] = evidence
    manifest["run_resources"] = resources
    manifest_path = base / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


class ReportPipelineTests(unittest.TestCase):
    def test_production_workload_accepts_exact_duration_contract(self) -> None:
        records = generate_report._validate_client_workload(
            [
                {
                    "id": "duration-contract",
                    "text": "A duration contract fixture.",
                    "voice_description": "A calm adult voice.",
                    "max_duration_seconds": 20.48,
                }
            ]
        )
        generate_report._validate_production_workload_durations(
            records, "workload.jsonl"
        )

    def test_production_workload_rejects_missing_or_different_duration(self) -> None:
        for duration in (None, 20.4, 20.480_001):
            record = {
                "id": "duration-contract",
                "text": "A duration contract fixture.",
                "voice_description": "A calm adult voice.",
            }
            if duration is not None:
                record["max_duration_seconds"] = duration
            records = generate_report._validate_client_workload([record])
            with (
                self.subTest(duration=duration),
                self.assertRaisesRegex(
                    generate_report.EvidenceError,
                    r"max_duration_seconds.*requires exactly 20\.48 seconds",
                ),
            ):
                generate_report._validate_production_workload_durations(
                    records, "workload.jsonl"
                )

    def test_production_sglang_accepts_audio_strictly_below_boundary(self) -> None:
        samples = generate_report.SGLANG_EXCLUSIVE_SAMPLE_LIMIT - 1
        generate_report._validate_production_sglang_audio_limit(
            {
                "samples": samples,
                "audio_seconds": samples / generate_report.PRODUCTION_SAMPLE_RATE_HZ,
            },
            [{"payload_bytes": samples * 2}],
            generate_report.PRODUCTION_SAMPLE_RATE_HZ,
            "requests.jsonl:1",
        )

    def test_production_sglang_rejects_255_frame_boundary(self) -> None:
        samples = generate_report.SGLANG_EXCLUSIVE_SAMPLE_LIMIT
        with self.assertRaisesRegex(
            generate_report.EvidenceError,
            r"strictly shorter than 255 codec frames.*489600 samples.*20\.40 seconds",
        ):
            generate_report._validate_production_sglang_audio_limit(
                {
                    "samples": samples,
                    "audio_seconds": samples
                    / generate_report.PRODUCTION_SAMPLE_RATE_HZ,
                },
                [{"payload_bytes": samples * 2}],
                generate_report.PRODUCTION_SAMPLE_RATE_HZ,
                "requests.jsonl:1",
            )

    def test_production_sglang_rejects_inconsistent_audio_bytes(self) -> None:
        samples = 24_000
        with self.assertRaisesRegex(
            generate_report.EvidenceError,
            "audio payload bytes do not match the validated request sample count",
        ):
            generate_report._validate_production_sglang_audio_limit(
                {"samples": samples, "audio_seconds": 1.0},
                [{"payload_bytes": (samples - 1) * 2}],
                generate_report.PRODUCTION_SAMPLE_RATE_HZ,
                "requests.jsonl:1",
            )

    def test_production_sglang_rejects_inconsistent_audio_duration(self) -> None:
        samples = 24_000
        with self.assertRaisesRegex(
            generate_report.EvidenceError,
            "audio duration does not match the decoded PCM sample count",
        ):
            generate_report._validate_production_sglang_audio_limit(
                {"samples": samples, "audio_seconds": 0.999},
                [{"payload_bytes": samples * 2}],
                generate_report.PRODUCTION_SAMPLE_RATE_HZ,
                "requests.jsonl:1",
            )

    def test_direct_client_bundle_validates_and_uses_scenario_rtf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = create_client_bundle(Path(temporary))
            bundle = generate_report.load_bundle(manifest_path, allow_test_fixture=True)
            self.assertEqual(len(bundle.measurements["native"]), 6)
            self.assertEqual(len(bundle.measurements["sglang"]), 6)
            aggregates = generate_report.aggregate(bundle)
            self.assertAlmostEqual(aggregates["native"]["B3"]["aggregate_rtf"], 0.6)
            self.assertAlmostEqual(
                aggregates["native"]["B3"]["summed_request_wall_rtf"], 0.5005
            )
            self.assertAlmostEqual(
                aggregates["native"]["B3"]["request_throughput_rps"], 2 / 1.2
            )
            self.assertAlmostEqual(
                aggregates["native"]["B3"]["attempted_throughput_rps"], 2 / 1.2
            )
            self.assertEqual(aggregates["native"]["B3"]["natural_eos"], 2)
            self.assertEqual(aggregates["sglang"]["B3"]["eos_unknown"], 2)

    def test_direct_client_incorrect_aggregate_rtf_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = create_client_bundle(Path(temporary))
            summary_path = (
                Path(temporary) / "runs" / "native" / "B1" / "round-1" / "summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["aggregate_rtf"] = 999.0
            write_json(summary_path, summary)
            update_descriptor(
                manifest_path, summary_path.relative_to(Path(temporary)).as_posix()
            )
            with self.assertRaisesRegex(
                generate_report.EvidenceError, "aggregate_rtf.*expected"
            ):
                generate_report.load_bundle(manifest_path, allow_test_fixture=True)

    def test_direct_client_native_non_natural_eos_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest_path = create_client_bundle(base)
            requests_path = (
                base / "runs" / "native" / "B1" / "round-1" / "requests.jsonl"
            )
            rows = [
                json.loads(line)
                for line in requests_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["natural_eos"] = False
            rows[0]["length_limited"] = True
            rows[0]["finish_reason"] = "length"
            write_jsonl(requests_path, rows)
            summary_path = base / "runs" / "native" / "B1" / "round-1" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["natural_eos_requests"] = 1
            summary["length_limited_requests"] = 1
            write_json(summary_path, summary)
            update_descriptor(manifest_path, requests_path.relative_to(base).as_posix())
            update_descriptor(manifest_path, summary_path.relative_to(base).as_posix())
            with self.assertRaisesRegex(
                generate_report.EvidenceError,
                "Native successful requests must all be natural EOS",
            ):
                generate_report.load_bundle(manifest_path, allow_test_fixture=True)

    def test_direct_client_sglang_imputed_eos_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest_path = create_client_bundle(base)
            requests_path = (
                base / "runs" / "sglang" / "B1" / "round-1" / "requests.jsonl"
            )
            rows = [
                json.loads(line)
                for line in requests_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["natural_eos"] = True
            rows[0]["length_limited"] = False
            rows[0]["finish_reason"] = "stop"
            write_jsonl(requests_path, rows)
            summary_path = base / "runs" / "sglang" / "B1" / "round-1" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["natural_eos_requests"] = 1
            summary["eos_unknown_requests"] = 1
            write_json(summary_path, summary)
            update_descriptor(manifest_path, requests_path.relative_to(base).as_posix())
            update_descriptor(manifest_path, summary_path.relative_to(base).as_posix())
            with self.assertRaisesRegex(
                generate_report.EvidenceError,
                "stock SGLang successful requests must retain eos_unknown",
            ):
                generate_report.load_bundle(manifest_path, allow_test_fixture=True)

    @unittest.skipIf(
        generate_report.REPORTLAB_IMPORT_ERROR is not None, "ReportLab unavailable"
    )
    def test_direct_client_fixture_render_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest_path = create_client_bundle(base)
            bundle = generate_report.load_bundle(manifest_path, allow_test_fixture=True)
            first = base / "first.pdf"
            second = base / "second.pdf"
            generate_report.build_pdf(bundle, first)
            generate_report.build_pdf(bundle, second)
            self.assertTrue(first.read_bytes().startswith(b"%PDF-"))
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).digest(),
                hashlib.sha256(second.read_bytes()).digest(),
            )

    def test_fixture_is_rejected_without_explicit_flag(self) -> None:
        with self.assertRaisesRegex(
            generate_report.EvidenceError, "--allow-test-fixture"
        ):
            generate_report.load_bundle(FIXTURE_DIR / "manifest.json")

    def test_fixture_validates_with_explicit_flag(self) -> None:
        bundle = generate_report.load_bundle(
            FIXTURE_DIR / "manifest.json",
            allow_test_fixture=True,
        )
        self.assertEqual(len(bundle.requests), 2)
        self.assertEqual(len(bundle.measurements["native"]), 6)
        self.assertEqual(len(bundle.measurements["sglang"]), 6)
        aggregates = generate_report.aggregate(bundle)
        self.assertEqual(set(aggregates["native"]), {"B1", "B3", "B6"})

    def test_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "fixture"
            shutil.copytree(FIXTURE_DIR, copied)
            target = copied / "requests.jsonl"
            payload = target.read_bytes()
            target.write_bytes(b"[" + payload[1:])
            with self.assertRaisesRegex(
                generate_report.EvidenceError, "sha256.*declared .* observed"
            ):
                generate_report.load_bundle(
                    copied / "manifest.json", allow_test_fixture=True
                )

    def test_unequal_engine_workload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "fixture"
            shutil.copytree(FIXTURE_DIR, copied)
            target = copied / "measurements.sglang.jsonl"
            lines = target.read_text(encoding="utf-8").splitlines()
            target.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            update_descriptor(copied / "manifest.json", "measurements.sglang.jsonl")
            with self.assertRaisesRegex(
                generate_report.EvidenceError, "unequal workload keys"
            ):
                generate_report.load_bundle(
                    copied / "manifest.json", allow_test_fixture=True
                )

    def test_model_revision_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "fixture"
            shutil.copytree(FIXTURE_DIR, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["implementations"][1]["model_revision"] = "different-revision"
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                generate_report.EvidenceError, "must exactly match model.revision"
            ):
                generate_report.load_bundle(manifest_path, allow_test_fixture=True)

    def test_fixture_cannot_use_release_output_directory(self) -> None:
        bundle = generate_report.load_bundle(
            FIXTURE_DIR / "manifest.json",
            allow_test_fixture=True,
        )
        target = REPORTS_DIR / "output" / "forbidden-fixture.pdf"
        with self.assertRaisesRegex(generate_report.EvidenceError, "cannot be written"):
            generate_report.resolve_output(bundle, target, overwrite=False)

    @unittest.skipIf(
        generate_report.REPORTLAB_IMPORT_ERROR is not None, "ReportLab unavailable"
    )
    def test_fixture_render_is_deterministic(self) -> None:
        bundle = generate_report.load_bundle(
            FIXTURE_DIR / "manifest.json",
            allow_test_fixture=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.pdf"
            second = Path(temporary) / "second.pdf"
            generate_report.build_pdf(bundle, first)
            generate_report.build_pdf(bundle, second)
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()
            self.assertTrue(first_bytes.startswith(b"%PDF-"))
            self.assertGreater(len(first_bytes), 10_000)
            self.assertEqual(
                hashlib.sha256(first_bytes).digest(),
                hashlib.sha256(second_bytes).digest(),
            )


if __name__ == "__main__":
    unittest.main()
