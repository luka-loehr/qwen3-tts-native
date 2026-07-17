#!/usr/bin/env python3
"""Validate benchmark evidence and build a deterministic monochrome PDF report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any, Iterable, Sequence

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.platypus import (
        BaseDocTemplate,
        Flowable,
        Frame,
        KeepTogether,
        LongTable,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # Validation remains available without PDF dependencies.
    REPORTLAB_IMPORT_ERROR = exc
    Flowable = object  # type: ignore[assignment,misc]


REPORTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPORTS_DIR / "output"
TMP_PDF_DIR = REPORTS_DIR / "tmp" / "pdfs"
SCHEMA_PATH = REPORTS_DIR / "evidence.schema.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BENCHMARK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
PROFILE_ID_RE = re.compile(r"^B[0-9]+$")
FORBIDDEN_DASHES = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)
TEST_FIXTURE_BANNER = "TEST FIXTURE - NOT BENCHMARK EVIDENCE"
PRODUCTION_SAMPLE_RATE_HZ = 24_000
PRODUCTION_MAX_DURATION_SECONDS = 20.48
SGLANG_BOUNDARY_CODEC_FRAMES = 255
SAMPLES_PER_CODEC_FRAME = 1_920
SGLANG_EXCLUSIVE_SAMPLE_LIMIT = SGLANG_BOUNDARY_CODEC_FRAMES * SAMPLES_PER_CODEC_FRAME
SGLANG_EXCLUSIVE_DURATION_LIMIT_SECONDS = (
    SGLANG_EXCLUSIVE_SAMPLE_LIMIT / PRODUCTION_SAMPLE_RATE_HZ
)


class EvidenceError(ValueError):
    """Raised when any structural or semantic evidence check fails."""


@dataclass(frozen=True)
class Bundle:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    requests: tuple[dict[str, Any], ...]
    measurements: dict[str, tuple[dict[str, Any], ...]]
    evidence_payloads: dict[str, Any]
    run_summaries: dict[tuple[str, str, int], dict[str, Any]]
    run_resources: dict[tuple[str, str, int], dict[str, Any]]


def _fail(path: str, message: str) -> None:
    raise EvidenceError(f"{path}: {message}")


def _strict_object(
    value: Any,
    path: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")
    if extra:
        _fail(path, f"unrecognized fields: {', '.join(extra)}")
    return value


def _string(value: Any, path: str, minimum: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        _fail(path, f"expected a non-empty string of at least {minimum} characters")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
        _fail(path, "contains a control character")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"expected an integer greater than or equal to {minimum}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        _fail(path, f"expected a finite number greater than or equal to {minimum}")
    return result


def _nullable_number(value: Any, path: str, minimum: float = 0.0) -> float | None:
    if value is None:
        return None
    return _number(value, path, minimum)


def _sha256(value: Any, path: str) -> str:
    result = _string(value, path)
    if not SHA256_RE.fullmatch(result):
        _fail(path, "expected a lowercase SHA-256 digest")
    return result


def _string_list(value: Any, path: str, minimum_items: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum_items:
        _fail(path, f"expected an array with at least {minimum_items} entries")
    for index, item in enumerate(value):
        _string(item, f"{path}[{index}]", 2)
    return value


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise EvidenceError(f"non-finite JSON number is forbidden: {token}")


def _parse_json_bytes(payload: bytes, path: str) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"{path}: expected UTF-8: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, EvidenceError) as exc:
        raise EvidenceError(f"{path}: invalid JSON: {exc}") from exc


def _parse_jsonl_bytes(payload: bytes, path: str) -> list[Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"{path}: expected UTF-8: {exc}") from exc
    records: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            _fail(f"{path}:{line_number}", "blank JSONL lines are forbidden")
        try:
            record = json.loads(
                line,
                object_pairs_hook=_unique_object_pairs,
                parse_constant=_reject_nonfinite,
            )
        except (json.JSONDecodeError, EvidenceError) as exc:
            raise EvidenceError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        records.append(record)
    if not records:
        _fail(path, "JSONL evidence must contain at least one record")
    return records


def _file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_report(value: Any) -> dict[str, Any]:
    report = _strict_object(
        value, "report", ("benchmark_id", "title", "generated_at", "authors")
    )
    benchmark_id = _string(report["benchmark_id"], "report.benchmark_id")
    if not BENCHMARK_ID_RE.fullmatch(benchmark_id):
        _fail("report.benchmark_id", "must match [a-z0-9][a-z0-9._-]{2,63}")
    _string(report["title"], "report.title", 12)
    timestamp = _string(report["generated_at"], "report.generated_at")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError("report.generated_at: expected RFC 3339 date-time") from exc
    if parsed.tzinfo is None:
        _fail("report.generated_at", "timezone offset is required")
    _string_list(report["authors"], "report.authors", 1)
    return report


def _validate_system(value: Any) -> dict[str, Any]:
    fields = (
        "host_model",
        "hostname_alias",
        "os",
        "kernel",
        "architecture",
        "cpu",
        "accelerator",
        "driver_version",
        "cuda_version",
        "physical_unified_memory_bytes",
        "power_measurement_source",
        "notes",
    )
    system = _strict_object(value, "system", fields)
    for field in fields[:9]:
        _string(system[field], f"system.{field}")
    _integer(
        system["physical_unified_memory_bytes"],
        "system.physical_unified_memory_bytes",
        1,
    )
    _string(system["power_measurement_source"], "system.power_measurement_source", 3)
    _string_list(system["notes"], "system.notes")
    return system


def _validate_model(value: Any) -> dict[str, Any]:
    model = _strict_object(
        value,
        "model",
        (
            "repository",
            "revision",
            "variant",
            "parameter_count",
            "precision",
            "manifest_sha256",
            "weight_files",
        ),
    )
    _string(model["repository"], "model.repository", 3)
    _string(model["revision"], "model.revision", 7)
    _string(model["variant"], "model.variant", 3)
    _integer(model["parameter_count"], "model.parameter_count", 1)
    _string(model["precision"], "model.precision", 2)
    _sha256(model["manifest_sha256"], "model.manifest_sha256")
    weights = model["weight_files"]
    if not isinstance(weights, list) or not weights:
        _fail("model.weight_files", "expected at least one weight file")
    seen: set[str] = set()
    for index, value in enumerate(weights):
        item = _strict_object(
            value, f"model.weight_files[{index}]", ("path", "sha256", "bytes")
        )
        path = _string(item["path"], f"model.weight_files[{index}].path")
        if path in seen:
            _fail(f"model.weight_files[{index}].path", "duplicate path")
        seen.add(path)
        _sha256(item["sha256"], f"model.weight_files[{index}].sha256")
        _integer(item["bytes"], f"model.weight_files[{index}].bytes", 1)
    return model


def _validate_workload(
    value: Any,
    evidence_kind: str,
    schema_version: str,
) -> dict[str, Any]:
    legacy_fields = ("voice_description_sha256", "generation")
    workload = _strict_object(
        value,
        "workload",
        (
            "corpus_sha256",
            "seed",
            "sample_rate_hz",
            "channels",
            "sample_format",
            "response_mode",
            "warmup_requests_per_engine",
            "minimum_measured_requests_per_profile",
            "profiles",
            "language_policy",
        )
        + (
            legacy_fields
            if schema_version == "1.0"
            else ("minimum_rounds_per_subject",)
        ),
    )
    _sha256(workload["corpus_sha256"], "workload.corpus_sha256")
    _integer(workload["seed"], "workload.seed")
    _integer(workload["sample_rate_hz"], "workload.sample_rate_hz", 1)
    _integer(workload["channels"], "workload.channels", 1)
    if workload["sample_format"] != "pcm_s16le":
        _fail("workload.sample_format", "only pcm_s16le is supported")
    if workload["response_mode"] not in {"streaming", "buffered"}:
        _fail("workload.response_mode", "expected streaming or buffered")
    warmups = _integer(
        workload["warmup_requests_per_engine"],
        "workload.warmup_requests_per_engine",
    )
    if evidence_kind == "production" and warmups < 24:
        _fail(
            "workload.warmup_requests_per_engine",
            "production evidence requires at least 24",
        )
    minimum = _integer(
        workload["minimum_measured_requests_per_profile"],
        "workload.minimum_measured_requests_per_profile",
        1,
    )
    if evidence_kind == "production" and minimum < 200:
        _fail(
            "workload.minimum_measured_requests_per_profile",
            "production evidence requires at least 200",
        )
    if schema_version == "1.1":
        rounds = _integer(
            workload["minimum_rounds_per_subject"],
            "workload.minimum_rounds_per_subject",
            1,
        )
        if evidence_kind == "production" and rounds < 2:
            _fail(
                "workload.minimum_rounds_per_subject",
                "production evidence requires at least two rounds",
            )
    _string(workload["language_policy"], "workload.language_policy", 4)
    profiles = workload["profiles"]
    if not isinstance(profiles, list) or not profiles:
        _fail("workload.profiles", "expected at least one profile")
    profile_ids: set[str] = set()
    concurrency_values: set[int] = set()
    for index, value in enumerate(profiles):
        profile = _strict_object(
            value,
            f"workload.profiles[{index}]",
            ("id", "concurrency", "repetitions_per_request"),
        )
        profile_id = _string(profile["id"], f"workload.profiles[{index}].id")
        concurrency = _integer(
            profile["concurrency"], f"workload.profiles[{index}].concurrency", 1
        )
        _integer(
            profile["repetitions_per_request"],
            f"workload.profiles[{index}].repetitions_per_request",
            1,
        )
        if not PROFILE_ID_RE.fullmatch(profile_id) or profile_id != f"B{concurrency}":
            _fail(f"workload.profiles[{index}]", "profile id must equal B<concurrency>")
        if profile_id in profile_ids or concurrency in concurrency_values:
            _fail(f"workload.profiles[{index}]", "duplicate profile id or concurrency")
        profile_ids.add(profile_id)
        concurrency_values.add(concurrency)
    if evidence_kind == "production" and {
        item["id"]: item["concurrency"] for item in profiles
    } != {
        "B1": 1,
        "B3": 3,
        "B6": 6,
    }:
        _fail(
            "workload.profiles", "production evidence requires exactly B1, B3, and B6"
        )
    if schema_version == "1.0":
        _sha256(
            workload["voice_description_sha256"], "workload.voice_description_sha256"
        )
        generation = _strict_object(
            workload["generation"],
            "workload.generation",
            ("temperature", "top_p", "top_k", "max_codec_frames"),
        )
        if _number(generation["temperature"], "workload.generation.temperature") <= 0:
            _fail("workload.generation.temperature", "must be greater than zero")
        top_p = _number(generation["top_p"], "workload.generation.top_p")
        if top_p <= 0 or top_p > 1:
            _fail(
                "workload.generation.top_p", "must be greater than zero and at most one"
            )
        _integer(generation["top_k"], "workload.generation.top_k", 1)
        _integer(
            generation["max_codec_frames"], "workload.generation.max_codec_frames", 1
        )
    return workload


def _validate_implementations(
    value: Any, model: dict[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 2:
        _fail("implementations", "expected exactly Native and SGLang")
    required = (
        "id",
        "role",
        "name",
        "version",
        "source_commit",
        "source_url",
        "container_image",
        "image_digest",
        "image_size_bytes",
        "startup_ms",
        "model_repository",
        "model_revision",
        "model_precision",
        "model_manifest_sha256",
        "api_protocol",
        "streaming_semantics",
        "runtime_components",
        "command_sha256",
    )
    roles: set[str] = set()
    for index, raw in enumerate(value):
        item = _strict_object(raw, f"implementations[{index}]", required)
        role = item["role"]
        if item["id"] not in {"native", "sglang"} or role not in {"native", "sglang"}:
            _fail(f"implementations[{index}]", "id and role must be native or sglang")
        if item["id"] != role:
            _fail(f"implementations[{index}]", "id and role must be identical")
        if role in roles:
            _fail(f"implementations[{index}].role", "duplicate implementation role")
        roles.add(role)
        for field in (
            "name",
            "version",
            "source_commit",
            "source_url",
            "container_image",
            "api_protocol",
            "streaming_semantics",
        ):
            _string(item[field], f"implementations[{index}].{field}")
        digest = _string(item["image_digest"], f"implementations[{index}].image_digest")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            _fail(
                f"implementations[{index}].image_digest",
                "expected sha256:<64 lowercase hex>",
            )
        _integer(
            item["image_size_bytes"], f"implementations[{index}].image_size_bytes", 1
        )
        _number(item["startup_ms"], f"implementations[{index}].startup_ms")
        _string_list(
            item["runtime_components"],
            f"implementations[{index}].runtime_components",
            1,
        )
        _sha256(item["command_sha256"], f"implementations[{index}].command_sha256")
        _sha256(
            item["model_manifest_sha256"],
            f"implementations[{index}].model_manifest_sha256",
        )
        comparisons = (
            ("model_repository", "repository"),
            ("model_revision", "revision"),
            ("model_precision", "precision"),
            ("model_manifest_sha256", "manifest_sha256"),
        )
        for implementation_field, model_field in comparisons:
            if item[implementation_field] != model[model_field]:
                _fail(
                    f"implementations[{index}].{implementation_field}",
                    f"must exactly match model.{model_field}",
                )
    if roles != {"native", "sglang"}:
        _fail("implementations", "both native and sglang roles are required")
    return value


def _validate_methodology(value: Any) -> dict[str, Any]:
    required = (
        "clock_source",
        "ttfa_definition",
        "rtf_definition",
        "throughput_definition",
        "memory_definition",
        "power_definition",
        "energy_definition",
        "startup_definition",
        "sampling_interval_ms",
        "run_order",
        "statistical_method",
        "environment_controls",
    )
    methodology = _strict_object(value, "methodology", required)
    for field in required[:8] + required[9:11]:
        _string(methodology[field], f"methodology.{field}", 8)
    _integer(methodology["sampling_interval_ms"], "methodology.sampling_interval_ms", 1)
    _string_list(
        methodology["environment_controls"], "methodology.environment_controls", 1
    )
    return methodology


def _validate_run_resources(value: Any, evidence_kind: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail("run_resources", "schema 1.1 requires one resource record per client run")
    required = (
        "engine_id",
        "profile_id",
        "round",
        "process_rss_peak_bytes",
        "gpu_unified_memory_peak_bytes",
        "average_power_w",
        "peak_power_w",
        "energy_j",
        "sampling_interval_ms",
        "competing_cuda_processes",
        "telemetry_evidence_paths",
    )
    for index, raw in enumerate(value):
        path = f"run_resources[{index}]"
        item = _strict_object(raw, path, required)
        if item["engine_id"] not in {"native", "sglang"}:
            _fail(f"{path}.engine_id", "expected native or sglang")
        profile_id = _string(item["profile_id"], f"{path}.profile_id")
        if not PROFILE_ID_RE.fullmatch(profile_id):
            _fail(f"{path}.profile_id", "expected B<concurrency>")
        _integer(item["round"], f"{path}.round", 1)
        if (
            _number(item["process_rss_peak_bytes"], f"{path}.process_rss_peak_bytes")
            <= 0
        ):
            _fail(f"{path}.process_rss_peak_bytes", "must be greater than zero")
        if (
            _number(
                item["gpu_unified_memory_peak_bytes"],
                f"{path}.gpu_unified_memory_peak_bytes",
            )
            <= 0
        ):
            _fail(f"{path}.gpu_unified_memory_peak_bytes", "must be greater than zero")
        average_power = _number(item["average_power_w"], f"{path}.average_power_w")
        peak_power = _number(item["peak_power_w"], f"{path}.peak_power_w")
        if peak_power < average_power:
            _fail(
                f"{path}.peak_power_w", "must be greater than or equal to average power"
            )
        _number(item["energy_j"], f"{path}.energy_j")
        interval = _integer(
            item["sampling_interval_ms"], f"{path}.sampling_interval_ms", 1
        )
        competing = _integer(
            item["competing_cuda_processes"],
            f"{path}.competing_cuda_processes",
        )
        if evidence_kind == "production" and interval > 200:
            _fail(
                f"{path}.sampling_interval_ms",
                "production evidence requires 200 ms or faster sampling",
            )
        if evidence_kind == "production" and competing != 0:
            _fail(
                f"{path}.competing_cuda_processes",
                "production evidence forbids competing CUDA processes",
            )
        paths = _string_list(
            item["telemetry_evidence_paths"], f"{path}.telemetry_evidence_paths", 1
        )
        if len(paths) != len(set(paths)):
            _fail(f"{path}.telemetry_evidence_paths", "contains duplicate paths")
    return value


def _validate_manifest(value: Any) -> dict[str, Any]:
    manifest = _strict_object(
        value,
        "manifest",
        (
            "schema_version",
            "evidence_kind",
            "report",
            "system",
            "model",
            "workload",
            "implementations",
            "methodology",
            "evidence_files",
            "limitations",
        ),
        ("test_fixture_notice", "run_resources"),
    )
    if manifest["schema_version"] not in {"1.0", "1.1"}:
        _fail("schema_version", "expected version 1.0 or 1.1")
    kind = manifest["evidence_kind"]
    if kind not in {"production", "test_fixture"}:
        _fail("evidence_kind", "expected production or test_fixture")
    if kind == "test_fixture":
        notice = _string(manifest.get("test_fixture_notice"), "test_fixture_notice", 20)
        if TEST_FIXTURE_BANNER not in notice:
            _fail("test_fixture_notice", f"must contain '{TEST_FIXTURE_BANNER}'")
    elif "test_fixture_notice" in manifest:
        _fail("test_fixture_notice", "is forbidden for production evidence")
    if kind == "production" and manifest["schema_version"] != "1.1":
        _fail(
            "schema_version",
            "production evidence requires the direct client-run schema 1.1",
        )
    if manifest["schema_version"] == "1.0" and kind != "test_fixture":
        _fail(
            "schema_version",
            "legacy normalized schema 1.0 is restricted to test fixtures",
        )
    _validate_report(manifest["report"])
    _validate_system(manifest["system"])
    model = _validate_model(manifest["model"])
    _validate_workload(manifest["workload"], kind, manifest["schema_version"])
    _validate_implementations(manifest["implementations"], model)
    _validate_methodology(manifest["methodology"])
    limitations = _string_list(manifest["limitations"], "limitations", 1)
    for index, limitation in enumerate(limitations):
        if len(limitation.strip()) < 8:
            _fail(f"limitations[{index}]", "must contain at least eight characters")
    if (
        not isinstance(manifest["evidence_files"], list)
        or len(manifest["evidence_files"]) < 3
    ):
        _fail("evidence_files", "expected a non-empty evidence inventory")
    if manifest["schema_version"] == "1.1":
        _validate_run_resources(manifest.get("run_resources"), kind)
    elif "run_resources" in manifest:
        _fail("run_resources", "is only valid with schema 1.1")
    return manifest


def _resolve_evidence_file(base: Path, relative: str, path: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or not pure.parts:
        _fail(path, "must be a normalized relative path without traversal")
    if pure.suffix not in {
        ".json",
        ".jsonl",
        ".csv",
        ".txt",
        ".log",
        ".stdout",
        ".stderr",
    }:
        _fail(path, "unsupported evidence extension")
    current = base
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            _fail(path, "symlinks are forbidden")
    resolved_base = base.resolve(strict=True)
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvidenceError(
            f"{path}: evidence file does not exist: {relative}"
        ) from exc
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise EvidenceError(
            f"{path}: evidence file escapes the bundle directory"
        ) from exc
    if not resolved.is_file():
        _fail(path, "evidence path is not a regular file")
    return resolved


def _load_evidence_files(
    manifest: dict[str, Any], base: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payloads: dict[str, Any] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    request_descriptors: list[dict[str, Any]] = []
    measurement_engines: set[str] = set()
    for index, raw in enumerate(manifest["evidence_files"]):
        descriptor = _strict_object(
            raw,
            f"evidence_files[{index}]",
            ("role", "path", "format", "sha256", "bytes"),
            ("engine_id",),
        )
        role = descriptor["role"]
        if role not in {"requests", "measurements", "raw"}:
            _fail(
                f"evidence_files[{index}].role",
                "expected requests, measurements, or raw",
            )
        relative = _string(descriptor["path"], f"evidence_files[{index}].path")
        if relative in descriptors:
            _fail(f"evidence_files[{index}].path", "duplicate evidence path")
        format_name = descriptor["format"]
        if format_name not in {"json", "jsonl"}:
            _fail(f"evidence_files[{index}].format", "expected json or jsonl")
        if not relative.endswith(f".{format_name}"):
            _fail(
                f"evidence_files[{index}].format", "does not match the file extension"
            )
        expected_digest = _sha256(
            descriptor["sha256"], f"evidence_files[{index}].sha256"
        )
        expected_bytes = _integer(
            descriptor["bytes"], f"evidence_files[{index}].bytes", 1
        )
        engine = descriptor.get("engine_id")
        if role == "requests":
            if engine is not None:
                _fail(
                    f"evidence_files[{index}].engine_id",
                    "requests must not be engine-specific",
                )
            request_descriptors.append(descriptor)
        elif role == "measurements":
            if engine not in {"native", "sglang"}:
                _fail(
                    f"evidence_files[{index}].engine_id",
                    "measurement engine is required",
                )
            if engine in measurement_engines:
                _fail(
                    f"evidence_files[{index}].engine_id", "duplicate measurement engine"
                )
            measurement_engines.add(engine)
        elif engine is not None and engine not in {"native", "sglang"}:
            _fail(f"evidence_files[{index}].engine_id", "unknown raw evidence engine")
        resolved = _resolve_evidence_file(
            base, relative, f"evidence_files[{index}].path"
        )
        payload = resolved.read_bytes()
        if len(payload) != expected_bytes:
            _fail(
                f"evidence_files[{index}].bytes",
                f"declared {expected_bytes}, observed {len(payload)}",
            )
        observed_digest = _file_sha256(payload)
        if observed_digest != expected_digest:
            _fail(
                f"evidence_files[{index}].sha256",
                f"declared {expected_digest}, observed {observed_digest}",
            )
        parsed = (
            _parse_jsonl_bytes(payload, relative)
            if format_name == "jsonl"
            else _parse_json_bytes(payload, relative)
        )
        payloads[relative] = parsed
        descriptors[relative] = descriptor
    if len(request_descriptors) != 1:
        _fail("evidence_files", "exactly one requests file is required")
    if measurement_engines != {"native", "sglang"}:
        _fail("evidence_files", "exactly one measurement file per engine is required")
    requests_descriptor = request_descriptors[0]
    if requests_descriptor["sha256"] != manifest["workload"]["corpus_sha256"]:
        _fail("workload.corpus_sha256", "must equal the canonical requests file digest")
    return payloads, descriptors


def _validate_requests(
    records: Any, workload: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    if not isinstance(records, list) or not records:
        _fail("requests", "expected at least one request record")
    required = (
        "request_id",
        "prompt_sha256",
        "text_sha256",
        "voice_description_sha256",
        "language",
        "text_characters",
        "text_utf8_bytes",
        "seed",
    )
    identifiers: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        item = _strict_object(raw, f"requests[{index}]", required)
        request_id = _string(item["request_id"], f"requests[{index}].request_id")
        if request_id in identifiers:
            _fail(f"requests[{index}].request_id", "duplicate request id")
        identifiers.add(request_id)
        _sha256(item["prompt_sha256"], f"requests[{index}].prompt_sha256")
        _sha256(item["text_sha256"], f"requests[{index}].text_sha256")
        voice_digest = _sha256(
            item["voice_description_sha256"],
            f"requests[{index}].voice_description_sha256",
        )
        if voice_digest != workload["voice_description_sha256"]:
            _fail(
                f"requests[{index}].voice_description_sha256",
                "does not match the workload",
            )
        _string(item["language"], f"requests[{index}].language", 2)
        _integer(item["text_characters"], f"requests[{index}].text_characters", 1)
        _integer(item["text_utf8_bytes"], f"requests[{index}].text_utf8_bytes", 1)
        _integer(item["seed"], f"requests[{index}].seed")
        validated.append(item)
    return tuple(validated)


SUCCESS_FIELDS = (
    "ttfa_ms",
    "total_latency_ms",
    "audio_duration_ms",
    "rtf",
    "packet_count",
    "packet_intervals_ms",
    "output_bytes",
)


def _validate_measurement(
    raw: Any,
    path: str,
    engine: str,
    request_ids: set[str],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = (
        "engine_id",
        "request_id",
        "profile_id",
        "repeat",
        "status",
        "http_status",
        "ttfa_ms",
        "total_latency_ms",
        "audio_duration_ms",
        "rtf",
        "packet_count",
        "packet_intervals_ms",
        "output_bytes",
        "process_rss_peak_bytes",
        "gpu_unified_memory_peak_bytes",
        "average_power_w",
        "peak_power_w",
        "energy_j",
        "run_elapsed_start_ms",
        "run_elapsed_end_ms",
    )
    item = _strict_object(raw, path, required)
    if item["engine_id"] != engine:
        _fail(f"{path}.engine_id", f"must equal descriptor engine {engine}")
    request_id = _string(item["request_id"], f"{path}.request_id")
    if request_id not in request_ids:
        _fail(f"{path}.request_id", "does not exist in the canonical corpus")
    profile_id = _string(item["profile_id"], f"{path}.profile_id")
    if profile_id not in profiles:
        _fail(f"{path}.profile_id", "unknown workload profile")
    repeat = _integer(item["repeat"], f"{path}.repeat", 1)
    if repeat > profiles[profile_id]["repetitions_per_request"]:
        _fail(f"{path}.repeat", "exceeds repetitions_per_request")
    status = item["status"]
    if status not in {"success", "error", "timeout", "cancelled"}:
        _fail(f"{path}.status", "expected success, error, timeout, or cancelled")
    if item["http_status"] is not None:
        http_status = _integer(item["http_status"], f"{path}.http_status", 100)
        if http_status > 599:
            _fail(f"{path}.http_status", "must be between 100 and 599")
    start = _number(item["run_elapsed_start_ms"], f"{path}.run_elapsed_start_ms")
    end = _number(item["run_elapsed_end_ms"], f"{path}.run_elapsed_end_ms")
    if end <= start:
        _fail(f"{path}.run_elapsed_end_ms", "must be greater than the start offset")
    rss = _number(item["process_rss_peak_bytes"], f"{path}.process_rss_peak_bytes")
    gpu_memory = _number(
        item["gpu_unified_memory_peak_bytes"],
        f"{path}.gpu_unified_memory_peak_bytes",
    )
    if rss <= 0 or gpu_memory <= 0:
        _fail(path, "resource peaks must be greater than zero")
    average_power = _number(item["average_power_w"], f"{path}.average_power_w")
    peak_power = _number(item["peak_power_w"], f"{path}.peak_power_w")
    if peak_power < average_power:
        _fail(f"{path}.peak_power_w", "must be greater than or equal to average power")
    _number(item["energy_j"], f"{path}.energy_j")
    if status == "success":
        if item["http_status"] is None or not 200 <= item["http_status"] <= 299:
            _fail(f"{path}.http_status", "successful measurements require a 2xx status")
        ttfa = _number(item["ttfa_ms"], f"{path}.ttfa_ms")
        latency = _number(item["total_latency_ms"], f"{path}.total_latency_ms")
        duration = _number(item["audio_duration_ms"], f"{path}.audio_duration_ms")
        if duration <= 0:
            _fail(f"{path}.audio_duration_ms", "must be greater than zero")
        rtf = _number(item["rtf"], f"{path}.rtf")
        if ttfa > latency:
            _fail(f"{path}.ttfa_ms", "cannot exceed total latency")
        observed_rtf = latency / duration
        if not math.isclose(rtf, observed_rtf, rel_tol=0.01, abs_tol=0.005):
            _fail(
                f"{path}.rtf",
                f"does not match latency/audio duration ({observed_rtf:.6f})",
            )
        elapsed = end - start
        if not math.isclose(latency, elapsed, rel_tol=0.02, abs_tol=5.0):
            _fail(
                f"{path}.total_latency_ms",
                f"does not match run offsets ({elapsed:.3f} ms)",
            )
        packet_count = _integer(item["packet_count"], f"{path}.packet_count", 1)
        intervals = item["packet_intervals_ms"]
        if not isinstance(intervals, list) or len(intervals) != packet_count - 1:
            _fail(
                f"{path}.packet_intervals_ms", "must contain packet_count - 1 intervals"
            )
        for index, interval in enumerate(intervals):
            _number(interval, f"{path}.packet_intervals_ms[{index}]")
        _integer(item["output_bytes"], f"{path}.output_bytes", 1)
    else:
        for field in SUCCESS_FIELDS:
            if item[field] is not None:
                _fail(f"{path}.{field}", "must be null when status is not success")
    return item


def _validate_measurements(
    payloads: dict[str, Any],
    descriptors: dict[str, dict[str, Any]],
    requests: tuple[dict[str, Any], ...],
    workload: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], ...]]:
    request_ids = {item["request_id"] for item in requests}
    profiles = {item["id"]: item for item in workload["profiles"]}
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    expected_keys = {
        (profile_id, request_id, repeat)
        for profile_id, profile in profiles.items()
        for request_id in request_ids
        for repeat in range(1, profile["repetitions_per_request"] + 1)
    }
    minimum = workload["minimum_measured_requests_per_profile"]
    for relative, descriptor in descriptors.items():
        if descriptor["role"] != "measurements":
            continue
        engine = descriptor["engine_id"]
        records = payloads[relative]
        if not isinstance(records, list) or not records:
            _fail(relative, "expected a non-empty JSONL measurement array")
        validated: list[dict[str, Any]] = []
        observed_keys: set[tuple[str, str, int]] = set()
        for index, raw in enumerate(records):
            item = _validate_measurement(
                raw,
                f"{relative}:{index + 1}",
                engine,
                request_ids,
                profiles,
            )
            key = (item["profile_id"], item["request_id"], item["repeat"])
            if key in observed_keys:
                _fail(f"{relative}:{index + 1}", f"duplicate measurement key {key}")
            observed_keys.add(key)
            validated.append(item)
        missing = expected_keys - observed_keys
        extra = observed_keys - expected_keys
        if missing or extra:
            sample_missing = sorted(missing)[:3]
            sample_extra = sorted(extra)[:3]
            _fail(
                relative,
                f"unequal workload keys; missing={sample_missing} extra={sample_extra}",
            )
        for profile_id in profiles:
            count = sum(1 for item in validated if item["profile_id"] == profile_id)
            if count < minimum:
                _fail(
                    relative,
                    f"profile {profile_id} has {count} rows, below required {minimum}",
                )
        result[engine] = tuple(validated)
    if set(result) != {"native", "sglang"}:
        _fail("measurements", "both engine measurement sets are required")
    native_keys = {
        (item["profile_id"], item["request_id"], item["repeat"])
        for item in result["native"]
    }
    sglang_keys = {
        (item["profile_id"], item["request_id"], item["repeat"])
        for item in result["sglang"]
    }
    if native_keys != sglang_keys:
        _fail("measurements", "Native and SGLang workload keys differ")
    return result


CLIENT_SCHEMA_VERSION = "qwen3-tts-http-bench/v1"
CLIENT_ROLES = {"client_summary", "client_requests", "client_packets"}


def _load_client_evidence_files(
    manifest: dict[str, Any],
    base: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payloads: dict[str, Any] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    workload_descriptors: list[dict[str, Any]] = []
    client_keys: dict[str, set[tuple[str, str, int]]] = {
        role: set() for role in CLIENT_ROLES
    }
    allowed_formats = {"json", "jsonl", "csv", "txt", "log", "stdout", "stderr"}
    for index, raw in enumerate(manifest["evidence_files"]):
        path = f"evidence_files[{index}]"
        descriptor = _strict_object(
            raw,
            path,
            ("role", "path", "format", "sha256", "bytes"),
            ("engine_id", "profile_id", "round"),
        )
        role = descriptor["role"]
        if role not in {"workload", "raw"} | CLIENT_ROLES:
            _fail(
                f"{path}.role",
                "expected workload, client_summary, client_requests, client_packets, or raw",
            )
        relative = _string(descriptor["path"], f"{path}.path")
        if relative in descriptors:
            _fail(f"{path}.path", "duplicate evidence path")
        format_name = descriptor["format"]
        if format_name not in allowed_formats:
            _fail(f"{path}.format", "unsupported evidence format")
        if not relative.endswith(f".{format_name}"):
            _fail(f"{path}.format", "does not match the file extension")
        expected_digest = _sha256(descriptor["sha256"], f"{path}.sha256")
        expected_bytes = _integer(descriptor["bytes"], f"{path}.bytes", 1)
        engine = descriptor.get("engine_id")
        profile_id = descriptor.get("profile_id")
        round_number = descriptor.get("round")
        if role == "workload":
            if any(value is not None for value in (engine, profile_id, round_number)):
                _fail(
                    path,
                    "the shared workload must not declare engine, profile, or round",
                )
            if format_name != "jsonl":
                _fail(f"{path}.format", "the benchmark workload must be JSONL")
            workload_descriptors.append(descriptor)
        elif role in CLIENT_ROLES:
            if engine not in {"native", "sglang"}:
                _fail(f"{path}.engine_id", "client evidence requires native or sglang")
            if profile_id not in {
                item["id"] for item in manifest["workload"]["profiles"]
            }:
                _fail(
                    f"{path}.profile_id",
                    "client evidence references an unknown profile",
                )
            round_number = _integer(round_number, f"{path}.round", 1)
            expected_format = "json" if role == "client_summary" else "jsonl"
            if format_name != expected_format:
                _fail(f"{path}.format", f"{role} requires {expected_format}")
            key = (engine, profile_id, round_number)
            if key in client_keys[role]:
                _fail(path, f"duplicate {role} for {key}")
            client_keys[role].add(key)
        else:
            specified = [
                engine is not None,
                profile_id is not None,
                round_number is not None,
            ]
            if any(specified) and not all(specified):
                _fail(
                    path,
                    "run-specific raw evidence requires engine, profile, and round together",
                )
            if all(specified):
                if engine not in {"native", "sglang"}:
                    _fail(f"{path}.engine_id", "expected native or sglang")
                if profile_id not in {
                    item["id"] for item in manifest["workload"]["profiles"]
                }:
                    _fail(
                        f"{path}.profile_id",
                        "raw evidence references an unknown profile",
                    )
                _integer(round_number, f"{path}.round", 1)
        resolved = _resolve_evidence_file(base, relative, f"{path}.path")
        payload = resolved.read_bytes()
        if len(payload) != expected_bytes:
            _fail(
                f"{path}.bytes", f"declared {expected_bytes}, observed {len(payload)}"
            )
        observed_digest = _file_sha256(payload)
        if observed_digest != expected_digest:
            _fail(
                f"{path}.sha256",
                f"declared {expected_digest}, observed {observed_digest}",
            )
        if format_name == "jsonl":
            parsed: Any = _parse_jsonl_bytes(payload, relative)
        elif format_name == "json":
            parsed = _parse_json_bytes(payload, relative)
        else:
            parsed = payload
        payloads[relative] = parsed
        descriptors[relative] = descriptor
    if len(workload_descriptors) != 1:
        _fail("evidence_files", "schema 1.1 requires exactly one shared workload JSONL")
    if workload_descriptors[0]["sha256"] != manifest["workload"]["corpus_sha256"]:
        _fail("workload.corpus_sha256", "must equal the shared workload file digest")
    if not client_keys["client_summary"]:
        _fail("evidence_files", "at least one client run is required")
    if not all(keys == client_keys["client_summary"] for keys in client_keys.values()):
        _fail(
            "evidence_files",
            "every client run requires one summary, requests, and packets artifact",
        )
    expected_profiles = {item["id"] for item in manifest["workload"]["profiles"]}
    minimum_rounds = manifest["workload"]["minimum_rounds_per_subject"]
    for engine in ("native", "sglang"):
        for profile_id in expected_profiles:
            rounds = sorted(
                key[2]
                for key in client_keys["client_summary"]
                if key[:2] == (engine, profile_id)
            )
            if len(rounds) < minimum_rounds:
                _fail(
                    "evidence_files",
                    f"{engine} {profile_id} has {len(rounds)} rounds, below required {minimum_rounds}",
                )
            if rounds != list(range(1, len(rounds) + 1)):
                _fail(
                    "evidence_files",
                    f"{engine} {profile_id} rounds must be contiguous from 1",
                )
    for profile_id in expected_profiles:
        native_rounds = {
            key[2]
            for key in client_keys["client_summary"]
            if key[:2] == ("native", profile_id)
        }
        sglang_rounds = {
            key[2]
            for key in client_keys["client_summary"]
            if key[:2] == ("sglang", profile_id)
        }
        if native_rounds != sglang_rounds:
            _fail("evidence_files", f"Native and SGLang rounds differ for {profile_id}")
    return payloads, descriptors


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected a boolean")
    return value


def _validate_sampling_stage(value: Any, path: str, predictor: bool) -> dict[str, Any]:
    required = ("strategy", "temperature", "top_p", "top_k")
    if not predictor:
        required += ("repetition_penalty",)
    stage = _strict_object(value, path, required)
    if stage["strategy"] not in {"sample", "greedy"}:
        _fail(f"{path}.strategy", "expected sample or greedy")
    nullable_fields = ("temperature", "top_p", "top_k")
    if not predictor:
        nullable_fields += ("repetition_penalty",)
    for field in nullable_fields:
        item = stage[field]
        if item is None:
            continue
        if field == "top_k":
            _integer(item, f"{path}.{field}", 1)
        elif _number(item, f"{path}.{field}") <= 0:
            _fail(f"{path}.{field}", "must be greater than zero")
        elif field == "top_p" and item > 1:
            _fail(f"{path}.{field}", "must be at most one")
    if stage["strategy"] == "greedy" and any(
        stage[field] is not None for field in ("temperature", "top_p", "top_k")
    ):
        _fail(path, "greedy sampling must not declare temperature, top_p, or top_k")
    return stage


def _validate_normalized_sampling(value: Any, path: str) -> dict[str, Any]:
    normalized = _strict_object(
        value, path, ("contract", "seed", "talker", "predictor")
    )
    if normalized["contract"] != "qwen3-tts-native-sglang-common/v1":
        _fail(f"{path}.contract", "unsupported normalized sampling contract")
    if normalized["seed"] is not None:
        _integer(normalized["seed"], f"{path}.seed")
    if normalized["talker"] is not None:
        _validate_sampling_stage(
            normalized["talker"], f"{path}.talker", predictor=False
        )
    if normalized["predictor"] is not None:
        _validate_sampling_stage(
            normalized["predictor"], f"{path}.predictor", predictor=True
        )
    return normalized


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_client_workload(records: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(records, list) or not records:
        _fail("workload", "expected at least one benchmark workload record")
    validated: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(records):
        path = f"workload:{index + 1}"
        item = _strict_object(
            raw,
            path,
            ("id", "text", "voice_description"),
            ("language", "seed", "max_duration_seconds", "sampling", "stream"),
        )
        identifier = _string(item["id"], f"{path}.id")
        if len(identifier) > 128 or not re.fullmatch(r"[A-Za-z0-9._-]+", identifier):
            _fail(
                f"{path}.id",
                "expected at most 128 ASCII letters, digits, '.', '_', or '-'",
            )
        if identifier in identifiers:
            _fail(f"{path}.id", "duplicate workload id")
        identifiers.add(identifier)
        _string(item["text"], f"{path}.text")
        _string(item["voice_description"], f"{path}.voice_description")
        _string(item.get("language", "auto"), f"{path}.language")
        if item.get("seed") is not None:
            _integer(item["seed"], f"{path}.seed")
        if (
            item.get("max_duration_seconds") is not None
            and _number(item["max_duration_seconds"], f"{path}.max_duration_seconds")
            <= 0
        ):
            _fail(f"{path}.max_duration_seconds", "must be greater than zero")
        if "stream" in item:
            _boolean(item["stream"], f"{path}.stream")
        if "sampling" in item:
            sampling = _strict_object(
                item["sampling"],
                f"{path}.sampling",
                (),
                (
                    "strategy",
                    "temperature",
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                    "predictor",
                ),
            )
            strategy = sampling.get("strategy", "sample")
            if strategy not in {"sample", "greedy"}:
                _fail(f"{path}.sampling.strategy", "expected sample or greedy")
            if "predictor" in sampling:
                predictor = _strict_object(
                    sampling["predictor"],
                    f"{path}.sampling.predictor",
                    (),
                    ("strategy", "temperature", "top_p", "top_k"),
                )
                if predictor.get("strategy", "sample") not in {"sample", "greedy"}:
                    _fail(
                        f"{path}.sampling.predictor.strategy",
                        "expected sample or greedy",
                    )
        validated.append(item)
    return tuple(validated)


def _validate_production_workload_durations(
    records: tuple[dict[str, Any], ...],
    workload_path: str,
) -> None:
    for index, item in enumerate(records):
        path = f"{workload_path}:{index + 1}.max_duration_seconds"
        duration = item.get("max_duration_seconds")
        if duration is None:
            _fail(
                path,
                f"production comparison requires exactly {PRODUCTION_MAX_DURATION_SECONDS} seconds",
            )
        if _number(duration, path) != PRODUCTION_MAX_DURATION_SECONDS:
            _fail(
                path,
                f"production comparison requires exactly {PRODUCTION_MAX_DURATION_SECONDS} seconds",
            )


def _validate_production_sglang_audio_limit(
    request: dict[str, Any],
    packets: Sequence[dict[str, Any]],
    sample_rate_hz: int,
    path: str,
) -> None:
    if sample_rate_hz != PRODUCTION_SAMPLE_RATE_HZ:
        _fail(
            "workload.sample_rate_hz",
            f"production comparison requires exactly {PRODUCTION_SAMPLE_RATE_HZ} Hz",
        )

    audio_bytes = sum(packet["payload_bytes"] for packet in packets)
    if audio_bytes % 2 != 0:
        _fail(path, "SGLang PCM16 audio payload has an odd byte count")
    samples_from_audio_bytes = audio_bytes // 2
    if samples_from_audio_bytes != request["samples"]:
        _fail(
            path,
            "SGLang audio payload bytes do not match the validated request sample count",
        )

    samples_from_duration = request["audio_seconds"] * sample_rate_hz
    if not math.isclose(
        samples_from_duration,
        float(samples_from_audio_bytes),
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        _fail(
            path,
            "SGLang request audio duration does not match the decoded PCM sample count",
        )

    if (
        samples_from_audio_bytes >= SGLANG_EXCLUSIVE_SAMPLE_LIMIT
        or request["audio_seconds"] >= SGLANG_EXCLUSIVE_DURATION_LIMIT_SECONDS
    ):
        _fail(
            path,
            "successful stock SGLang audio must be strictly shorter than "
            f"{SGLANG_BOUNDARY_CODEC_FRAMES} codec frames "
            f"({SGLANG_EXCLUSIVE_SAMPLE_LIMIT} samples / "
            f"{SGLANG_EXCLUSIVE_DURATION_LIMIT_SECONDS:.2f} seconds at "
            f"{PRODUCTION_SAMPLE_RATE_HZ} Hz) to exclude the max_new_tokens boundary",
        )


def _normalized_sampling_from_workload(item: dict[str, Any]) -> dict[str, Any]:
    sampling = item.get("sampling")
    talker = None
    predictor = None
    if sampling is not None:
        talker = {
            "strategy": sampling.get("strategy", "sample"),
            "temperature": sampling.get("temperature"),
            "top_p": sampling.get("top_p"),
            "top_k": sampling.get("top_k"),
            "repetition_penalty": sampling.get("repetition_penalty"),
        }
        if sampling.get("predictor") is not None:
            predictor_source = sampling["predictor"]
            predictor = {
                "strategy": predictor_source.get("strategy", "sample"),
                "temperature": predictor_source.get("temperature"),
                "top_p": predictor_source.get("top_p"),
                "top_k": predictor_source.get("top_k"),
            }
    return {
        "contract": "qwen3-tts-native-sglang-common/v1",
        "seed": item.get("seed"),
        "talker": talker,
        "predictor": predictor,
    }


CLIENT_REQUEST_REQUIRED = (
    "schema_version",
    "request_index",
    "workload_id",
    "backend",
    "text_sha256",
    "voice_description_sha256",
    "request_body_sha256",
    "normalized_sampling",
    "normalized_sampling_sha256",
    "sampling_parity_qualifying",
    "sampling_parity_non_qualifying_reasons",
    "language",
    "streaming",
    "success",
    "http_status",
    "server_request_id",
    "server_seed",
    "ttfa_ms",
    "wall_ms",
    "sample_rate_hz",
    "samples",
    "audio_sha256",
    "audio_seconds",
    "rtf",
    "response_bytes",
    "packet_count",
    "continuity_valid",
    "final_flag_seen",
    "finish_reason",
    "natural_eos",
    "length_limited",
    "end_metrics",
    "failure",
)


def _nullable_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _nullable_boolean(value: Any, path: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, path)


def _validate_client_failure(value: Any, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _strict_object(
        value,
        path,
        ("code", "message", "response_body_bytes", "response_body_sha256"),
    )
    _string(item["code"], f"{path}.code")
    _string(item["message"], f"{path}.message")
    if item["response_body_bytes"] is not None:
        _integer(item["response_body_bytes"], f"{path}.response_body_bytes")
    if item["response_body_sha256"] is not None:
        _sha256(item["response_body_sha256"], f"{path}.response_body_sha256")
    return item


def _validate_client_request(
    raw: Any,
    path: str,
    engine: str,
    workload: tuple[dict[str, Any], ...],
    sample_rate_hz: int,
) -> dict[str, Any]:
    item = _strict_object(
        raw, path, CLIENT_REQUEST_REQUIRED, ("text", "voice_description")
    )
    if item["schema_version"] != CLIENT_SCHEMA_VERSION:
        _fail(f"{path}.schema_version", f"expected {CLIENT_SCHEMA_VERSION}")
    request_index = _integer(item["request_index"], f"{path}.request_index")
    expected = workload[request_index % len(workload)]
    expected_backend = "native" if engine == "native" else "sglang-omni"
    if item["backend"] != expected_backend:
        _fail(f"{path}.backend", f"expected {expected_backend}")
    if item["workload_id"] != expected["id"]:
        _fail(f"{path}.workload_id", "does not match deterministic workload order")
    expected_text_hash = hashlib.sha256(expected["text"].encode("utf-8")).hexdigest()
    expected_voice_hash = hashlib.sha256(
        expected["voice_description"].encode("utf-8")
    ).hexdigest()
    if _sha256(item["text_sha256"], f"{path}.text_sha256") != expected_text_hash:
        _fail(f"{path}.text_sha256", "does not match the workload text")
    if (
        _sha256(item["voice_description_sha256"], f"{path}.voice_description_sha256")
        != expected_voice_hash
    ):
        _fail(
            f"{path}.voice_description_sha256",
            "does not match the workload voice description",
        )
    _sha256(item["request_body_sha256"], f"{path}.request_body_sha256")
    normalized = _validate_normalized_sampling(
        item["normalized_sampling"], f"{path}.normalized_sampling"
    )
    expected_normalized = _normalized_sampling_from_workload(expected)
    if normalized != expected_normalized:
        _fail(f"{path}.normalized_sampling", "does not match the benchmark workload")
    observed_sampling_hash = _sha256(
        item["normalized_sampling_sha256"],
        f"{path}.normalized_sampling_sha256",
    )
    expected_sampling_hash = _canonical_json_sha256(normalized)
    if observed_sampling_hash != expected_sampling_hash:
        _fail(
            f"{path}.normalized_sampling_sha256",
            f"does not match normalized_sampling ({expected_sampling_hash})",
        )
    parity = _boolean(
        item["sampling_parity_qualifying"], f"{path}.sampling_parity_qualifying"
    )
    reasons = _string_list(
        item["sampling_parity_non_qualifying_reasons"],
        f"{path}.sampling_parity_non_qualifying_reasons",
    )
    if parity == bool(reasons):
        _fail(path, "sampling parity qualification and reasons are inconsistent")
    expected_language = expected.get("language", "auto")
    if item["language"] != expected_language:
        _fail(f"{path}.language", "does not match the workload")
    expected_streaming = expected.get("stream", True)
    if _boolean(item["streaming"], f"{path}.streaming") != expected_streaming:
        _fail(f"{path}.streaming", "does not match the workload")
    success = _boolean(item["success"], f"{path}.success")
    if item["http_status"] is not None:
        status = _integer(item["http_status"], f"{path}.http_status", 100)
        if status > 599:
            _fail(f"{path}.http_status", "must be between 100 and 599")
    _nullable_string(item["server_request_id"], f"{path}.server_request_id")
    _nullable_string(item["server_seed"], f"{path}.server_seed")
    for field in ("ttfa_ms", "wall_ms", "audio_seconds", "rtf"):
        _nullable_number(item[field], f"{path}.{field}")
    for field in ("sample_rate_hz", "samples", "response_bytes"):
        if item[field] is not None:
            _integer(item[field], f"{path}.{field}")
    if item["audio_sha256"] is not None:
        _sha256(item["audio_sha256"], f"{path}.audio_sha256")
    packet_count = _integer(item["packet_count"], f"{path}.packet_count")
    continuity = _boolean(item["continuity_valid"], f"{path}.continuity_valid")
    _nullable_boolean(item["final_flag_seen"], f"{path}.final_flag_seen")
    _nullable_string(item["finish_reason"], f"{path}.finish_reason")
    _nullable_boolean(item["natural_eos"], f"{path}.natural_eos")
    _nullable_boolean(item["length_limited"], f"{path}.length_limited")
    if item["end_metrics"] is not None and not isinstance(item["end_metrics"], dict):
        _fail(f"{path}.end_metrics", "expected an object or null")
    failure = _validate_client_failure(item["failure"], f"{path}.failure")
    if "text" in item and item["text"] != expected["text"]:
        _fail(f"{path}.text", "does not match the workload")
    if (
        "voice_description" in item
        and item["voice_description"] != expected["voice_description"]
    ):
        _fail(f"{path}.voice_description", "does not match the workload")
    if success:
        if item["http_status"] is None or not 200 <= item["http_status"] <= 299:
            _fail(f"{path}.http_status", "successful requests require a 2xx response")
        required_success = (
            "ttfa_ms",
            "wall_ms",
            "sample_rate_hz",
            "samples",
            "audio_sha256",
            "audio_seconds",
            "rtf",
            "response_bytes",
        )
        for field in required_success:
            if item[field] is None:
                _fail(f"{path}.{field}", "successful requests require this field")
        if item["sample_rate_hz"] != sample_rate_hz:
            _fail(f"{path}.sample_rate_hz", "does not match the manifest sample rate")
        if item["samples"] <= 0 or item["audio_seconds"] <= 0 or packet_count <= 0:
            _fail(path, "successful requests require non-empty audio and packets")
        if not continuity:
            _fail(
                f"{path}.continuity_valid",
                "successful requests require validated continuity",
            )
        if failure is not None:
            _fail(f"{path}.failure", "must be null for successful requests")
        expected_audio_seconds = item["samples"] / sample_rate_hz
        if not math.isclose(
            item["audio_seconds"], expected_audio_seconds, rel_tol=1e-9, abs_tol=1e-9
        ):
            _fail(f"{path}.audio_seconds", "does not match samples/sample_rate_hz")
        expected_rtf = (item["wall_ms"] / 1000.0) / item["audio_seconds"]
        if not math.isclose(item["rtf"], expected_rtf, rel_tol=1e-9, abs_tol=1e-9):
            _fail(
                f"{path}.rtf",
                f"does not match wall/audio duration ({expected_rtf:.9f})",
            )
        if item["ttfa_ms"] > item["wall_ms"]:
            _fail(f"{path}.ttfa_ms", "cannot exceed wall time")
        if engine == "native":
            if item["final_flag_seen"] is not True:
                _fail(
                    f"{path}.final_flag_seen",
                    "Native streaming success requires the final packet flag",
                )
            if item["end_metrics"] is None:
                _fail(
                    f"{path}.end_metrics",
                    "Native streaming success requires end metrics",
                )
        elif item["final_flag_seen"] is not None or item["end_metrics"] is not None:
            _fail(path, "SGLang raw PCM must not invent final flags or end metrics")
    elif failure is None:
        _fail(f"{path}.failure", "failed requests require failure metadata")
    return item


CLIENT_PACKET_REQUIRED = (
    "schema_version",
    "request_index",
    "workload_id",
    "backend",
    "kind",
    "sequence",
    "arrival_ms",
    "inter_arrival_ms",
    "payload_bytes",
    "payload_sha256",
    "byte_offset",
    "first_codec_frame",
    "first_sample",
    "sample_count",
    "codec_frames",
    "is_first",
    "is_final",
)


def _validate_client_packet(raw: Any, path: str, engine: str) -> dict[str, Any]:
    item = _strict_object(raw, path, CLIENT_PACKET_REQUIRED)
    if item["schema_version"] != CLIENT_SCHEMA_VERSION:
        _fail(f"{path}.schema_version", f"expected {CLIENT_SCHEMA_VERSION}")
    _integer(item["request_index"], f"{path}.request_index")
    _string(item["workload_id"], f"{path}.workload_id")
    expected_backend = "native" if engine == "native" else "sglang-omni"
    expected_kind = (
        "native_audio_packet" if engine == "native" else "raw_pcm_transport_arrival"
    )
    if item["backend"] != expected_backend:
        _fail(f"{path}.backend", f"expected {expected_backend}")
    if item["kind"] != expected_kind:
        _fail(f"{path}.kind", f"expected {expected_kind}")
    _integer(item["sequence"], f"{path}.sequence")
    _number(item["arrival_ms"], f"{path}.arrival_ms")
    _nullable_number(item["inter_arrival_ms"], f"{path}.inter_arrival_ms")
    if _integer(item["payload_bytes"], f"{path}.payload_bytes", 1) % 2 != 0:
        _fail(f"{path}.payload_bytes", "PCM16 payload length must be even")
    _sha256(item["payload_sha256"], f"{path}.payload_sha256")
    _integer(item["byte_offset"], f"{path}.byte_offset")
    for field in ("first_codec_frame", "first_sample", "sample_count", "codec_frames"):
        if item[field] is not None:
            _integer(item[field], f"{path}.{field}")
    _boolean(item["is_first"], f"{path}.is_first")
    _nullable_boolean(item["is_final"], f"{path}.is_final")
    if engine == "native":
        for field in (
            "first_codec_frame",
            "first_sample",
            "sample_count",
            "codec_frames",
            "is_final",
        ):
            if item[field] is None:
                _fail(
                    f"{path}.{field}", "native application packets require this field"
                )
        if item["sample_count"] <= 0 or item["codec_frames"] <= 0:
            _fail(
                path,
                "native application packets require positive samples and codec frames",
            )
        if item["sample_count"] * 2 != item["payload_bytes"]:
            _fail(f"{path}.payload_bytes", "does not match native sample_count")
    elif any(
        item[field] is not None
        for field in (
            "first_codec_frame",
            "first_sample",
            "sample_count",
            "codec_frames",
            "is_final",
        )
    ):
        _fail(
            path, "SGLang raw transport arrivals must keep model packet metadata null"
        )
    return item


def _validate_packet_groups(
    packets: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    path: str,
    engine: str,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        index = packet["request_index"]
        if index >= len(requests):
            _fail(path, f"packet references unknown request_index {index}")
        grouped[index].append(packet)
    for request_index, request in enumerate(requests):
        group = sorted(
            grouped.get(request_index, []), key=lambda item: item["sequence"]
        )
        if len(group) != request["packet_count"]:
            _fail(
                path,
                f"request_index {request_index} declares {request['packet_count']} packets but has {len(group)}",
            )
        if not request["success"]:
            if group:
                _fail(
                    path, f"failed request_index {request_index} must not have packets"
                )
            continue
        byte_offset = 0
        previous_arrival: float | None = None
        sample_offset = 0
        codec_offset = 0
        for sequence, packet in enumerate(group):
            packet_path = f"{path}[request_index={request_index},sequence={sequence}]"
            if packet["sequence"] != sequence:
                _fail(packet_path, "packet sequence is not contiguous from zero")
            if packet["workload_id"] != request["workload_id"]:
                _fail(f"{packet_path}.workload_id", "does not match its request")
            if packet["is_first"] != (sequence == 0):
                _fail(f"{packet_path}.is_first", "does not match packet sequence")
            if packet["byte_offset"] != byte_offset:
                _fail(
                    f"{packet_path}.byte_offset",
                    "packet byte offsets are not contiguous",
                )
            if previous_arrival is None:
                if packet["inter_arrival_ms"] is not None:
                    _fail(
                        f"{packet_path}.inter_arrival_ms",
                        "first packet interval must be null",
                    )
                if not math.isclose(
                    packet["arrival_ms"], request["ttfa_ms"], rel_tol=1e-9, abs_tol=1e-6
                ):
                    _fail(
                        f"{packet_path}.arrival_ms",
                        "first packet arrival does not match request TTFA",
                    )
            else:
                expected_interval = packet["arrival_ms"] - previous_arrival
                if expected_interval < 0:
                    _fail(
                        f"{packet_path}.arrival_ms", "packet arrivals are not monotonic"
                    )
                if packet["inter_arrival_ms"] is None or not math.isclose(
                    packet["inter_arrival_ms"],
                    expected_interval,
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                ):
                    _fail(
                        f"{packet_path}.inter_arrival_ms",
                        "does not match consecutive arrivals",
                    )
            if engine == "native":
                if packet["first_sample"] != sample_offset:
                    _fail(
                        f"{packet_path}.first_sample",
                        "native samples are not contiguous",
                    )
                if packet["first_codec_frame"] != codec_offset:
                    _fail(
                        f"{packet_path}.first_codec_frame",
                        "native codec frames are not contiguous",
                    )
                sample_offset += packet["sample_count"]
                codec_offset += packet["codec_frames"]
                if packet["is_final"] != (sequence == len(group) - 1):
                    _fail(
                        f"{packet_path}.is_final",
                        "only the final native packet may be final",
                    )
            byte_offset += packet["payload_bytes"]
            previous_arrival = packet["arrival_ms"]
        if byte_offset != request["samples"] * 2:
            _fail(
                path,
                f"request_index {request_index} packet bytes do not match decoded PCM samples",
            )
        if engine == "native" and sample_offset != request["samples"]:
            _fail(
                path,
                f"request_index {request_index} native packet samples do not match request",
            )
    return grouped


CLIENT_SUMMARY_REQUIRED = (
    "schema_version",
    "endpoint",
    "backend",
    "sglang_model",
    "concurrency",
    "synchronized_batch_width",
    "warmups",
    "planned_requests",
    "completed_requests",
    "successful_requests",
    "failed_requests",
    "natural_eos_requests",
    "length_limited_requests",
    "eos_unknown_requests",
    "sampling_parity_qualifying_requests",
    "sampling_parity_non_qualifying_requests",
    "normalized_sampling_sha256s",
    "benchmark_wall_seconds",
    "attempted_requests_per_second",
    "throughput_requests_per_second",
    "total_audio_seconds",
    "aggregate_rtf",
    "summed_request_wall_rtf",
    "ttfa_ms",
    "wall_ms",
    "request_rtf",
)


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)


def _expected_distribution(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _validate_client_distribution(
    value: Any,
    path: str,
    expected: dict[str, float | int] | None,
) -> None:
    if expected is None:
        if value is not None:
            _fail(path, "must be null when there are no successful values")
        return
    item = _strict_object(
        value, path, ("count", "min", "mean", "p50", "p90", "p95", "p99", "max")
    )
    if _integer(item["count"], f"{path}.count", 1) != expected["count"]:
        _fail(f"{path}.count", f"expected {expected['count']}")
    for field in ("min", "mean", "p50", "p90", "p95", "p99", "max"):
        observed = _number(item[field], f"{path}.{field}")
        if not _close(observed, float(expected[field])):
            _fail(
                f"{path}.{field}",
                f"expected {float(expected[field]):.12g}, observed {observed:.12g}",
            )


def _validate_client_summary(
    raw: Any,
    path: str,
    engine: str,
    profile_id: str,
    requests: list[dict[str, Any]],
    workload: dict[str, Any],
) -> dict[str, Any]:
    summary = _strict_object(raw, path, CLIENT_SUMMARY_REQUIRED)
    if summary["schema_version"] != CLIENT_SCHEMA_VERSION:
        _fail(f"{path}.schema_version", f"expected {CLIENT_SCHEMA_VERSION}")
    _string(summary["endpoint"], f"{path}.endpoint")
    expected_backend = "native" if engine == "native" else "sglang-omni"
    if summary["backend"] != expected_backend:
        _fail(f"{path}.backend", f"expected {expected_backend}")
    if engine == "native":
        if summary["sglang_model"] is not None:
            _fail(f"{path}.sglang_model", "must be null for Native")
    else:
        _string(summary["sglang_model"], f"{path}.sglang_model")
    profile = next(item for item in workload["profiles"] if item["id"] == profile_id)
    if summary["concurrency"] != profile_id:
        _fail(f"{path}.concurrency", f"expected {profile_id}")
    if (
        _integer(
            summary["synchronized_batch_width"], f"{path}.synchronized_batch_width", 1
        )
        != profile["concurrency"]
    ):
        _fail(f"{path}.synchronized_batch_width", "does not match the profile")
    warmups = _integer(summary["warmups"], f"{path}.warmups")
    if warmups < workload["warmup_requests_per_engine"]:
        _fail(f"{path}.warmups", "below the manifest warmup requirement")
    for field in (
        "planned_requests",
        "completed_requests",
        "successful_requests",
        "failed_requests",
        "natural_eos_requests",
        "length_limited_requests",
        "eos_unknown_requests",
        "sampling_parity_qualifying_requests",
        "sampling_parity_non_qualifying_requests",
    ):
        _integer(summary[field], f"{path}.{field}")
    completed = len(requests)
    successful = [item for item in requests if item["success"]]
    expected_counts = {
        "planned_requests": completed,
        "completed_requests": completed,
        "successful_requests": len(successful),
        "failed_requests": completed - len(successful),
        "natural_eos_requests": sum(item["natural_eos"] is True for item in successful),
        "length_limited_requests": sum(
            item["length_limited"] is True for item in successful
        ),
        "eos_unknown_requests": sum(item["natural_eos"] is None for item in successful),
        "sampling_parity_qualifying_requests": sum(
            item["sampling_parity_qualifying"] for item in requests
        ),
        "sampling_parity_non_qualifying_requests": sum(
            not item["sampling_parity_qualifying"] for item in requests
        ),
    }
    for field, expected in expected_counts.items():
        if summary[field] != expected:
            _fail(f"{path}.{field}", f"expected {expected}, observed {summary[field]}")
    if [item["request_index"] for item in requests] != list(range(completed)):
        _fail(path, "request indices must be contiguous from zero")
    expected_hashes = sorted({item["normalized_sampling_sha256"] for item in requests})
    hashes = summary["normalized_sampling_sha256s"]
    if not isinstance(hashes, list) or any(
        not isinstance(item, str) for item in hashes
    ):
        _fail(f"{path}.normalized_sampling_sha256s", "expected an array of digests")
    for index, digest in enumerate(hashes):
        _sha256(digest, f"{path}.normalized_sampling_sha256s[{index}]")
    if hashes != expected_hashes:
        _fail(f"{path}.normalized_sampling_sha256s", "does not match request records")
    wall_seconds = _number(
        summary["benchmark_wall_seconds"], f"{path}.benchmark_wall_seconds"
    )
    if wall_seconds <= 0:
        _fail(f"{path}.benchmark_wall_seconds", "must be greater than zero")
    total_audio_seconds = sum(float(item["audio_seconds"]) for item in successful)
    summed_wall_seconds = sum(float(item["wall_ms"]) for item in successful) / 1000.0
    expected_numbers = {
        "attempted_requests_per_second": completed / wall_seconds,
        "throughput_requests_per_second": len(successful) / wall_seconds,
        "total_audio_seconds": total_audio_seconds,
    }
    for field, expected in expected_numbers.items():
        observed = _number(summary[field], f"{path}.{field}")
        if not _close(observed, expected):
            _fail(
                f"{path}.{field}", f"expected {expected:.12g}, observed {observed:.12g}"
            )
    expected_aggregate = (
        wall_seconds / total_audio_seconds if total_audio_seconds > 0 else None
    )
    expected_summed = (
        summed_wall_seconds / total_audio_seconds if total_audio_seconds > 0 else None
    )
    for field, expected in (
        ("aggregate_rtf", expected_aggregate),
        ("summed_request_wall_rtf", expected_summed),
    ):
        observed = _nullable_number(summary[field], f"{path}.{field}")
        if expected is None:
            if observed is not None:
                _fail(f"{path}.{field}", "must be null without successful audio")
        elif observed is None or not _close(observed, expected):
            _fail(f"{path}.{field}", f"expected {expected:.12g}, observed {observed}")
    _validate_client_distribution(
        summary["ttfa_ms"],
        f"{path}.ttfa_ms",
        _expected_distribution([item["ttfa_ms"] for item in successful]),
    )
    _validate_client_distribution(
        summary["wall_ms"],
        f"{path}.wall_ms",
        _expected_distribution([item["wall_ms"] for item in successful]),
    )
    _validate_client_distribution(
        summary["request_rtf"],
        f"{path}.request_rtf",
        _expected_distribution([item["rtf"] for item in successful]),
    )
    return summary


def _client_descriptor_map(
    descriptors: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, int, str], tuple[str, dict[str, Any]]]:
    result: dict[tuple[str, str, int, str], tuple[str, dict[str, Any]]] = {}
    for relative, descriptor in descriptors.items():
        if descriptor["role"] in CLIENT_ROLES:
            key = (
                descriptor["engine_id"],
                descriptor["profile_id"],
                descriptor["round"],
                descriptor["role"],
            )
            result[key] = (relative, descriptor)
    return result


def _resource_map(
    manifest: dict[str, Any],
    descriptors: dict[str, dict[str, Any]],
    run_keys: set[tuple[str, str, int]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    raw_by_path = {
        relative: descriptor
        for relative, descriptor in descriptors.items()
        if descriptor["role"] == "raw"
    }
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, item in enumerate(manifest["run_resources"]):
        key = (item["engine_id"], item["profile_id"], item["round"])
        if key in result:
            _fail(f"run_resources[{index}]", f"duplicate resource record for {key}")
        if key not in run_keys:
            _fail(f"run_resources[{index}]", f"does not match a client run: {key}")
        if (
            item["sampling_interval_ms"]
            != manifest["methodology"]["sampling_interval_ms"]
        ):
            _fail(
                f"run_resources[{index}].sampling_interval_ms",
                "must equal methodology.sampling_interval_ms",
            )
        for telemetry_path in item["telemetry_evidence_paths"]:
            descriptor = raw_by_path.get(telemetry_path)
            if descriptor is None:
                _fail(
                    f"run_resources[{index}].telemetry_evidence_paths",
                    f"does not reference raw evidence: {telemetry_path}",
                )
            descriptor_key = (
                descriptor.get("engine_id"),
                descriptor.get("profile_id"),
                descriptor.get("round"),
            )
            if descriptor_key != key:
                _fail(
                    f"run_resources[{index}].telemetry_evidence_paths",
                    f"raw evidence {telemetry_path} belongs to {descriptor_key}, not {key}",
                )
        result[key] = item
    if set(result) != run_keys:
        missing = sorted(run_keys - set(result))
        extra = sorted(set(result) - run_keys)
        _fail(
            "run_resources",
            f"must match client runs exactly; missing={missing} extra={extra}",
        )
    return result


def _request_identity(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["request_index"],
        item["workload_id"],
        item["text_sha256"],
        item["voice_description_sha256"],
        item["normalized_sampling_sha256"],
        item["language"],
        item["streaming"],
    )


def _status_from_client_request(item: dict[str, Any]) -> str:
    if item["success"]:
        return "success"
    code = item["failure"]["code"].lower()
    if "timeout" in code:
        return "timeout"
    if "cancel" in code:
        return "cancelled"
    return "error"


def _validate_client_runs(
    manifest: dict[str, Any],
    payloads: dict[str, Any],
    descriptors: dict[str, dict[str, Any]],
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[str, tuple[dict[str, Any], ...]],
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
]:
    workload_path = next(
        relative
        for relative, descriptor in descriptors.items()
        if descriptor["role"] == "workload"
    )
    workload_records = _validate_client_workload(payloads[workload_path])
    if manifest["evidence_kind"] == "production":
        if manifest["workload"]["response_mode"] != "streaming":
            _fail("workload.response_mode", "production comparison requires streaming")
        _validate_production_workload_durations(workload_records, workload_path)
        for index, item in enumerate(workload_records):
            if item.get("stream", True) is not True:
                _fail(
                    f"{workload_path}:{index + 1}.stream",
                    "production comparison requires streaming",
                )
    lookup = _client_descriptor_map(descriptors)
    run_keys = {key[:3] for key in lookup}
    resources = _resource_map(manifest, descriptors, run_keys)
    run_summaries: dict[tuple[str, str, int], dict[str, Any]] = {}
    measurements: dict[str, list[dict[str, Any]]] = {"native": [], "sglang": []}
    request_sets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    minimum = manifest["workload"]["minimum_measured_requests_per_profile"]
    for engine, profile_id, round_number in sorted(run_keys):
        request_path = lookup[(engine, profile_id, round_number, "client_requests")][0]
        packet_path = lookup[(engine, profile_id, round_number, "client_packets")][0]
        summary_path = lookup[(engine, profile_id, round_number, "client_summary")][0]
        raw_requests = payloads[request_path]
        raw_packets = payloads[packet_path]
        if not isinstance(raw_requests, list) or not raw_requests:
            _fail(request_path, "expected non-empty client requests JSONL")
        if not isinstance(raw_packets, list):
            _fail(packet_path, "expected client packets JSONL")
        requests = [
            _validate_client_request(
                raw,
                f"{request_path}:{index + 1}",
                engine,
                workload_records,
                manifest["workload"]["sample_rate_hz"],
            )
            for index, raw in enumerate(raw_requests)
        ]
        if [item["request_index"] for item in requests] != list(range(len(requests))):
            _fail(
                request_path,
                "records must be sorted with contiguous request_index values",
            )
        packets = [
            _validate_client_packet(raw, f"{packet_path}:{index + 1}", engine)
            for index, raw in enumerate(raw_packets)
        ]
        packet_groups = _validate_packet_groups(packets, requests, packet_path, engine)
        summary = _validate_client_summary(
            payloads[summary_path],
            summary_path,
            engine,
            profile_id,
            requests,
            manifest["workload"],
        )
        successes = [item for item in requests if item["success"]]
        if len(successes) < minimum:
            _fail(
                request_path,
                f"has {len(successes)} successful requests, below required {minimum}",
            )
        if manifest["evidence_kind"] == "production" and any(
            not item["sampling_parity_qualifying"] for item in requests
        ):
            _fail(
                request_path,
                "production comparison requires sampling parity for every measured request",
            )
        if engine == "native":
            invalid = [
                item["request_index"]
                for item in successes
                if item["natural_eos"] is not True
                or item["length_limited"] is not False
                or item["finish_reason"] != "stop"
            ]
            if invalid:
                _fail(
                    request_path,
                    f"Native successful requests must all be natural EOS; invalid indices={invalid[:5]}",
                )
        else:
            invalid = [
                item["request_index"]
                for item in successes
                if item["natural_eos"] is not None
                or item["length_limited"] is not None
                or item["finish_reason"] is not None
            ]
            if invalid:
                _fail(
                    request_path,
                    f"stock SGLang successful requests must retain eos_unknown; invalid indices={invalid[:5]}",
                )
            if summary["sglang_model"] != manifest["model"]["repository"]:
                _fail(f"{summary_path}.sglang_model", "must equal model.repository")
            if manifest["evidence_kind"] == "production":
                for item in successes:
                    _validate_production_sglang_audio_limit(
                        item,
                        packet_groups[item["request_index"]],
                        manifest["workload"]["sample_rate_hz"],
                        f"{request_path}:{item['request_index'] + 1}",
                    )
        resource = resources[(engine, profile_id, round_number)]
        per_row_energy = resource["energy_j"] / len(requests)
        for item in requests:
            intervals = [
                packet["inter_arrival_ms"]
                for packet in sorted(
                    packet_groups.get(item["request_index"], []),
                    key=lambda row: row["sequence"],
                )
                if packet["inter_arrival_ms"] is not None
            ]
            measurements[engine].append(
                {
                    "engine_id": engine,
                    "request_id": item["workload_id"],
                    "request_index": item["request_index"],
                    "profile_id": profile_id,
                    "round": round_number,
                    "status": _status_from_client_request(item),
                    "http_status": item["http_status"],
                    "ttfa_ms": item["ttfa_ms"],
                    "total_latency_ms": item["wall_ms"],
                    "audio_duration_ms": None
                    if item["audio_seconds"] is None
                    else item["audio_seconds"] * 1000.0,
                    "rtf": item["rtf"],
                    "packet_count": item["packet_count"] if item["success"] else None,
                    "packet_intervals_ms": intervals if item["success"] else None,
                    "output_bytes": None
                    if item["samples"] is None
                    else item["samples"] * 2,
                    "process_rss_peak_bytes": resource["process_rss_peak_bytes"],
                    "gpu_unified_memory_peak_bytes": resource[
                        "gpu_unified_memory_peak_bytes"
                    ],
                    "average_power_w": resource["average_power_w"],
                    "peak_power_w": resource["peak_power_w"],
                    "energy_j": per_row_energy,
                    "natural_eos": item["natural_eos"],
                    "length_limited": item["length_limited"],
                }
            )
        request_sets[(engine, profile_id, round_number)] = requests
        run_summaries[(engine, profile_id, round_number)] = summary
    for profile_id in {item["id"] for item in manifest["workload"]["profiles"]}:
        rounds = sorted(
            round_number
            for engine, candidate_profile, round_number in run_keys
            if engine == "native" and candidate_profile == profile_id
        )
        reference: list[tuple[Any, ...]] | None = None
        for round_number in rounds:
            native_identity = [
                _request_identity(item)
                for item in request_sets[("native", profile_id, round_number)]
            ]
            sglang_identity = [
                _request_identity(item)
                for item in request_sets[("sglang", profile_id, round_number)]
            ]
            if native_identity != sglang_identity:
                _fail(
                    "client_runs",
                    f"Native and SGLang workload/sampling identity differs for {profile_id} round {round_number}",
                )
            if reference is None:
                reference = native_identity
            elif native_identity != reference:
                _fail(
                    "client_runs",
                    f"workload identity differs across {profile_id} rounds",
                )
    return (
        workload_records,
        {engine: tuple(rows) for engine, rows in measurements.items()},
        run_summaries,
        resources,
    )


def load_bundle(manifest_path: Path, allow_test_fixture: bool = False) -> Bundle:
    manifest_path = manifest_path.expanduser()
    if manifest_path.is_symlink():
        _fail("manifest", "symlinks are forbidden")
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvidenceError(f"manifest does not exist: {manifest_path}") from exc
    if not manifest_path.is_file() or manifest_path.suffix != ".json":
        _fail("manifest", "expected a regular .json file")
    manifest_bytes = manifest_path.read_bytes()
    if not manifest_bytes:
        _fail("manifest", "file is empty")
    manifest = _validate_manifest(_parse_json_bytes(manifest_bytes, str(manifest_path)))
    if manifest["evidence_kind"] == "test_fixture" and not allow_test_fixture:
        _fail("evidence_kind", "test fixtures require --allow-test-fixture")
    if manifest["schema_version"] == "1.1":
        payloads, descriptors = _load_client_evidence_files(
            manifest, manifest_path.parent
        )
        requests, measurements, run_summaries, run_resources = _validate_client_runs(
            manifest,
            payloads,
            descriptors,
        )
    else:
        payloads, descriptors = _load_evidence_files(manifest, manifest_path.parent)
        request_path = next(
            relative
            for relative, descriptor in descriptors.items()
            if descriptor["role"] == "requests"
        )
        requests = _validate_requests(payloads[request_path], manifest["workload"])
        if (
            len(requests)
            < manifest["workload"]["minimum_measured_requests_per_profile"]
        ):
            _fail(
                "requests",
                f"contains {len(requests)} canonical requests, below required minimum "
                f"{manifest['workload']['minimum_measured_requests_per_profile']}",
            )
        measurements = _validate_measurements(
            payloads, descriptors, requests, manifest["workload"]
        )
        run_summaries = {}
        run_resources = {}
    return Bundle(
        manifest_path=manifest_path,
        manifest_sha256=_file_sha256(manifest_bytes),
        manifest=manifest,
        requests=requests,
        measurements=measurements,
        evidence_payloads=payloads,
        run_summaries=run_summaries,
        run_resources=run_resources,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise EvidenceError("cannot calculate a percentile from an empty series")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate(
    bundle: Bundle,
) -> dict[str, dict[str, dict[str, float | int | Counter[str]]]]:
    result: dict[str, dict[str, dict[str, float | int | Counter[str]]]] = {}
    profile_order = [
        item["id"]
        for item in sorted(
            bundle.manifest["workload"]["profiles"], key=lambda row: row["concurrency"]
        )
    ]
    for engine in ("native", "sglang"):
        result[engine] = {}
        for profile_id in profile_order:
            rows = [
                item
                for item in bundle.measurements[engine]
                if item["profile_id"] == profile_id
            ]
            successes = [item for item in rows if item["status"] == "success"]
            if not successes:
                _fail(
                    f"measurements.{engine}.{profile_id}",
                    "at least one successful request is required",
                )
            statuses = Counter(item["status"] for item in rows)
            if bundle.manifest["schema_version"] == "1.1":
                keys = sorted(
                    key
                    for key in bundle.run_summaries
                    if key[:2] == (engine, profile_id)
                )
                summaries = [bundle.run_summaries[key] for key in keys]
                resources = [bundle.run_resources[key] for key in keys]
                span_seconds = sum(
                    float(item["benchmark_wall_seconds"]) for item in summaries
                )
                total_audio_from_summaries = sum(
                    float(item["total_audio_seconds"]) for item in summaries
                )
                energy_sum = sum(float(item["energy_j"]) for item in resources)
                average_power = (
                    sum(
                        float(resource["average_power_w"])
                        * float(summary["benchmark_wall_seconds"])
                        for resource, summary in zip(resources, summaries, strict=True)
                    )
                    / span_seconds
                )
                attempted_throughput = len(rows) / span_seconds
                aggregate_rtf = span_seconds / total_audio_from_summaries
                summed_wall_seconds = sum(
                    float(item["summed_request_wall_rtf"])
                    * float(item["total_audio_seconds"])
                    for item in summaries
                )
                summed_request_wall_rtf = (
                    summed_wall_seconds / total_audio_from_summaries
                )
                rounds = len(keys)
            else:
                start = min(float(item["run_elapsed_start_ms"]) for item in rows)
                end = max(float(item["run_elapsed_end_ms"]) for item in rows)
                span_seconds = (end - start) / 1000.0
                energy_sum = sum(float(item["energy_j"]) for item in rows)
                average_power = fmean(float(item["average_power_w"]) for item in rows)
                attempted_throughput = len(rows) / span_seconds
                total_audio_from_summaries = (
                    sum(float(item["audio_duration_ms"]) for item in successes) / 1000.0
                )
                aggregate_rtf = span_seconds / total_audio_from_summaries
                summed_request_wall_rtf = (
                    sum(float(item["total_latency_ms"]) for item in successes)
                    / 1000.0
                    / total_audio_from_summaries
                )
                resources = []
                rounds = 1
            if span_seconds <= 0:
                _fail(
                    f"measurements.{engine}.{profile_id}", "run span must be positive"
                )
            ttfa = [float(item["ttfa_ms"]) for item in successes]
            rtf = [float(item["rtf"]) for item in successes]
            audio_seconds = (
                sum(float(item["audio_duration_ms"]) for item in successes) / 1000.0
            )
            intervals = [
                float(interval)
                for item in successes
                for interval in item["packet_intervals_ms"]
            ]
            result[engine][profile_id] = {
                "total": len(rows),
                "success": len(successes),
                "rounds": rounds,
                "statuses": statuses,
                "ttfa_p50": _percentile(ttfa, 0.50),
                "ttfa_p95": _percentile(ttfa, 0.95),
                "ttfa_p99": _percentile(ttfa, 0.99),
                "rtf_p50": _percentile(rtf, 0.50),
                "rtf_p95": _percentile(rtf, 0.95),
                "rtf_p99": _percentile(rtf, 0.99),
                "aggregate_rtf": aggregate_rtf,
                "summed_request_wall_rtf": summed_request_wall_rtf,
                "attempted_throughput_rps": attempted_throughput,
                "request_throughput_rps": len(successes) / span_seconds,
                "audio_throughput_x": audio_seconds / span_seconds,
                "rss_peak_bytes": max(
                    float(item["process_rss_peak_bytes"]) for item in rows
                ),
                "gpu_memory_peak_bytes": max(
                    float(item["gpu_unified_memory_peak_bytes"]) for item in rows
                ),
                "average_power_w": average_power,
                "peak_power_w": max(float(item["peak_power_w"]) for item in rows),
                "energy_per_request_j": energy_sum / len(successes),
                "energy_per_audio_second_j": energy_sum / audio_seconds,
                "packet_count_mean": fmean(
                    float(item["packet_count"]) for item in successes
                ),
                "cadence_p50_ms": _percentile(intervals, 0.50) if intervals else 0.0,
                "cadence_p95_ms": _percentile(intervals, 0.95) if intervals else 0.0,
                "success_rate": len(successes) / len(rows),
                "natural_eos": sum(
                    item.get("natural_eos") is True for item in successes
                ),
                "length_limited": sum(
                    item.get("length_limited") is True for item in successes
                ),
                "eos_unknown": sum(
                    item.get("natural_eos") is None for item in successes
                ),
            }
    return result


def _ascii_hyphens(value: Any) -> str:
    return str(value).translate(FORBIDDEN_DASHES)


def _xml(value: Any) -> str:
    return html.escape(_ascii_hyphens(value), quote=True)


def _gib(value: float | int) -> str:
    return f"{float(value) / (1024**3):.2f} GiB"


def _size(value: float | int) -> str:
    value = float(value)
    if value >= 1024**3:
        return f"{value / (1024**3):.2f} GiB"
    if value >= 1024**2:
        return f"{value / (1024**2):.2f} MiB"
    return f"{value:.0f} B"


def _ms(value: float | int) -> str:
    return f"{float(value):.2f} ms"


def _number_text(value: float | int, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


class ProfileLineChart(Flowable):  # type: ignore[misc]
    """Compact vector line chart with monochrome lines and distinct markers."""

    def __init__(
        self,
        width: float,
        height: float,
        categories: Sequence[str],
        native: Sequence[float],
        sglang: Sequence[float],
        y_label: str,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.categories = list(categories)
        self.native = list(native)
        self.sglang = list(sglang)
        self.y_label = y_label

    def draw(self) -> None:
        c = self.canv
        left, right, bottom, top = 48.0, 14.0, 30.0, 26.0
        chart_w = self.width - left - right
        chart_h = self.height - bottom - top
        maximum = max(self.native + self.sglang) if self.native or self.sglang else 1.0
        maximum = maximum * 1.12 if maximum > 0 else 1.0
        c.setStrokeColor(colors.black)
        c.setFillColor(colors.black)
        c.setLineWidth(0.6)
        c.line(left, bottom, left, bottom + chart_h)
        c.line(left, bottom, left + chart_w, bottom)
        c.setFont("Helvetica", 6.8)
        for tick in range(5):
            value = maximum * tick / 4
            y = bottom + chart_h * tick / 4
            c.setStrokeColor(colors.HexColor("#B0B0B0"))
            c.setDash(1, 2)
            c.line(left, y, left + chart_w, y)
            c.setDash()
            c.setFillColor(colors.black)
            c.drawRightString(left - 5, y - 2, f"{value:.2f}")
        c.saveState()
        c.translate(9, bottom + chart_h / 2)
        c.rotate(90)
        c.setFont("Helvetica", 7)
        c.drawCentredString(0, 0, _ascii_hyphens(self.y_label))
        c.restoreState()
        count = max(len(self.categories), 1)
        step = chart_w / count
        x_positions = [left + step * (index + 0.5) for index in range(count)]
        for x, label in zip(x_positions, self.categories):
            c.setFont("Helvetica", 7)
            c.drawCentredString(x, bottom - 13, _ascii_hyphens(label))

        def draw_series(values: Sequence[float], dashed: bool, square: bool) -> None:
            points = [
                (x, bottom + chart_h * value / maximum)
                for x, value in zip(x_positions, values)
            ]
            c.setStrokeColor(colors.black)
            c.setFillColor(colors.white if square else colors.black)
            c.setLineWidth(1.4)
            c.setDash(4, 2) if dashed else c.setDash()
            for first, second in zip(points, points[1:]):
                c.line(first[0], first[1], second[0], second[1])
            c.setDash()
            for x, y in points:
                if square:
                    c.rect(x - 3, y - 3, 6, 6, stroke=1, fill=1)
                else:
                    c.circle(x, y, 3, stroke=1, fill=1)

        draw_series(self.native, dashed=False, square=False)
        draw_series(self.sglang, dashed=True, square=True)
        legend_y = self.height - 10
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.circle(left + 3, legend_y, 3, stroke=1, fill=1)
        c.drawString(left + 10, legend_y - 2, "Native")
        c.setDash(4, 2)
        c.line(left + 61, legend_y, left + 79, legend_y)
        c.setDash()
        c.setFillColor(colors.white)
        c.rect(left + 67, legend_y - 3, 6, 6, stroke=1, fill=1)
        c.setFillColor(colors.black)
        c.drawString(left + 84, legend_y - 2, "SGLang")


class PatternBarChart(Flowable):  # type: ignore[misc]
    """Grouped vector bars: solid Native and diagonal-hatched SGLang."""

    def __init__(
        self,
        width: float,
        height: float,
        categories: Sequence[str],
        native: Sequence[float],
        sglang: Sequence[float],
        y_label: str,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.categories = list(categories)
        self.native = list(native)
        self.sglang = list(sglang)
        self.y_label = y_label

    def _hatched_bar(self, x: float, y: float, width: float, height: float) -> None:
        c = self.canv
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.black)
        c.rect(x, y, width, height, stroke=1, fill=1)
        c.saveState()
        path = c.beginPath()
        path.rect(x, y, width, height)
        c.clipPath(path, stroke=0, fill=0)
        c.setLineWidth(0.45)
        spacing = 5
        offset = -height
        while offset <= width:
            c.line(x + offset, y, x + offset + height, y + height)
            offset += spacing
        c.restoreState()

    def draw(self) -> None:
        c = self.canv
        left, right, bottom, top = 48.0, 14.0, 34.0, 27.0
        chart_w = self.width - left - right
        chart_h = self.height - bottom - top
        maximum = max(self.native + self.sglang) if self.native or self.sglang else 1.0
        maximum = maximum * 1.18 if maximum > 0 else 1.0
        c.setStrokeColor(colors.black)
        c.setFillColor(colors.black)
        c.setLineWidth(0.6)
        c.line(left, bottom, left, bottom + chart_h)
        c.line(left, bottom, left + chart_w, bottom)
        c.setFont("Helvetica", 6.8)
        for tick in range(5):
            value = maximum * tick / 4
            y = bottom + chart_h * tick / 4
            c.setStrokeColor(colors.HexColor("#B0B0B0"))
            c.setDash(1, 2)
            c.line(left, y, left + chart_w, y)
            c.setDash()
            c.setFillColor(colors.black)
            c.drawRightString(left - 5, y - 2, f"{value:.2f}")
        c.saveState()
        c.translate(9, bottom + chart_h / 2)
        c.rotate(90)
        c.setFont("Helvetica", 7)
        c.drawCentredString(0, 0, _ascii_hyphens(self.y_label))
        c.restoreState()
        count = max(len(self.categories), 1)
        group = chart_w / count
        bar_width = min(30.0, group * 0.28)
        for index, (label, native_value, sglang_value) in enumerate(
            zip(self.categories, self.native, self.sglang)
        ):
            center = left + group * (index + 0.5)
            native_height = chart_h * native_value / maximum
            sglang_height = chart_h * sglang_value / maximum
            c.setFillColor(colors.black)
            c.rect(
                center - bar_width - 2,
                bottom,
                bar_width,
                native_height,
                stroke=1,
                fill=1,
            )
            self._hatched_bar(center + 2, bottom, bar_width, sglang_height)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 7)
            c.drawCentredString(center, bottom - 14, _ascii_hyphens(label))
        legend_y = self.height - 10
        c.setFillColor(colors.black)
        c.rect(left, legend_y - 4, 10, 8, stroke=1, fill=1)
        c.setFont("Helvetica", 7)
        c.drawString(left + 15, legend_y - 2, "Native")
        self._hatched_bar(left + 64, legend_y - 4, 10, 8)
        c.setFillColor(colors.black)
        c.drawString(left + 79, legend_y - 2, "SGLang")


def _styles() -> dict[str, Any]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            alignment=TA_LEFT,
            textColor=colors.black,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#333333"),
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.black,
            spaceBefore=6,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.black,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.black,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=9,
            textColor=colors.black,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=sample["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#333333"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=8.5,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=6.7,
            leading=8.5,
            textColor=colors.black,
        ),
        "mono": ParagraphStyle(
            "Mono",
            parent=sample["BodyText"],
            fontName="Courier",
            fontSize=5.8,
            leading=7.3,
            textColor=colors.black,
        ),
    }


def _paragraph(value: Any, style: Any) -> Any:
    return Paragraph(_xml(value), style)


def _table(
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    styles: dict[str, Any],
    repeat_rows: int = 1,
    long: bool = False,
) -> Any:
    converted: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        style = (
            styles["table_header"] if row_index < repeat_rows else styles["table_cell"]
        )
        converted.append(
            [
                value if hasattr(value, "wrap") else _paragraph(value, style)
                for value in row
            ]
        )
    cls = LongTable if long else Table
    table = cls(
        converted, colWidths=list(widths), repeatRows=repeat_rows, hAlign="LEFT"
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, repeat_rows - 1), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, repeat_rows - 1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#777777")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row_index in range(repeat_rows, len(rows)):
        if (row_index - repeat_rows) % 2:
            commands.append(
                (
                    "BACKGROUND",
                    (0, row_index),
                    (-1, row_index),
                    colors.HexColor("#EEEEEE"),
                )
            )
    table.setStyle(TableStyle(commands))
    return table


def _bullet_list(items: Sequence[str], styles: dict[str, Any]) -> list[Any]:
    result: list[Any] = []
    for item in items:
        result.append(Paragraph(f"- {_xml(item)}", styles["body"]))
    return result


def _engine_map(bundle: Bundle) -> dict[str, dict[str, Any]]:
    return {item["role"]: item for item in bundle.manifest["implementations"]}


def _metric_rows(
    aggregates: dict[str, dict[str, dict[str, Any]]],
    profiles: Sequence[str],
    fields: Sequence[tuple[str, str, Any]],
) -> list[list[str]]:
    rows: list[list[str]] = [["Profile", "Engine"] + [label for label, _, _ in fields]]
    for profile in profiles:
        for engine, label in (("native", "Native"), ("sglang", "SGLang")):
            metric = aggregates[engine][profile]
            rows.append(
                [profile, label]
                + [formatter(metric[field]) for _, field, formatter in fields]
            )
    return rows


def _section_title(number: int, title: str, styles: dict[str, Any]) -> Any:
    return _paragraph(f"{number}. {title}", styles["h1"])


def _build_story(bundle: Bundle, doc_width: float) -> list[Any]:
    manifest = bundle.manifest
    aggregates = aggregate(bundle)
    styles = _styles()
    profiles = [
        item["id"]
        for item in sorted(
            manifest["workload"]["profiles"], key=lambda row: row["concurrency"]
        )
    ]
    engines = _engine_map(bundle)
    fixture = manifest["evidence_kind"] == "test_fixture"
    story: list[Any] = []
    if fixture:
        warning = Table(
            [[_paragraph(TEST_FIXTURE_BANNER, styles["table_header"])]],
            colWidths=[doc_width],
        )
        warning.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("BOX", (0, 0), (-1, -1), 1, colors.black),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.extend([warning, Spacer(1, 16)])
    story.extend(
        [
            _paragraph(manifest["report"]["title"], styles["title"]),
            _paragraph(
                "A reproducible Native versus SGLang comparison for Qwen3-TTS 1.7B VoiceDesign",
                styles["subtitle"],
            ),
            Spacer(1, 8),
            _table(
                [
                    ["Evidence field", "Value"],
                    ["Benchmark ID", manifest["report"]["benchmark_id"]],
                    ["Evidence kind", manifest["evidence_kind"]],
                    ["Evidence timestamp", manifest["report"]["generated_at"]],
                    ["Manifest SHA-256", bundle.manifest_sha256],
                    ["Authors", ", ".join(manifest["report"]["authors"])],
                ],
                [doc_width * 0.28, doc_width * 0.72],
                styles,
            ),
            Spacer(1, 18),
            _paragraph(
                "This report is generated only from digest-verified JSON and JSONL evidence. "
                "The validator confirmed identical canonical workload keys and exact model "
                "identity across both implementations before aggregation.",
                styles["body"],
            ),
            Spacer(1, 12),
            _paragraph(
                "Lower is better for TTFA, RTF, memory, power, energy, startup, and image size. "
                "Higher is better for throughput and reliability.",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    b1 = profiles[0]
    native_b1 = aggregates["native"][b1]
    sglang_b1 = aggregates["sglang"][b1]
    story.extend(
        [
            _section_title(1, "Executive summary", styles),
            _paragraph(
                "The summary reports observed measurements without extrapolation. Full "
                "percentiles and every benchmark profile appear in the sections that follow.",
                styles["body"],
            ),
            _table(
                [
                    ["B1 metric", "Native", "SGLang", "Native / SGLang"],
                    [
                        "TTFA p95",
                        _ms(native_b1["ttfa_p95"]),
                        _ms(sglang_b1["ttfa_p95"]),
                        _number_text(native_b1["ttfa_p95"] / sglang_b1["ttfa_p95"]),
                    ],
                    [
                        "Aggregate RTF",
                        _number_text(native_b1["aggregate_rtf"]),
                        _number_text(sglang_b1["aggregate_rtf"]),
                        _number_text(
                            native_b1["aggregate_rtf"] / sglang_b1["aggregate_rtf"]
                        ),
                    ],
                    [
                        "Request throughput",
                        f"{native_b1['request_throughput_rps']:.3f} req/s",
                        f"{sglang_b1['request_throughput_rps']:.3f} req/s",
                        _number_text(
                            native_b1["request_throughput_rps"]
                            / sglang_b1["request_throughput_rps"]
                        ),
                    ],
                    [
                        "Peak GPU unified memory",
                        _gib(native_b1["gpu_memory_peak_bytes"]),
                        _gib(sglang_b1["gpu_memory_peak_bytes"]),
                        _number_text(
                            native_b1["gpu_memory_peak_bytes"]
                            / sglang_b1["gpu_memory_peak_bytes"]
                        ),
                    ],
                    [
                        "Success rate",
                        f"{native_b1['success_rate'] * 100:.2f}%",
                        f"{sglang_b1['success_rate'] * 100:.2f}%",
                        "n/a",
                    ],
                ],
                [
                    doc_width * 0.28,
                    doc_width * 0.22,
                    doc_width * 0.22,
                    doc_width * 0.28,
                ],
                styles,
            ),
            Spacer(1, 10),
            _paragraph(
                f"Native streaming semantics: {engines['native']['streaming_semantics']}. "
                f"SGLang streaming semantics: {engines['sglang']['streaming_semantics']}.",
                styles["body"],
            ),
            _section_title(2, "Methodology and fairness controls", styles),
            _table(
                [
                    ["Method", "Declared definition"],
                    ["Clock", manifest["methodology"]["clock_source"]],
                    ["TTFA", manifest["methodology"]["ttfa_definition"]],
                    ["RTF", manifest["methodology"]["rtf_definition"]],
                    ["Throughput", manifest["methodology"]["throughput_definition"]],
                    ["Memory", manifest["methodology"]["memory_definition"]],
                    ["Power", manifest["methodology"]["power_definition"]],
                    ["Energy", manifest["methodology"]["energy_definition"]],
                    ["Startup", manifest["methodology"]["startup_definition"]],
                    ["Run order", manifest["methodology"]["run_order"]],
                    ["Statistics", manifest["methodology"]["statistical_method"]],
                ],
                [doc_width * 0.20, doc_width * 0.80],
                styles,
                long=True,
            ),
            Spacer(1, 6),
            _paragraph(
                f"Power sampling interval: {manifest['methodology']['sampling_interval_ms']} ms. "
                f"Warmup requests per engine: {manifest['workload']['warmup_requests_per_engine']}.",
                styles["body"],
            ),
            _paragraph("Environment controls", styles["h2"]),
            *_bullet_list(manifest["methodology"]["environment_controls"], styles),
            PageBreak(),
        ]
    )

    system = manifest["system"]
    model = manifest["model"]
    story.extend(
        [
            _section_title(3, "System, model, and implementation versions", styles),
            _paragraph("System under test", styles["h2"]),
            _table(
                [
                    ["Field", "Value"],
                    ["Host", f"{system['host_model']} ({system['hostname_alias']})"],
                    ["OS and kernel", f"{system['os']} / {system['kernel']}"],
                    [
                        "Architecture and CPU",
                        f"{system['architecture']} / {system['cpu']}",
                    ],
                    ["Accelerator", system["accelerator"]],
                    [
                        "Driver and CUDA",
                        f"{system['driver_version']} / {system['cuda_version']}",
                    ],
                    [
                        "Physical unified memory",
                        _gib(system["physical_unified_memory_bytes"]),
                    ],
                    ["Power source", system["power_measurement_source"]],
                ],
                [doc_width * 0.28, doc_width * 0.72],
                styles,
            ),
            _paragraph("Model identity", styles["h2"]),
            _table(
                [
                    ["Field", "Value"],
                    ["Repository", model["repository"]],
                    ["Revision", model["revision"]],
                    ["Variant", model["variant"]],
                    ["Parameters", f"{model['parameter_count']:,}"],
                    ["Precision", model["precision"]],
                    ["Manifest SHA-256", model["manifest_sha256"]],
                ],
                [doc_width * 0.28, doc_width * 0.72],
                styles,
            ),
            _paragraph("Implementations", styles["h2"]),
            _table(
                [
                    ["Field", "Native", "SGLang"],
                    ["Name", engines["native"]["name"], engines["sglang"]["name"]],
                    [
                        "Version",
                        engines["native"]["version"],
                        engines["sglang"]["version"],
                    ],
                    [
                        "Source commit",
                        engines["native"]["source_commit"],
                        engines["sglang"]["source_commit"],
                    ],
                    [
                        "Image",
                        engines["native"]["container_image"],
                        engines["sglang"]["container_image"],
                    ],
                    [
                        "Image digest",
                        engines["native"]["image_digest"],
                        engines["sglang"]["image_digest"],
                    ],
                    [
                        "API",
                        engines["native"]["api_protocol"],
                        engines["sglang"]["api_protocol"],
                    ],
                    [
                        "Streaming",
                        engines["native"]["streaming_semantics"],
                        engines["sglang"]["streaming_semantics"],
                    ],
                ],
                [doc_width * 0.20, doc_width * 0.40, doc_width * 0.40],
                styles,
                long=True,
            ),
            PageBreak(),
        ]
    )

    latency_widths = [doc_width * 0.11, doc_width * 0.15] + [doc_width * 0.246] * 3
    story.extend(
        [
            _section_title(4, "Time to first audio", styles),
            _paragraph(
                "TTFA ends when the client receives the first playable PCM bytes. Percentiles "
                "use linear interpolation over successful measured requests.",
                styles["body"],
            ),
            _table(
                _metric_rows(
                    aggregates,
                    profiles,
                    (
                        ("p50", "ttfa_p50", _ms),
                        ("p95", "ttfa_p95", _ms),
                        ("p99", "ttfa_p99", _ms),
                    ),
                ),
                latency_widths,
                styles,
            ),
            Spacer(1, 8),
            KeepTogether(
                [
                    ProfileLineChart(
                        doc_width,
                        185,
                        profiles,
                        [
                            float(aggregates["native"][profile]["ttfa_p95"])
                            for profile in profiles
                        ],
                        [
                            float(aggregates["sglang"][profile]["ttfa_p95"])
                            for profile in profiles
                        ],
                        "TTFA p95 (ms)",
                    ),
                    _paragraph(
                        "Figure 1. TTFA p95 by concurrency. Native uses a solid line and circle markers; "
                        "SGLang uses a dashed line and square markers.",
                        styles["caption"],
                    ),
                ]
            ),
            PageBreak(),
            _section_title(5, "Real-time factor", styles),
            _paragraph(
                "Aggregate RTF is total scenario wall time divided by total successful audio "
                "duration. Summed-request-wall RTF is shown separately and must not be used as "
                "the B3/B6 throughput metric.",
                styles["body"],
            ),
            _table(
                _metric_rows(
                    aggregates,
                    profiles,
                    (
                        ("Aggregate", "aggregate_rtf", _number_text),
                        (
                            "Summed request wall",
                            "summed_request_wall_rtf",
                            _number_text,
                        ),
                    ),
                ),
                [
                    doc_width * 0.15,
                    doc_width * 0.19,
                    doc_width * 0.33,
                    doc_width * 0.33,
                ],
                styles,
            ),
            Spacer(1, 5),
            _paragraph("Successful per-request RTF distribution", styles["small"]),
            _table(
                _metric_rows(
                    aggregates,
                    profiles,
                    (
                        ("p50", "rtf_p50", _number_text),
                        ("p95", "rtf_p95", _number_text),
                        ("p99", "rtf_p99", _number_text),
                    ),
                ),
                latency_widths,
                styles,
            ),
            Spacer(1, 8),
            KeepTogether(
                [
                    ProfileLineChart(
                        doc_width,
                        185,
                        profiles,
                        [
                            float(aggregates["native"][profile]["aggregate_rtf"])
                            for profile in profiles
                        ],
                        [
                            float(aggregates["sglang"][profile]["aggregate_rtf"])
                            for profile in profiles
                        ],
                        "Aggregate RTF (scenario wall / audio)",
                    ),
                    _paragraph(
                        "Figure 2. Scenario aggregate RTF by concurrency. Values below 1.0 mean the "
                        "measured scenario produced audio faster than real time.",
                        styles["caption"],
                    ),
                ]
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _section_title(6, "Throughput", styles),
            _table(
                _metric_rows(
                    aggregates,
                    profiles,
                    (
                        (
                            "Successful requests",
                            "request_throughput_rps",
                            lambda value: f"{float(value):.3f} req/s",
                        ),
                        (
                            "Attempted requests",
                            "attempted_throughput_rps",
                            lambda value: f"{float(value):.3f} req/s",
                        ),
                        (
                            "Audio",
                            "audio_throughput_x",
                            lambda value: f"{float(value):.3f} x",
                        ),
                    ),
                ),
                [
                    doc_width * 0.17,
                    doc_width * 0.23,
                    doc_width * 0.30,
                    doc_width * 0.30,
                ],
                styles,
            ),
            Spacer(1, 8),
            KeepTogether(
                [
                    ProfileLineChart(
                        doc_width,
                        185,
                        profiles,
                        [
                            float(aggregates["native"][profile]["audio_throughput_x"])
                            for profile in profiles
                        ],
                        [
                            float(aggregates["sglang"][profile]["audio_throughput_x"])
                            for profile in profiles
                        ],
                        "Audio seconds per wall second",
                    ),
                    _paragraph(
                        "Figure 3. Aggregate audio throughput over each measured profile run.",
                        styles["caption"],
                    ),
                ]
            ),
            _section_title(7, "Memory", styles),
            _table(
                _metric_rows(
                    aggregates,
                    profiles,
                    (
                        ("Process RSS", "rss_peak_bytes", _gib),
                        ("GPU unified", "gpu_memory_peak_bytes", _gib),
                    ),
                ),
                [
                    doc_width * 0.17,
                    doc_width * 0.23,
                    doc_width * 0.30,
                    doc_width * 0.30,
                ],
                styles,
            ),
            Spacer(1, 8),
            KeepTogether(
                [
                    PatternBarChart(
                        doc_width,
                        185,
                        [f"{profile} RSS" for profile in profiles]
                        + [f"{profile} GPU" for profile in profiles],
                        [
                            float(aggregates["native"][profile]["rss_peak_bytes"])
                            / (1024**3)
                            for profile in profiles
                        ]
                        + [
                            float(
                                aggregates["native"][profile]["gpu_memory_peak_bytes"]
                            )
                            / (1024**3)
                            for profile in profiles
                        ],
                        [
                            float(aggregates["sglang"][profile]["rss_peak_bytes"])
                            / (1024**3)
                            for profile in profiles
                        ]
                        + [
                            float(
                                aggregates["sglang"][profile]["gpu_memory_peak_bytes"]
                            )
                            / (1024**3)
                            for profile in profiles
                        ],
                        "Peak memory (GiB)",
                    ),
                    _paragraph(
                        "Figure 4. Peak process RSS and GPU-visible unified memory. Native is solid; "
                        "SGLang is diagonal-hatched.",
                        styles["caption"],
                    ),
                ]
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _section_title(8, "Power and energy", styles),
            _table(
                _metric_rows(
                    aggregates,
                    profiles,
                    (
                        (
                            "Mean power",
                            "average_power_w",
                            lambda value: f"{float(value):.2f} W",
                        ),
                        (
                            "Peak power",
                            "peak_power_w",
                            lambda value: f"{float(value):.2f} W",
                        ),
                        (
                            "Energy/request",
                            "energy_per_request_j",
                            lambda value: f"{float(value):.2f} J",
                        ),
                        (
                            "Energy/audio s",
                            "energy_per_audio_second_j",
                            lambda value: f"{float(value):.2f} J/s",
                        ),
                    ),
                ),
                [doc_width * 0.10, doc_width * 0.13] + [doc_width * 0.1925] * 4,
                styles,
            ),
            Spacer(1, 8),
            KeepTogether(
                [
                    PatternBarChart(
                        doc_width,
                        185,
                        profiles,
                        [
                            float(
                                aggregates["native"][profile][
                                    "energy_per_audio_second_j"
                                ]
                            )
                            for profile in profiles
                        ],
                        [
                            float(
                                aggregates["sglang"][profile][
                                    "energy_per_audio_second_j"
                                ]
                            )
                            for profile in profiles
                        ],
                        "Energy per audio second (J/s)",
                    ),
                    _paragraph(
                        "Figure 5. Request energy normalized by successful emitted audio duration.",
                        styles["caption"],
                    ),
                ]
            ),
            _section_title(9, "Streaming cadence", styles),
            _table(
                _metric_rows(
                    aggregates,
                    profiles,
                    (
                        (
                            "Mean packets",
                            "packet_count_mean",
                            lambda value: f"{float(value):.2f}",
                        ),
                        ("Interval p50", "cadence_p50_ms", _ms),
                        ("Interval p95", "cadence_p95_ms", _ms),
                    ),
                ),
                [doc_width * 0.12, doc_width * 0.18] + [doc_width * 0.2333] * 3,
                styles,
            ),
            _paragraph(
                "A packet count of one yields no inter-packet intervals and is reported as "
                "0.00 ms cadence. The implementation-declared streaming semantics remain "
                "visible because an HTTP streaming response is not necessarily progressive "
                "model generation.",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    reliability_rows: list[list[str]] = [
        [
            "Profile",
            "Engine",
            "Total",
            "Success",
            "Error",
            "Timeout",
            "Cancelled",
            "Rate",
        ]
    ]
    eos_rows: list[list[str]] = [
        [
            "Profile",
            "Engine",
            "Natural EOS",
            "Length limited",
            "EOS unknown",
            "Validated policy",
        ]
    ]
    for profile in profiles:
        for engine, label in (("native", "Native"), ("sglang", "SGLang")):
            metric = aggregates[engine][profile]
            statuses: Counter[str] = metric["statuses"]  # type: ignore[assignment]
            reliability_rows.append(
                [
                    profile,
                    label,
                    str(metric["total"]),
                    str(statuses["success"]),
                    str(statuses["error"]),
                    str(statuses["timeout"]),
                    str(statuses["cancelled"]),
                    f"{float(metric['success_rate']) * 100:.2f}%",
                ]
            )
            if manifest["schema_version"] == "1.0":
                eos_rows.append(
                    [profile, label, "n/a", "n/a", "n/a", "Fixture field unavailable"]
                )
            else:
                eos_rows.append(
                    [
                        profile,
                        label,
                        str(metric["natural_eos"]),
                        str(metric["length_limited"]),
                        str(metric["eos_unknown"]),
                        "Natural EOS required"
                        if engine == "native"
                        else "Unknown retained",
                    ]
                )
    story.extend(
        [
            _section_title(10, "Reliability", styles),
            _table(
                reliability_rows,
                [
                    doc_width * fraction
                    for fraction in (0.10, 0.16, 0.11, 0.12, 0.11, 0.13, 0.14, 0.13)
                ],
                styles,
            ),
            Spacer(1, 9),
            _paragraph(
                "EOS policy is asymmetric by design: the Native protocol exposes a finish reason; "
                "stock SGLang raw PCM does not. Unknown is preserved rather than imputed.",
                styles["body"],
            ),
            _paragraph(
                "Boundary policy: every production workload entry uses an exact 20.48-second "
                "ceiling. Successful stock SGLang audio must remain strictly below 489,600 "
                "PCM samples and 20.40 seconds at 24 kHz. The exclusive 255-frame boundary "
                "rejects the off-by-one case in which max_new_tokens=256 can yield 255 "
                "decodable frames; a frame-aligned accepted response therefore contains at "
                "most 254 frames (20.32 seconds). This excludes the known length boundary but "
                "does not establish natural EOS, which remains unknown.",
                styles["body"],
            ),
            _table(
                eos_rows,
                [
                    doc_width * fraction
                    for fraction in (0.11, 0.16, 0.16, 0.16, 0.16, 0.25)
                ],
                styles,
            ),
            Spacer(1, 12),
            _section_title(11, "Startup time and image size", styles),
            _table(
                [
                    ["Implementation", "Startup", "Image size", "Image digest"],
                    [
                        "Native",
                        _ms(engines["native"]["startup_ms"]),
                        _size(engines["native"]["image_size_bytes"]),
                        engines["native"]["image_digest"],
                    ],
                    [
                        "SGLang",
                        _ms(engines["sglang"]["startup_ms"]),
                        _size(engines["sglang"]["image_size_bytes"]),
                        engines["sglang"]["image_digest"],
                    ],
                ],
                [
                    doc_width * 0.15,
                    doc_width * 0.15,
                    doc_width * 0.16,
                    doc_width * 0.54,
                ],
                styles,
            ),
            Spacer(1, 12),
            KeepTogether(
                [
                    PatternBarChart(
                        doc_width,
                        175,
                        ["Startup (s)"],
                        [float(engines["native"]["startup_ms"]) / 1000],
                        [float(engines["sglang"]["startup_ms"]) / 1000],
                        "Seconds",
                    ),
                    _paragraph(
                        "Figure 6. Ready-to-serve startup time.", styles["caption"]
                    ),
                ]
            ),
            _section_title(12, "Limitations", styles),
            *_bullet_list(manifest["limitations"], styles),
            PageBreak(),
        ]
    )

    evidence_rows = [["Role", "Engine", "Path", "Bytes", "SHA-256"]]
    for descriptor in sorted(
        manifest["evidence_files"],
        key=lambda item: (item["role"], item.get("engine_id", ""), item["path"]),
    ):
        evidence_rows.append(
            [
                descriptor["role"],
                descriptor.get("engine_id", "shared"),
                descriptor["path"],
                f"{descriptor['bytes']:,}",
                descriptor["sha256"],
            ]
        )
    weight_rows = [["Weight file", "Bytes", "SHA-256"]]
    for weight in sorted(model["weight_files"], key=lambda item: item["path"]):
        weight_rows.append([weight["path"], f"{weight['bytes']:,}", weight["sha256"]])
    count_rows = [["Profile", "Engine", "Rows", "Successful", "Required minimum"]]
    for profile in profiles:
        for engine, label in (("native", "Native"), ("sglang", "SGLang")):
            metric = aggregates[engine][profile]
            count_rows.append(
                [
                    profile,
                    label,
                    str(metric["total"]),
                    str(metric["success"]),
                    str(manifest["workload"]["minimum_measured_requests_per_profile"]),
                ]
            )
    story.extend(
        [
            _section_title(13, "Raw evidence appendix", styles),
            _paragraph(
                "The following digests are the immutable audit trail used by this report. "
                "The generator verified byte lengths and SHA-256 values before parsing any "
                "record.",
                styles["body"],
            ),
            _paragraph("Evidence files", styles["h2"]),
            _table(
                evidence_rows,
                [doc_width * fraction for fraction in (0.17, 0.10, 0.23, 0.10, 0.40)],
                styles,
                long=True,
            ),
            _paragraph("Model artifacts", styles["h2"]),
            _table(
                weight_rows,
                [doc_width * 0.34, doc_width * 0.16, doc_width * 0.50],
                styles,
                long=True,
            ),
            _paragraph("Validated workload cardinality", styles["h2"]),
            _table(
                count_rows,
                [doc_width * fraction for fraction in (0.14, 0.22, 0.18, 0.20, 0.26)],
                styles,
            ),
            Spacer(1, 8),
            _paragraph(f"Manifest SHA-256: {bundle.manifest_sha256}", styles["mono"]),
            _paragraph(
                f"Schema: {SCHEMA_PATH.name} version {manifest['schema_version']}",
                styles["small"],
            ),
        ]
    )
    return story


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_output(bundle: Bundle, requested: Path | None, overwrite: bool) -> Path:
    fixture = bundle.manifest["evidence_kind"] == "test_fixture"
    if requested is None:
        if fixture:
            _fail(
                "output",
                "test fixtures require an explicit --output outside reports/output",
            )
        output = OUTPUT_DIR / f"{bundle.manifest['report']['benchmark_id']}.pdf"
    else:
        output = requested.expanduser().resolve()
    if output.suffix.lower() != ".pdf":
        _fail("output", "must use a .pdf extension")
    if fixture and _is_within(output, OUTPUT_DIR):
        _fail("output", "test fixtures cannot be written to reports/output")
    if output.exists() and not overwrite:
        _fail("output", f"already exists: {output}; pass --overwrite to replace it")
    if output.exists() and not output.is_file():
        _fail("output", "existing path is not a regular file")
    return output


def build_pdf(bundle: Bundle, output: Path, overwrite: bool = False) -> Path:
    if REPORTLAB_IMPORT_ERROR is not None:
        raise EvidenceError(
            "ReportLab is required for PDF generation. Use reports/requirements-report.txt "
            f"or the bundled PDF runtime. Import error: {REPORTLAB_IMPORT_ERROR}"
        )
    if output.exists() and not overwrite:
        _fail("output", f"already exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    TMP_PDF_DIR.mkdir(parents=True, exist_ok=True)
    build_path = (
        TMP_PDF_DIR
        / f".{bundle.manifest['report']['benchmark_id']}.{os.getpid()}.build.pdf"
    )
    page_width, page_height = A4
    left_margin = 18 * mm
    right_margin = 18 * mm
    top_margin = 24 * mm
    bottom_margin = 18 * mm
    doc = BaseDocTemplate(
        str(build_path),
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title=_ascii_hyphens(bundle.manifest["report"]["title"]),
        author=_ascii_hyphens(", ".join(bundle.manifest["report"]["authors"])),
        subject="Validated Native versus SGLang Qwen3-TTS benchmark",
        creator="qwen3-tts-native deterministic report pipeline",
    )
    frame = Frame(
        left_margin,
        bottom_margin,
        page_width - left_margin - right_margin,
        page_height - top_margin - bottom_margin,
        id="body",
    )
    fixture = bundle.manifest["evidence_kind"] == "test_fixture"
    report_title = _ascii_hyphens(bundle.manifest["report"]["title"])
    benchmark_id = bundle.manifest["report"]["benchmark_id"]

    def on_page(c: Any, document: Any) -> None:
        c.saveState()
        c.setTitle(report_title)
        c.setAuthor(_ascii_hyphens(", ".join(bundle.manifest["report"]["authors"])))
        c.setSubject("Validated Native versus SGLang Qwen3-TTS benchmark")
        c.setCreator("qwen3-tts-native deterministic report pipeline")
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.line(
            left_margin,
            page_height - 16 * mm,
            page_width - right_margin,
            page_height - 16 * mm,
        )
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 7)
        c.drawString(left_margin, page_height - 12.8 * mm, report_title[:88])
        c.setFont("Helvetica", 6.5)
        footer = f"{benchmark_id} | manifest {bundle.manifest_sha256[:12]} | Page {document.page}"
        c.drawString(left_margin, 9 * mm, footer)
        c.line(left_margin, 12 * mm, page_width - right_margin, 12 * mm)
        if fixture:
            c.setFillColor(colors.black)
            c.rect(
                left_margin,
                page_height - 21 * mm,
                page_width - left_margin - right_margin,
                4.5 * mm,
                stroke=0,
                fill=1,
            )
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 7)
            c.drawCentredString(
                page_width / 2, page_height - 19.5 * mm, TEST_FIXTURE_BANNER
            )
        c.restoreState()

    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=on_page)])

    class InvariantCanvas(pdfcanvas.Canvas):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["invariant"] = 1
            kwargs["pageCompression"] = 1
            super().__init__(*args, **kwargs)

    story = _build_story(bundle, page_width - left_margin - right_margin)
    try:
        doc.build(story, canvasmaker=InvariantCanvas)
        if not build_path.is_file() or build_path.stat().st_size < 1024:
            _fail("output", "ReportLab did not produce a valid PDF-sized artifact")
        try:
            os.replace(build_path, output)
        except OSError:
            shutil.copyfile(build_path, output)
            build_path.unlink()
    finally:
        if build_path.exists():
            build_path.unlink()
    return output


def generate(
    manifest_path: Path,
    output: Path | None = None,
    allow_test_fixture: bool = False,
    overwrite: bool = False,
    validate_only: bool = False,
) -> tuple[Bundle, Path | None]:
    bundle = load_bundle(manifest_path, allow_test_fixture=allow_test_fixture)
    aggregate(bundle)
    if validate_only:
        return bundle, None
    resolved_output = resolve_output(bundle, output, overwrite)
    return bundle, build_pdf(bundle, resolved_output, overwrite=overwrite)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Native versus SGLang evidence and build a monochrome PDF report."
    )
    parser.add_argument(
        "manifest", type=Path, help="Path to the evidence manifest JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PDF output path; production defaults to reports/output",
    )
    parser.add_argument(
        "--allow-test-fixture",
        action="store_true",
        help="Allow explicitly marked synthetic layout fixtures",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and aggregate without writing a PDF",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output PDF"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bundle, output = generate(
            args.manifest,
            output=args.output,
            allow_test_fixture=args.allow_test_fixture,
            overwrite=args.overwrite,
            validate_only=args.validate_only,
        )
    except EvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Validated {bundle.manifest['evidence_kind']} evidence: "
        f"{len(bundle.requests)} canonical requests, "
        f"{len(bundle.measurements['native'])} Native rows, "
        f"{len(bundle.measurements['sglang'])} SGLang rows."
    )
    if output is not None:
        print(f"Wrote PDF: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
