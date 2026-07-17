#!/usr/bin/env python3
"""Assemble one immutable schema-v1.2 production evidence manifest.

This tool intentionally uses only the Python standard library.  It does not
normalize, repair, or infer benchmark metadata.  Static claims come from an
explicit configuration file; observed file identities and run resources come
from the qualifying-run directories.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn


SCHEMA_VERSION = "1.2"
EVIDENCE_KIND = "production"
RUN_SCHEMA_VERSION = "qwen3-tts-qualifying-run/v1"
CLIENT_SCHEMA_VERSION = "qwen3-tts-http-bench/v1"
AUDIT_SCHEMA_VERSION = "qwen3-tts-spark-resource-audit/v1"
MODEL_ARTIFACT_SCHEMA_VERSION = "qwen3-tts-model-artifact/v1"
REGISTRY_METADATA_SCHEMA_VERSION = "qwen3-tts-registry-image/v1"
ENGINES = ("native", "sglang")
PROFILES = {"B1": 1, "B3": 3, "B6": 6}
ROUNDS = (1, 2)
EXPECTED_RUN_KEYS = {
    (engine, profile, round_number)
    for engine in ENGINES
    for profile in PROFILES
    for round_number in ROUNDS
}
TELEMETRY_RELATIVE_PATHS = (
    "raw/gpu.csv",
    "raw/system.csv",
    "raw/process-rss.csv",
    "raw/process-rss-total.csv",
    "raw/gpu-processes.csv",
    "raw/gpu-process-summary.csv",
    "raw/phase-events.jsonl",
    "raw/run.txt",
)
AUDIT_SOURCE_PATHS = (
    "raw/phase-events.jsonl",
    "raw/run.txt",
    "raw/gpu.csv",
    "raw/system.csv",
    "raw/process-rss.csv",
    "raw/process-rss-total.csv",
    "raw/gpu-processes.csv",
    "raw/gpu-process-summary.csv",
    "client/summary.json",
    "raw/command.stdout",
    "raw/command.stderr",
)
CLIENT_FILES = {
    "client/summary.json": ("client_summary", "json"),
    "client/requests.jsonl": ("client_requests", "jsonl"),
    "client/packets.jsonl": ("client_packets", "jsonl"),
}
RAW_FORMATS = {"json", "jsonl", "csv", "txt", "log", "stdout", "stderr"}
CONFIG_KEYS = {
    "report",
    "system",
    "model",
    "workload",
    "implementations",
    "methodology",
    "limitations",
}
WORKLOAD_CONFIG_KEYS = {
    "corpus_sha256",
    "ordered_seeds",
    "sample_rate_hz",
    "channels",
    "sample_format",
    "response_mode",
    "warmup_requests_per_run",
    "minimum_measured_requests_per_profile",
    "minimum_rounds_per_subject",
    "profiles",
    "language_policy",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CHECKSUM_RE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
WORKLOAD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
PRODUCTION_DURATION_SECONDS = 20.48
SAMPLE_RATE_HZ = 24_000
SAMPLES_PER_CODEC_FRAME = 1_920
SGLANG_EXCLUSIVE_CODEC_FRAME_LIMIT = 255
SGLANG_EXCLUSIVE_SAMPLE_LIMIT = (
    SGLANG_EXCLUSIVE_CODEC_FRAME_LIMIT * SAMPLES_PER_CODEC_FRAME
)
SGLANG_EXCLUSIVE_DURATION_LIMIT_SECONDS = SGLANG_EXCLUSIVE_SAMPLE_LIMIT / SAMPLE_RATE_HZ


class AssemblyError(ValueError):
    """Raised when the evidence cannot be assembled without guessing."""


class SchemaViolation(AssemblyError):
    """Raised when the assembled manifest does not satisfy the JSON schema."""


@dataclass(frozen=True)
class FileIdentity:
    sha256: str
    bytes: int


@dataclass(frozen=True)
class RunBundle:
    key: tuple[str, str, int]
    directory: Path
    evidence_prefix: str
    identities: dict[str, FileIdentity]
    invocation: dict[str, Any]
    summary: dict[str, Any]
    resource: dict[str, Any]


@dataclass(frozen=True)
class BoundEvidence:
    role: str
    engine: str
    path: str
    identity: FileIdentity
    payload: dict[str, Any]


def _fail(message: str) -> NoReturn:
    raise AssemblyError(message)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssemblyError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise AssemblyError(f"non-finite JSON number is forbidden: {value}")


def _parse_json_bytes(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssemblyError(f"{label}: expected UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, AssemblyError) as exc:
        raise AssemblyError(f"{label}: invalid JSON: {exc}") from exc


def _parse_jsonl(path: Path, root: Path, label: str) -> list[Any]:
    payload = _read_regular(path, root, require_nonempty=True)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssemblyError(f"{label}: expected UTF-8 JSONL") from exc
    records: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            _fail(f"{label}:{line_number}: blank JSONL records are forbidden")
        try:
            record = json.loads(
                line,
                object_pairs_hook=_strict_pairs,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, AssemblyError) as exc:
            raise AssemblyError(f"{label}:{line_number}: invalid JSON: {exc}") from exc
        records.append(record)
    if not records:
        _fail(f"{label}: expected at least one JSONL record")
    return records


def _strict_object(
    value: Any,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label}: expected an object")
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        _fail(f"{label}: key mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    return value


def _integer(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label}: expected an integer")
    if minimum is not None and value < minimum:
        _fail(f"{label}: expected at least {minimum}, observed {value}")
    return value


def _number(value: Any, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label}: expected a finite number")
    if minimum is not None and result < minimum:
        _fail(f"{label}: expected at least {minimum}, observed {result}")
    return result


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label}: expected a non-empty string")
    return value


def _safe_relative_posix(value: str, label: str) -> PurePosixPath:
    _string(value, label)
    if "\\" in value or "\n" in value or "\r" in value:
        _fail(f"{label}: expected a normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        _fail(f"{label}: expected a normalized relative POSIX path")
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{label}: unsafe path component")
    return path


def _path_inside(path: Path, base: Path, label: str) -> str:
    try:
        relative = path.relative_to(base)
    except ValueError as exc:
        raise AssemblyError(f"{label}: path must remain inside {base}") from exc
    result = relative.as_posix()
    _safe_relative_posix(result, label)
    return result


def _reject_symlink(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise AssemblyError(f"{label}: path does not exist: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        _fail(f"{label}: symlinks are forbidden: {path}")


def _validate_tree(root: Path, label: str) -> None:
    _reject_symlink(root, label)
    if not root.is_dir():
        _fail(f"{label}: expected a directory: {root}")
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = current / name
            _reject_symlink(child, label)
            if not child.is_dir():
                _fail(f"{label}: expected a directory: {child}")
        for name in file_names:
            child = current / name
            _reject_symlink(child, label)
            if not child.is_file():
                _fail(f"{label}: expected a regular file: {child}")


def _open_regular(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AssemblyError(f"cannot open regular file safely: {path}: {exc}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        _fail(f"expected a regular file: {path}")
    return descriptor, metadata


def _read_regular(path: Path, root: Path, *, require_nonempty: bool) -> bytes:
    _path_inside(path, root, str(path))
    _reject_symlink(path, str(path))
    descriptor, metadata = _open_regular(path)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            payload = stream.read()
    except OSError as exc:
        raise AssemblyError(f"failed to read {path}: {exc}") from exc
    if len(payload) != metadata.st_size:
        _fail(f"file changed while it was read: {path}")
    if require_nonempty and not payload:
        _fail(f"required file is empty: {path}")
    return payload


def _hash_regular(path: Path, root: Path) -> FileIdentity:
    _path_inside(path, root, str(path))
    _reject_symlink(path, str(path))
    descriptor, before = _open_regular(path)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise AssemblyError(f"failed to hash {path}: {exc}") from exc
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ) or count != before.st_size:
        _fail(f"file changed while it was hashed: {path}")
    return FileIdentity(digest.hexdigest(), count)


def _load_json_file(path: Path, root: Path, label: str) -> Any:
    return _parse_json_bytes(
        _read_regular(path, root, require_nonempty=True),
        label,
    )


def _load_config(path: Path) -> dict[str, Any]:
    _reject_symlink(path, "config")
    if not path.is_file():
        _fail(f"config: expected a regular file: {path}")
    payload = path.read_bytes()
    config = _parse_json_bytes(payload, str(path))
    return _strict_object(config, "config", CONFIG_KEYS)


def _validate_workload(
    path: Path, evidence_root: Path
) -> tuple[FileIdentity, tuple[int, ...]]:
    records = _parse_jsonl(path, evidence_root, "workload")
    identifiers: set[str] = set()
    ordered_seeds: list[int] = []
    for index, raw in enumerate(records, 1):
        label = f"workload:{index}"
        item = _strict_object(
            raw,
            label,
            {"id", "text", "voice_description"},
            {"language", "seed", "max_duration_seconds", "sampling", "stream"},
        )
        identifier = _string(item["id"], f"{label}.id")
        if not WORKLOAD_ID_RE.fullmatch(identifier):
            _fail(f"{label}.id: invalid canonical workload identifier")
        if identifier in identifiers:
            _fail(f"{label}.id: duplicate identifier {identifier!r}")
        identifiers.add(identifier)
        _string(item["text"], f"{label}.text")
        _string(item["voice_description"], f"{label}.voice_description")
        ordered_seeds.append(_integer(item.get("seed"), f"{label}.seed", 0))
        if item.get("stream") is not True:
            _fail(f"{label}.stream: production evidence requires true")
        duration = _number(
            item.get("max_duration_seconds"),
            f"{label}.max_duration_seconds",
        )
        if duration != PRODUCTION_DURATION_SECONDS:
            _fail(
                f"{label}.max_duration_seconds: production evidence requires "
                f"exactly {PRODUCTION_DURATION_SECONDS}"
            )
    return _hash_regular(path, evidence_root), tuple(ordered_seeds)


def _load_bound_json_evidence(
    reference: dict[str, Any],
    evidence_root: Path,
    label: str,
) -> tuple[str, FileIdentity, dict[str, Any]]:
    reference = _strict_object(reference, label, {"path", "sha256"})
    relative = _safe_relative_posix(reference["path"], f"{label}.path").as_posix()
    if not relative.endswith(".json"):
        _fail(f"{label}.path: evidence must be a JSON file")
    path = evidence_root / relative
    if not path.exists() or not path.is_file():
        _fail(
            f"{label}.path: required digest-bound evidence is unavailable: {relative}"
        )
    identity = _hash_regular(path, evidence_root)
    if identity.sha256 != reference["sha256"]:
        _fail(
            f"{label}.sha256: declared {reference['sha256']}, "
            f"observed {identity.sha256}"
        )
    payload = _load_json_file(path, evidence_root, label)
    if not isinstance(payload, dict):
        _fail(f"{label}: expected a JSON object")
    return relative, identity, payload


def _validate_artifact_weights(artifact: dict[str, Any], label: str) -> None:
    weights = artifact.get("weight_files")
    if not isinstance(weights, list) or not weights:
        _fail(f"{label}.weight_files: expected at least one weight file")
    paths: set[str] = set()
    precisions: set[str] = set()
    parameter_total = 0
    for index, raw in enumerate(weights):
        item_label = f"{label}.weight_files[{index}]"
        item = _strict_object(
            raw,
            item_label,
            {"path", "sha256", "bytes", "parameter_count", "precision"},
        )
        path = _safe_relative_posix(item["path"], f"{item_label}.path").as_posix()
        if path in paths:
            _fail(f"{item_label}.path: duplicate artifact path")
        paths.add(path)
        if not SHA256_RE.fullmatch(str(item["sha256"])):
            _fail(f"{item_label}.sha256: invalid digest")
        _integer(item["bytes"], f"{item_label}.bytes", 1)
        parameter_total += _integer(
            item["parameter_count"], f"{item_label}.parameter_count", 1
        )
        precisions.add(_string(item["precision"], f"{item_label}.precision"))
    if parameter_total != artifact.get("parameter_count"):
        _fail(
            f"{label}.parameter_count: expected the exact sum of weight-file "
            f"parameter counts ({parameter_total})"
        )
    declared_precisions = artifact.get("precision")
    if not isinstance(declared_precisions, list) or declared_precisions != sorted(
        precisions
    ):
        _fail(
            f"{label}.precision: expected the sorted unique weight precision set "
            f"{sorted(precisions)}"
        )


def _load_model_artifact_evidence(
    implementation: dict[str, Any],
    common_model: dict[str, Any],
    evidence_root: Path,
) -> BoundEvidence:
    engine = implementation["id"]
    artifact = implementation["model_artifact"]
    label = f"config.implementations[{engine}].model_artifact"
    relative, identity, payload = _load_bound_json_evidence(
        artifact["evidence"], evidence_root, f"{label}.evidence"
    )
    payload = _strict_object(
        payload,
        f"{label}.evidence_payload",
        {
            "schema_version",
            "implementation_id",
            "local_image_id",
            "repository",
            "revision",
            "variant",
            "parameter_count",
            "precision",
            "manifest_sha256",
            "weight_files",
            "source",
        },
    )
    if payload["schema_version"] != MODEL_ARTIFACT_SCHEMA_VERSION:
        _fail(f"{label}.evidence_payload.schema_version: unexpected schema")
    if payload["implementation_id"] != engine:
        _fail(f"{label}.evidence_payload.implementation_id: engine mismatch")
    if payload["local_image_id"] != implementation["local_image"]["id"]:
        _fail(
            f"{label}.evidence_payload.local_image_id: does not bind the tested "
            "local Docker image ID"
        )
    for field in (
        "repository",
        "revision",
        "variant",
        "parameter_count",
        "precision",
        "manifest_sha256",
        "weight_files",
    ):
        if payload[field] != artifact[field]:
            _fail(f"{label}.{field}: differs from digest-bound artifact evidence")
    for field in ("repository", "revision", "variant"):
        if artifact[field] != common_model[field]:
            _fail(f"{label}.{field}: must equal config.model.{field}")
    manifest_digest = artifact["manifest_sha256"]
    if manifest_digest is not None and not SHA256_RE.fullmatch(str(manifest_digest)):
        _fail(f"{label}.manifest_sha256: invalid digest")
    _integer(artifact["parameter_count"], f"{label}.parameter_count", 1)
    _validate_artifact_weights(artifact, label)
    source = _strict_object(
        payload["source"],
        f"{label}.evidence_payload.source",
        {"kind", "container_path", "read_only"},
        {"host_path", "snapshot_path", "revision_ref_path"},
    )
    if source["kind"] not in {"container_image", "read_only_bind_mount"}:
        _fail(f"{label}.evidence_payload.source.kind: unsupported source")
    _string(source["container_path"], f"{label}.evidence_payload.source.container_path")
    if source["read_only"] is not True:
        _fail(f"{label}.evidence_payload.source.read_only: expected true")
    for field in ("host_path", "snapshot_path", "revision_ref_path"):
        if field in source:
            _string(source[field], f"{label}.evidence_payload.source.{field}")
    if engine == "sglang" and source["kind"] != "read_only_bind_mount":
        _fail(
            f"{label}.evidence_payload.source.kind: stock SGLang weights must "
            "identify the observed read-only bind mount"
        )
    return BoundEvidence("model_artifact", engine, relative, identity, payload)


def _load_registry_evidence(
    implementation: dict[str, Any], evidence_root: Path
) -> BoundEvidence | None:
    registry = implementation.get("registry_image")
    if registry is None:
        return None
    engine = implementation["id"]
    label = f"config.implementations[{engine}].registry_image"
    relative, identity, payload = _load_bound_json_evidence(
        registry["evidence"], evidence_root, f"{label}.evidence"
    )
    payload = _strict_object(
        payload,
        f"{label}.evidence_payload",
        {
            "schema_version",
            "implementation_id",
            "local_image_id",
            "reference",
            "manifest_digest",
        },
        {"compressed_size_bytes"},
    )
    if payload["schema_version"] != REGISTRY_METADATA_SCHEMA_VERSION:
        _fail(f"{label}.evidence_payload.schema_version: unexpected schema")
    expected = {
        "implementation_id": engine,
        "local_image_id": implementation["local_image"]["id"],
        "reference": registry["reference"],
        "manifest_digest": registry["manifest_digest"],
    }
    for field, value in expected.items():
        if payload[field] != value:
            _fail(f"{label}.{field}: differs from digest-bound registry evidence")
    if registry.get("compressed_size_bytes") != payload.get("compressed_size_bytes"):
        _fail(
            f"{label}.compressed_size_bytes: differs from digest-bound registry evidence"
        )
    if "compressed_size_bytes" in registry:
        _integer(registry["compressed_size_bytes"], f"{label}.compressed_size_bytes", 1)
    if not IMAGE_DIGEST_RE.fullmatch(str(registry["manifest_digest"])):
        _fail(f"{label}.manifest_digest: invalid OCI digest")
    return BoundEvidence("registry_metadata", engine, relative, identity, payload)


def _load_bound_implementation_evidence(
    config: dict[str, Any], evidence_root: Path
) -> list[BoundEvidence]:
    bound: list[BoundEvidence] = []
    seen_paths: set[str] = set()
    for implementation in config["implementations"]:
        items = [
            _load_model_artifact_evidence(
                implementation, config["model"], evidence_root
            ),
            _load_registry_evidence(implementation, evidence_root),
        ]
        for item in items:
            if item is None:
                continue
            if item.path in seen_paths:
                _fail(f"duplicate bound evidence path: {item.path}")
            seen_paths.add(item.path)
            bound.append(item)
    return bound


def _parse_checksum_inventory(run_dir: Path) -> dict[str, FileIdentity]:
    checksum_path = run_dir / "SHA256SUMS"
    payload = _read_regular(checksum_path, run_dir, require_nonempty=True)
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise AssemblyError(
            f"{checksum_path}: expected UTF-8 checksum inventory"
        ) from exc
    declared: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            _fail(f"{checksum_path}:{line_number}: malformed sha256sum record")
        digest, relative = match.groups()
        _safe_relative_posix(relative, f"{checksum_path}:{line_number}")
        if relative == "SHA256SUMS":
            _fail(f"{checksum_path}:{line_number}: inventory must not hash itself")
        if relative in declared:
            _fail(f"{checksum_path}:{line_number}: duplicate path {relative!r}")
        declared[relative] = digest

    observed_paths = {
        file.relative_to(run_dir).as_posix()
        for directory, _, names in os.walk(run_dir, followlinks=False)
        for name in names
        if (file := Path(directory) / name) != checksum_path
    }
    if set(declared) != observed_paths:
        _fail(
            f"{checksum_path}: inventory mismatch; "
            f"missing={sorted(observed_paths - set(declared))} "
            f"extra={sorted(set(declared) - observed_paths)}"
        )

    identities: dict[str, FileIdentity] = {}
    for relative in sorted(declared):
        identity = _hash_regular(run_dir / relative, run_dir)
        if identity.sha256 != declared[relative]:
            _fail(
                f"{run_dir / relative}: digest mismatch; declared "
                f"{declared[relative]}, observed {identity.sha256}"
            )
        identities[relative] = identity
    return identities


def _discover_run_directories(runs_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for directory, _, names in os.walk(runs_root, followlinks=False):
        current = Path(directory)
        if "run-resource.json" in names:
            candidates.add(current)
        if current.name == "provenance" and "invocation.json" in names:
            candidates.add(current.parent)
    return sorted(candidates, key=lambda item: item.as_posix())


def _validate_run_tree_ownership(runs_root: Path, run_dirs: list[Path]) -> None:
    """Reject every runs-root entry that is not owned by one discovered run."""

    for index, run_dir in enumerate(run_dirs):
        for other in run_dirs[index + 1 :]:
            if run_dir in other.parents or other in run_dir.parents:
                _fail(
                    "qualifying-run directories must not overlap: "
                    f"{run_dir} and {other}"
                )

    ancestors = {runs_root}
    for run_dir in run_dirs:
        current = run_dir.parent
        while current != runs_root:
            ancestors.add(current)
            current = current.parent

    def owned(path: Path) -> bool:
        return any(path == run_dir or run_dir in path.parents for run_dir in run_dirs)

    for directory, directory_names, file_names in os.walk(runs_root, followlinks=False):
        current = Path(directory)
        if current not in ancestors and not owned(current):
            _fail(f"unexpected directory outside a qualifying run: {current}")
        for name in directory_names:
            child = current / name
            if child not in ancestors and not owned(child):
                _fail(f"unexpected directory outside a qualifying run: {child}")
        for name in file_names:
            child = current / name
            if not owned(child):
                _fail(f"unexpected file outside a qualifying run: {child}")


def _validate_invocation_shape(value: Any, label: str) -> dict[str, Any]:
    invocation = _strict_object(
        value,
        label,
        {
            "schema_version",
            "engine",
            "profile",
            "round",
            "container",
            "image",
            "client",
            "workload",
            "evidence_prefix",
            "request",
            "telemetry",
            "tooling_repository",
        },
    )
    _strict_object(invocation["container"], f"{label}.container", {"name", "id"})
    _strict_object(invocation["image"], f"{label}.image", {"reference", "resolved_id"})
    _strict_object(invocation["client"], f"{label}.client", {"path", "sha256"})
    _strict_object(invocation["workload"], f"{label}.workload", {"path", "sha256"})
    _strict_object(
        invocation["request"],
        f"{label}.request",
        {"endpoint", "requests", "warmups", "timeout_seconds", "sglang_model"},
    )
    _strict_object(
        invocation["telemetry"],
        f"{label}.telemetry",
        {
            "idle_baseline_seconds",
            "configured_sample_interval_ms",
            "maximum_qualifying_observed_gap_ms",
            "gpu_index",
        },
    )
    _strict_object(
        invocation["tooling_repository"],
        f"{label}.tooling_repository",
        {"commit", "tracked_files_clean"},
    )
    return invocation


def _validate_resource_shape(value: Any, label: str) -> dict[str, Any]:
    resource = _strict_object(
        value,
        label,
        {
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
        },
    )
    _integer(resource["process_rss_peak_bytes"], f"{label}.process_rss_peak_bytes", 1)
    _integer(
        resource["gpu_unified_memory_peak_bytes"],
        f"{label}.gpu_unified_memory_peak_bytes",
        1,
    )
    average = _number(resource["average_power_w"], f"{label}.average_power_w", 0)
    peak = _number(resource["peak_power_w"], f"{label}.peak_power_w", 0)
    if peak < average:
        _fail(f"{label}.peak_power_w: must be at least average_power_w")
    _number(resource["energy_j"], f"{label}.energy_j", 0)
    _integer(resource["sampling_interval_ms"], f"{label}.sampling_interval_ms", 1)
    if (
        _integer(
            resource["competing_cuda_processes"],
            f"{label}.competing_cuda_processes",
            0,
        )
        != 0
    ):
        _fail(f"{label}.competing_cuda_processes: production requires zero")
    paths = resource["telemetry_evidence_paths"]
    if not isinstance(paths, list) or not paths:
        _fail(f"{label}.telemetry_evidence_paths: expected a non-empty array")
    if len(paths) != len(set(paths)):
        _fail(f"{label}.telemetry_evidence_paths: duplicate paths")
    for index, path in enumerate(paths):
        _safe_relative_posix(path, f"{label}.telemetry_evidence_paths[{index}]")
    return resource


def _validate_resource_audit(
    audit: Any,
    label: str,
    key: tuple[str, str, int],
    identities: dict[str, FileIdentity],
    resource: dict[str, Any],
) -> None:
    item = _strict_object(
        audit,
        label,
        {
            "schema_version",
            "engine_id",
            "profile_id",
            "round",
            "phase_boundaries",
            "sampling",
            "power",
            "memory",
            "source_files",
        },
    )
    if item["schema_version"] != AUDIT_SCHEMA_VERSION or key != (
        item["engine_id"],
        item["profile_id"],
        item["round"],
    ):
        _fail(f"{label}: identity or schema mismatch")
    _strict_object(
        item["phase_boundaries"],
        f"{label}.phase_boundaries",
        {
            "idle_start_wall_time_unix_ns",
            "idle_end_wall_time_unix_ns",
            "measured_start_wall_time_unix_ns",
            "measured_end_wall_time_unix_ns",
            "idle_duration_seconds",
            "measured_monotonic_duration_seconds",
            "measured_wall_duration_seconds",
        },
    )
    sampling = _strict_object(
        item["sampling"],
        f"{label}.sampling",
        {
            "configured_interval_ms",
            "maximum_allowed_observed_gap_ms",
            "idle_power_samples",
            "measured_power_samples",
            "measured_process_rss_samples",
            "measured_gpu_process_samples",
            "measured_system_samples",
        },
    )
    if sampling["configured_interval_ms"] != resource["sampling_interval_ms"]:
        _fail(f"{label}.sampling.configured_interval_ms: differs from run-resource")
    if sampling["maximum_allowed_observed_gap_ms"] != 200:
        _fail(f"{label}.sampling.maximum_allowed_observed_gap_ms: expected 200")
    power = _strict_object(
        item["power"],
        f"{label}.power",
        {
            "source",
            "integration",
            "idle_average_power_w",
            "idle_peak_power_w",
            "idle_gross_energy_j",
            "measured_average_power_w",
            "measured_peak_power_w",
            "measured_gross_energy_j",
            "measured_idle_adjusted_energy_j",
            "idle_adjustment",
        },
    )
    power_comparisons = {
        "measured_average_power_w": "average_power_w",
        "measured_peak_power_w": "peak_power_w",
        "measured_idle_adjusted_energy_j": "energy_j",
    }
    for audit_field, resource_field in power_comparisons.items():
        if power[audit_field] != resource[resource_field]:
            _fail(f"{label}.power.{audit_field}: differs from run-resource")
    memory = _strict_object(
        item["memory"],
        f"{label}.memory",
        {
            "process_rss_definition",
            "process_rss_peak_bytes",
            "gpu_unified_memory_definition",
            "gpu_unified_memory_peak_bytes",
            "cgroup_memory_definition",
            "cgroup_memory_current_peak_bytes",
            "host_mem_available_min_kib",
            "host_swap_free_start_kib",
            "host_swap_free_end_kib",
        },
    )
    if memory["process_rss_peak_bytes"] != resource["process_rss_peak_bytes"]:
        _fail(f"{label}.memory.process_rss_peak_bytes: differs from run-resource")
    if (
        memory["gpu_unified_memory_peak_bytes"]
        != resource["gpu_unified_memory_peak_bytes"]
    ):
        _fail(
            f"{label}.memory.gpu_unified_memory_peak_bytes: differs from run-resource"
        )

    sources = item["source_files"]
    if not isinstance(sources, list) or len(sources) != len(AUDIT_SOURCE_PATHS):
        _fail(f"{label}.source_files: expected exactly {len(AUDIT_SOURCE_PATHS)} files")
    observed_paths: list[str] = []
    for index, raw in enumerate(sources):
        source = _strict_object(
            raw,
            f"{label}.source_files[{index}]",
            {"path", "sha256", "bytes"},
        )
        path = source["path"]
        _safe_relative_posix(path, f"{label}.source_files[{index}].path")
        observed_paths.append(path)
        expected = identities.get(path)
        if expected is None:
            _fail(f"{label}.source_files[{index}].path: missing from run inventory")
        if source["sha256"] != expected.sha256 or source["bytes"] != expected.bytes:
            _fail(f"{label}.source_files[{index}]: digest or byte count mismatch")
    if observed_paths != list(AUDIT_SOURCE_PATHS):
        _fail(f"{label}.source_files: paths or reducer order differ")


def _validate_summary_and_requests(
    run_dir: Path,
    engine: str,
    profile: str,
    invocation: dict[str, Any],
    minimum_successes: int,
    minimum_warmups: int,
) -> dict[str, Any]:
    label = f"{run_dir}/client/summary.json"
    summary = _load_json_file(run_dir / "client/summary.json", run_dir, label)
    if not isinstance(summary, dict):
        _fail(f"{label}: expected an object")
    expected_backend = "native" if engine == "native" else "sglang-omni"
    expected_values = {
        "schema_version": CLIENT_SCHEMA_VERSION,
        "backend": expected_backend,
        "concurrency": profile,
        "warmups": invocation["request"]["warmups"],
        "planned_requests": invocation["request"]["requests"],
    }
    for field, expected in expected_values.items():
        if summary.get(field) != expected:
            _fail(
                f"{label}.{field}: expected {expected!r}, observed {summary.get(field)!r}"
            )
    if _integer(summary.get("warmups"), f"{label}.warmups", 0) < minimum_warmups:
        _fail(f"{label}.warmups: below configured minimum {minimum_warmups}")
    successes = _integer(
        summary.get("successful_requests"), f"{label}.successful_requests", 0
    )
    if successes < minimum_successes:
        _fail(
            f"{label}.successful_requests: below configured minimum {minimum_successes}"
        )
    completed = _integer(
        summary.get("completed_requests"), f"{label}.completed_requests", 0
    )
    if completed != invocation["request"]["requests"]:
        _fail(f"{label}.completed_requests: does not match invocation")
    expected_model = invocation["request"]["sglang_model"]
    if engine == "native":
        if expected_model is not None or summary.get("sglang_model") is not None:
            _fail(f"{label}.sglang_model: Native requires null")
    elif summary.get("sglang_model") != expected_model:
        _fail(f"{label}.sglang_model: does not match invocation")

    request_records = _parse_jsonl(
        run_dir / "client/requests.jsonl",
        run_dir,
        f"{run_dir}/client/requests.jsonl",
    )
    if len(request_records) != completed:
        _fail(f"{run_dir}/client/requests.jsonl: count does not match summary")
    successful_records = 0
    for index, record in enumerate(request_records):
        record_label = f"{run_dir}/client/requests.jsonl:{index + 1}"
        if not isinstance(record, dict):
            _fail(f"{record_label}: expected an object")
        if record.get("schema_version") != CLIENT_SCHEMA_VERSION:
            _fail(f"{record_label}.schema_version: unexpected client schema")
        if record.get("request_index") != index:
            _fail(f"{record_label}.request_index: expected {index}")
        success = record.get("success")
        if not isinstance(success, bool):
            _fail(f"{record_label}.success: expected a boolean")
        if not success:
            continue
        successful_records += 1
        if engine == "native":
            if (
                record.get("finish_reason") != "stop"
                or record.get("natural_eos") is not True
                or record.get("length_limited") is not False
            ):
                _fail(f"{record_label}: Native success is not natural EOS")
        else:
            if any(
                record.get(field) is not None
                for field in ("finish_reason", "natural_eos", "length_limited")
            ):
                _fail(f"{record_label}: SGLang EOS fields must remain unknown")
            samples = _integer(record.get("samples"), f"{record_label}.samples", 1)
            duration = _number(
                record.get("audio_seconds"), f"{record_label}.audio_seconds", 0
            )
            if samples >= SGLANG_EXCLUSIVE_SAMPLE_LIMIT:
                _fail(
                    f"{record_label}.samples: must be below the exclusive "
                    f"{SGLANG_EXCLUSIVE_CODEC_FRAME_LIMIT}-frame boundary"
                )
            if duration >= SGLANG_EXCLUSIVE_DURATION_LIMIT_SECONDS:
                _fail(
                    f"{record_label}.audio_seconds: must be below the exclusive "
                    f"{SGLANG_EXCLUSIVE_DURATION_LIMIT_SECONDS:.2f}-second boundary"
                )
    if successful_records != successes:
        _fail(f"{run_dir}/client/requests.jsonl: success count does not match summary")
    return summary


def _validate_run(
    run_dir: Path,
    evidence_root: Path,
    workload_identity: FileIdentity,
    config: dict[str, Any],
) -> RunBundle:
    identities = _parse_checksum_inventory(run_dir)
    required_files = {
        "provenance/invocation.json",
        "provenance/run-qualifying-benchmark.sh",
        "provenance/capture-spark-telemetry.sh",
        "provenance/lib/process-rss-sampler.sh",
        "provenance/reduce-spark-run.sh",
        "provenance/image-inspect.json",
        "provenance/container-inspect.sanitized.json",
        "provenance/client-version.txt",
        "provenance/uname.txt",
        "provenance/nvidia-smi-list.txt",
        "provenance/nvidia-smi-query.txt",
        "provenance/docker-version.txt",
        "provenance/repository-status.txt",
        "input/qwen3-tts-http-bench",
        "input/workload.jsonl",
        "client/summary.json",
        "client/requests.jsonl",
        "client/packets.jsonl",
        "run-resource.json",
        "resource-audit.json",
        "raw/command.stdout",
        "raw/command.stderr",
        *TELEMETRY_RELATIVE_PATHS,
    }
    missing = required_files - set(identities)
    if missing:
        _fail(f"{run_dir}: missing qualifying-run files: {sorted(missing)}")

    invocation = _validate_invocation_shape(
        _load_json_file(run_dir / "provenance/invocation.json", run_dir, "invocation"),
        f"{run_dir}/provenance/invocation.json",
    )
    resource = _validate_resource_shape(
        _load_json_file(run_dir / "run-resource.json", run_dir, "run-resource"),
        f"{run_dir}/run-resource.json",
    )
    audit = _load_json_file(run_dir / "resource-audit.json", run_dir, "resource-audit")

    if invocation["schema_version"] != RUN_SCHEMA_VERSION:
        _fail(f"{run_dir}: unexpected qualifying-run schema")
    engine = invocation["engine"]
    profile = invocation["profile"]
    round_number = invocation["round"]
    key = (engine, profile, round_number)
    if engine not in ENGINES or profile not in PROFILES:
        _fail(f"{run_dir}: invalid engine/profile identity {key}")
    _integer(round_number, f"{run_dir}.round", 1)
    if key != (
        resource.get("engine_id"),
        resource.get("profile_id"),
        resource.get("round"),
    ):
        _fail(f"{run_dir}: invocation and run-resource identities differ")
    _validate_resource_audit(
        audit,
        f"{run_dir}/resource-audit.json",
        key,
        identities,
        resource,
    )

    evidence_prefix = _path_inside(run_dir, evidence_root, f"{run_dir} prefix")
    if invocation["evidence_prefix"] != evidence_prefix:
        _fail(
            f"{run_dir}: invocation evidence_prefix {invocation['evidence_prefix']!r} "
            f"does not equal {evidence_prefix!r}"
        )
    expected_telemetry = [
        f"{evidence_prefix}/{relative}" for relative in TELEMETRY_RELATIVE_PATHS
    ]
    if resource["telemetry_evidence_paths"] != expected_telemetry:
        _fail(f"{run_dir}: run-resource telemetry paths differ from the canonical set")

    if invocation["client"] != {
        "path": "input/qwen3-tts-http-bench",
        "sha256": identities["input/qwen3-tts-http-bench"].sha256,
    }:
        _fail(f"{run_dir}: invocation client identity differs from captured client")
    if invocation["workload"] != {
        "path": "input/workload.jsonl",
        "sha256": workload_identity.sha256,
    }:
        _fail(f"{run_dir}: invocation workload identity differs from central workload")
    if identities["input/workload.jsonl"].sha256 != workload_identity.sha256:
        _fail(f"{run_dir}: captured workload differs from central workload")

    implementation = next(
        (
            item
            for item in config["implementations"]
            if item.get("id") == engine and item.get("role") == engine
        ),
        None,
    )
    if implementation is None:
        _fail(f"config.implementations: no unique declaration for {engine}")
    local_image = implementation["local_image"]
    image_id = invocation["image"].get("resolved_id")
    if image_id != local_image["id"]:
        _fail(
            f"{run_dir}: resolved local Docker image ID differs from configured implementation"
        )
    if not IMAGE_DIGEST_RE.fullmatch(str(image_id)):
        _fail(f"{run_dir}: invalid resolved local Docker image ID")
    if invocation["image"].get("reference") != local_image["reference"]:
        _fail(f"{run_dir}: tested local image reference differs from configuration")
    image_inspect = _load_json_file(
        run_dir / "provenance/image-inspect.json", run_dir, "image-inspect"
    )
    if not isinstance(image_inspect, list) or len(image_inspect) != 1:
        _fail(f"{run_dir}: image-inspect evidence must contain exactly one image")
    inspected = image_inspect[0]
    if not isinstance(inspected, dict):
        _fail(f"{run_dir}: image-inspect entry must be an object")
    if inspected.get("Id") != image_id:
        _fail(f"{run_dir}: image-inspect Id differs from invocation resolved_id")
    inspected_size = _integer(inspected.get("Size"), f"{run_dir}.image-inspect.Size", 1)
    if inspected_size != local_image["unpacked_size_bytes"]:
        _fail(
            f"{run_dir}: image-inspect Size differs from configured local unpacked size"
        )

    request = invocation["request"]
    _integer(request["requests"], f"{run_dir}.request.requests", 200)
    _integer(request["warmups"], f"{run_dir}.request.warmups", 24)
    if engine == "sglang":
        if request["sglang_model"] != config["model"]["repository"]:
            _fail(f"{run_dir}: SGLang model differs from configured model repository")
    elif request["sglang_model"] is not None:
        _fail(f"{run_dir}: Native sglang_model must be null")
    interval = _integer(
        invocation["telemetry"]["configured_sample_interval_ms"],
        f"{run_dir}.telemetry.configured_sample_interval_ms",
        1,
    )
    if interval != resource["sampling_interval_ms"]:
        _fail(f"{run_dir}: invocation and run-resource sampling intervals differ")
    if interval != config["methodology"]["sampling_interval_ms"]:
        _fail(f"{run_dir}: run sampling interval differs from methodology")
    if invocation["telemetry"]["maximum_qualifying_observed_gap_ms"] != 200:
        _fail(f"{run_dir}: maximum qualifying telemetry gap must be 200 ms")
    tooling = invocation["tooling_repository"]
    if tooling["tracked_files_clean"] is not True:
        _fail(f"{run_dir}: benchmark tooling worktree was not clean")
    if not isinstance(tooling["commit"], str) or not COMMIT_RE.fullmatch(
        tooling["commit"]
    ):
        _fail(f"{run_dir}: benchmark tooling commit is missing or not immutable")

    summary = _validate_summary_and_requests(
        run_dir,
        engine,
        profile,
        invocation,
        config["workload"]["minimum_measured_requests_per_profile"],
        config["workload"]["warmup_requests_per_run"],
    )
    return RunBundle(
        key=key,
        directory=run_dir,
        evidence_prefix=evidence_prefix,
        identities=identities,
        invocation=invocation,
        summary=summary,
        resource=resource,
    )


def _descriptor(
    role: str,
    evidence_path: str,
    format_name: str,
    identity: FileIdentity,
    run: RunBundle | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "path": evidence_path,
        "format": format_name,
        "sha256": identity.sha256,
        "bytes": identity.bytes,
    }
    if run is not None:
        result.update(
            {
                "engine_id": run.key[0],
                "profile_id": run.key[1],
                "round": run.key[2],
            }
        )
    return result


def _raw_format(relative: str) -> str | None:
    name = PurePosixPath(relative).name
    if "." not in name:
        return None
    suffix = name.rsplit(".", 1)[1]
    return suffix if suffix in RAW_FORMATS else None


def _assemble_descriptors(
    workload_path: str,
    workload_identity: FileIdentity,
    bound_evidence: list[BoundEvidence],
    runs: list[RunBundle],
) -> list[dict[str, Any]]:
    descriptors = [_descriptor("workload", workload_path, "jsonl", workload_identity)]
    seen_paths = {workload_path}
    for item in bound_evidence:
        if item.path in seen_paths:
            _fail(f"duplicate evidence path: {item.path}")
        seen_paths.add(item.path)
        descriptor = _descriptor(item.role, item.path, "json", item.identity)
        descriptor["engine_id"] = item.engine
        descriptors.append(descriptor)
    for run in runs:
        for relative, (role, format_name) in CLIENT_FILES.items():
            evidence_path = f"{run.evidence_prefix}/{relative}"
            if evidence_path in seen_paths:
                _fail(f"duplicate evidence path: {evidence_path}")
            seen_paths.add(evidence_path)
            descriptors.append(
                _descriptor(
                    role,
                    evidence_path,
                    format_name,
                    run.identities[relative],
                    run,
                )
            )
        for relative in sorted(run.identities):
            if relative in CLIENT_FILES:
                continue
            identity = run.identities[relative]
            format_name = _raw_format(relative)
            if format_name is None or identity.bytes == 0:
                continue
            evidence_path = f"{run.evidence_prefix}/{relative}"
            if evidence_path in seen_paths:
                _fail(f"duplicate evidence path: {evidence_path}")
            seen_paths.add(evidence_path)
            descriptors.append(
                _descriptor("raw", evidence_path, format_name, identity, run)
            )
    return descriptors


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise SchemaViolation(
        f"unsupported JSON Schema type in repository schema: {expected}"
    )


def _resolve_schema_reference(root: dict[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise SchemaViolation(f"unsupported non-local schema reference: {reference}")
    current: Any = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        current = current[part]
    return current


def _validate_datetime(value: str, label: str) -> None:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise SchemaViolation(f"{label}: expected an RFC 3339 date-time") from exc
    if parsed.tzinfo is None:
        raise SchemaViolation(f"{label}: date-time must include a UTC offset")


def _schema_matches(value: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    try:
        _validate_schema(value, schema, root, "$if")
    except SchemaViolation:
        return False
    return True


def _validate_schema(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    label: str,
) -> None:
    if "$ref" in schema:
        _validate_schema(
            value,
            _resolve_schema_reference(root, schema["$ref"]),
            root,
            label,
        )
    for subschema in schema.get("allOf", []):
        _validate_schema(value, subschema, root, label)
    if "oneOf" in schema:
        matches = sum(
            _schema_matches(value, subschema, root) for subschema in schema["oneOf"]
        )
        if matches != 1:
            raise SchemaViolation(
                f"{label}: expected exactly one oneOf branch, got {matches}"
            )
    if "not" in schema and _schema_matches(value, schema["not"], root):
        raise SchemaViolation(f"{label}: matched a forbidden schema")
    if "if" in schema:
        branch = "then" if _schema_matches(value, schema["if"], root) else "else"
        if branch in schema:
            _validate_schema(value, schema[branch], root, label)
    if "const" in schema and value != schema["const"]:
        raise SchemaViolation(f"{label}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaViolation(f"{label}: value is outside the allowed enum")
    if "type" in schema:
        expected_types = schema["type"]
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_schema_type_matches(value, item) for item in expected_types):
            raise SchemaViolation(f"{label}: expected type {schema['type']!r}")

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = set(required) - set(value)
        if missing:
            raise SchemaViolation(f"{label}: missing required keys {sorted(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise SchemaViolation(f"{label}: unexpected keys {sorted(extra)}")
        for key, subschema in properties.items():
            if key in value:
                _validate_schema(value[key], subschema, root, f"{label}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise SchemaViolation(f"{label}: too few array items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaViolation(f"{label}: too many array items")
        if schema.get("uniqueItems"):
            canonical = [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in value
            ]
            if len(canonical) != len(set(canonical)):
                raise SchemaViolation(f"{label}: duplicate array items")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], root, f"{label}[{index}]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise SchemaViolation(f"{label}: string is too short")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise SchemaViolation(f"{label}: string does not match required pattern")
        if schema.get("format") == "date-time":
            _validate_datetime(value, label)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise SchemaViolation(f"{label}: non-finite number")
        if "minimum" in schema and numeric < schema["minimum"]:
            raise SchemaViolation(f"{label}: number is below minimum")
        if "maximum" in schema and numeric > schema["maximum"]:
            raise SchemaViolation(f"{label}: number is above maximum")
        if "exclusiveMinimum" in schema and numeric <= schema["exclusiveMinimum"]:
            raise SchemaViolation(f"{label}: number is below exclusive minimum")


def _validate_static_config(
    config: dict[str, Any],
    workload: FileIdentity,
    ordered_seeds: tuple[int, ...],
) -> None:
    workload_config = _strict_object(
        config["workload"], "config.workload", WORKLOAD_CONFIG_KEYS
    )
    if workload_config["corpus_sha256"] != workload.sha256:
        _fail(
            "config.workload.corpus_sha256: configured digest does not match "
            "the central workload"
        )
    if workload_config["ordered_seeds"] != list(ordered_seeds):
        _fail(
            "config.workload.ordered_seeds: must exactly match the ordered seed "
            "from every digest-verified workload row"
        )
    if workload_config["sample_rate_hz"] != SAMPLE_RATE_HZ:
        _fail(f"config.workload.sample_rate_hz: production requires {SAMPLE_RATE_HZ}")
    if workload_config["channels"] != 1:
        _fail("config.workload.channels: production requires mono audio")
    if workload_config["sample_format"] != "pcm_s16le":
        _fail("config.workload.sample_format: production requires pcm_s16le")
    if workload_config["response_mode"] != "streaming":
        _fail("config.workload.response_mode: production requires streaming")
    if (
        _integer(
            workload_config["warmup_requests_per_run"],
            "config.workload.warmup_requests_per_run",
            24,
        )
        < 24
    ):
        _fail("config.workload.warmup_requests_per_run: production requires 24")
    if (
        _integer(
            workload_config["minimum_measured_requests_per_profile"],
            "config.workload.minimum_measured_requests_per_profile",
            200,
        )
        < 200
    ):
        _fail("config.workload.minimum_measured_requests_per_profile: requires 200")
    if workload_config["minimum_rounds_per_subject"] != 2:
        _fail("config.workload.minimum_rounds_per_subject: this assembler requires 2")
    profiles = workload_config["profiles"]
    if not isinstance(profiles, list):
        _fail("config.workload.profiles: expected an array")
    observed_profiles: dict[str, int] = {}
    for index, raw in enumerate(profiles):
        profile = _strict_object(
            raw,
            f"config.workload.profiles[{index}]",
            {"id", "concurrency", "repetitions_per_request"},
        )
        _integer(
            profile["repetitions_per_request"],
            f"config.workload.profiles[{index}].repetitions_per_request",
            1,
        )
        observed_profiles[profile["id"]] = profile["concurrency"]
    if observed_profiles != PROFILES or len(profiles) != len(PROFILES):
        _fail("config.workload.profiles: expected exactly B1, B3, and B6")

    implementations = config["implementations"]
    if not isinstance(implementations, list) or len(implementations) != 2:
        _fail("config.implementations: expected exactly two entries")
    roles: set[str] = set()
    for index, implementation in enumerate(implementations):
        if not isinstance(implementation, dict):
            _fail(f"config.implementations[{index}]: expected an object")
        role = implementation.get("role")
        if role not in ENGINES or implementation.get("id") != role or role in roles:
            _fail(f"config.implementations[{index}]: invalid or duplicate role")
        roles.add(role)
        local_image = implementation.get("local_image")
        if not isinstance(local_image, dict):
            _fail(f"config.implementations[{index}].local_image: expected an object")
        if not IMAGE_DIGEST_RE.fullmatch(str(local_image.get("id"))):
            _fail(f"config.implementations[{index}].local_image.id: invalid ID")
        _string(
            local_image.get("reference"),
            f"config.implementations[{index}].local_image.reference",
        )
        _integer(
            local_image.get("unpacked_size_bytes"),
            f"config.implementations[{index}].local_image.unpacked_size_bytes",
            1,
        )
        artifact = implementation.get("model_artifact")
        if not isinstance(artifact, dict):
            _fail(
                f"config.implementations[{index}].model_artifact: required "
                "digest-bound artifact metadata is unavailable"
            )
        for field in ("repository", "revision", "variant"):
            if artifact.get(field) != config["model"].get(field):
                _fail(
                    f"config.implementations[{index}].model_artifact.{field}: "
                    f"must equal config.model.{field}"
                )


def _load_evidence_schema() -> dict[str, Any]:
    schema_path = Path(__file__).resolve().parents[1] / "evidence.schema.json"
    schema = _parse_json_bytes(schema_path.read_bytes(), str(schema_path))
    if not isinstance(schema, dict):
        _fail(f"{schema_path}: expected a JSON object")
    return schema


def _validate_config_against_schema(
    config: dict[str, Any], schema: dict[str, Any]
) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        _fail("evidence schema: missing root properties")
    for key in sorted(CONFIG_KEYS):
        field_schema = properties.get(key)
        if not isinstance(field_schema, dict):
            _fail(f"evidence schema: missing config field schema for {key}")
        _validate_schema(config[key], field_schema, schema, f"config.{key}")


def _cross_validate_runs(runs: list[RunBundle]) -> None:
    observed_keys = {run.key for run in runs}
    if len(observed_keys) != len(runs):
        duplicates = sorted(
            key for key in observed_keys if sum(r.key == key for r in runs) > 1
        )
        _fail(f"duplicate qualifying-run identities: {duplicates}")
    if observed_keys != EXPECTED_RUN_KEYS:
        _fail(
            "qualifying-run set mismatch; "
            f"missing={sorted(EXPECTED_RUN_KEYS - observed_keys)} "
            f"extra={sorted(observed_keys - EXPECTED_RUN_KEYS)}"
        )
    client_digests = {
        run.identities["input/qwen3-tts-http-bench"].sha256 for run in runs
    }
    if len(client_digests) != 1:
        _fail("qualifying runs used different benchmark-client binaries")
    tooling_commits = {run.invocation["tooling_repository"]["commit"] for run in runs}
    if len(tooling_commits) != 1:
        _fail("qualifying runs used different benchmark-tooling commits")
    for engine in ENGINES:
        engine_runs = [run for run in runs if run.key[0] == engine]
        image_ids = {run.invocation["image"]["resolved_id"] for run in engine_runs}
        image_references = {run.invocation["image"]["reference"] for run in engine_runs}
        if len(image_ids) != 1 or len(image_references) != 1:
            _fail(f"{engine}: qualifying runs used different container images")


def assemble_manifest(
    config_path: Path,
    workload_path: Path,
    runs_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate the bundle, create the manifest once, and return its object."""

    output_path = output_path.expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        _fail(f"output already exists; refusing to overwrite: {output_path}")
    output_parent = output_path.parent
    _reject_symlink(output_parent, "output parent")
    if not output_parent.is_dir():
        _fail(f"output parent does not exist: {output_parent}")
    evidence_root = output_parent.resolve(strict=True)
    output_path = evidence_root / output_path.name

    workload_path = workload_path.expanduser().absolute()
    runs_root = runs_root.expanduser().absolute()
    config_path = config_path.expanduser().absolute()
    _reject_symlink(workload_path, "workload")
    _reject_symlink(runs_root, "runs root")
    _reject_symlink(config_path, "config")
    workload_path = workload_path.resolve(strict=True)
    runs_root = runs_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    _path_inside(workload_path, evidence_root, "workload")
    _path_inside(runs_root, evidence_root, "runs root")
    if not workload_path.is_file():
        _fail(f"workload is not a regular file: {workload_path}")
    if not runs_root.is_dir():
        _fail(f"runs root is not a directory: {runs_root}")
    _validate_tree(runs_root, "runs root")

    config = _load_config(config_path)
    schema = _load_evidence_schema()
    _validate_config_against_schema(config, schema)
    workload_identity, ordered_seeds = _validate_workload(workload_path, evidence_root)
    _validate_static_config(config, workload_identity, ordered_seeds)
    bound_evidence = _load_bound_implementation_evidence(config, evidence_root)

    candidates = _discover_run_directories(runs_root)
    if len(candidates) != 12:
        _fail(
            f"expected exactly 12 qualifying-run directories, observed {len(candidates)}"
        )
    _validate_run_tree_ownership(runs_root, candidates)
    runs = [
        _validate_run(candidate, evidence_root, workload_identity, config)
        for candidate in candidates
    ]
    _cross_validate_runs(runs)
    runs.sort(key=lambda run: (run.key[2], run.key[0], PROFILES[run.key[1]]))

    workload_relative = _path_inside(workload_path, evidence_root, "workload")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_kind": EVIDENCE_KIND,
        "report": config["report"],
        "system": config["system"],
        "model": config["model"],
        "workload": config["workload"],
        "implementations": config["implementations"],
        "methodology": config["methodology"],
        "evidence_files": _assemble_descriptors(
            workload_relative, workload_identity, bound_evidence, runs
        ),
        "run_resources": [run.resource for run in runs],
        "limitations": config["limitations"],
    }

    _validate_schema(manifest, schema, schema, "manifest")

    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output_path, flags, 0o444)
    except FileExistsError as exc:
        raise AssemblyError(f"output already exists: {output_path}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            output_path.unlink()
        except OSError:
            pass
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble one fail-closed schema-v1.2 production manifest."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Explicit immutable production metadata JSON",
    )
    parser.add_argument(
        "--workload",
        required=True,
        type=Path,
        help="Canonical workload JSONL inside the manifest directory",
    )
    parser.add_argument(
        "--runs-root",
        required=True,
        type=Path,
        help="Directory containing exactly twelve qualifying runs",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Create-new manifest.json path; its parent is the evidence root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = assemble_manifest(
            args.config,
            args.workload,
            args.runs_root,
            args.output,
        )
    except (AssemblyError, OSError) as exc:
        print(f"assembly failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"created {args.output}: {len(manifest['run_resources'])} runs, "
        f"{len(manifest['evidence_files'])} evidence descriptors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
