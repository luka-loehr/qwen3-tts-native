#!/usr/bin/env python3
"""Prepare deterministic production metadata from declarations and evidence.

Static claims remain explicit in a declarations file.  Identity-bearing fields
that can be observed from the qualifying bundle are copied from checksum-bound
run provenance and artifact manifests.  Nothing is inferred from filenames,
rounded report output, or exploratory benchmark state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn


DECLARATION_KEYS = {
    "report",
    "system",
    "workload",
    "implementations",
    "methodology",
    "limitations",
}
WORKLOAD_DECLARATION_KEYS = {"profiles", "language_policy"}
IMPLEMENTATION_DECLARATION_KEYS = {
    "id",
    "name",
    "version",
    "source_commit",
    "source_url",
    "api_protocol",
    "streaming_semantics",
    "runtime_components",
}
IMPLEMENTATION_CLAIM_ORDER = (
    "name",
    "version",
    "source_commit",
    "source_url",
    "api_protocol",
    "streaming_semantics",
    "runtime_components",
)
PLACEHOLDER_TOKENS = ("PENDING_", "REPLACE_ME", "<REPLACE")


class MetadataError(ValueError):
    """Raised when production metadata cannot be prepared without guessing."""


def _fail(message: str) -> NoReturn:
    raise MetadataError(message)


def _load_assembler() -> Any:
    module_path = Path(__file__).with_name("assemble_production_manifest.py")
    spec = importlib.util.spec_from_file_location(
        "qwen3_tts_production_metadata_assembler", module_path
    )
    if spec is None or spec.loader is None:
        _fail(f"cannot load production manifest assembler: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _strict_object(
    value: Any,
    label: str,
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label}: expected an object")
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        _fail(f"{label}: key mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    return value


def _reject_placeholders(value: Any, label: str = "declarations") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{label}[{index}]")
    elif isinstance(value, str) and any(token in value for token in PLACEHOLDER_TOKENS):
        _fail(f"{label}: unresolved placeholder is forbidden")


def _load_declarations(path: Path, assembler: Any) -> dict[str, Any]:
    path = path.expanduser().absolute()
    assembler._reject_symlink(path, "declarations")
    if not path.is_file():
        _fail(f"declarations: expected a regular file: {path}")
    try:
        value = assembler._parse_json_bytes(path.read_bytes(), str(path))
    except assembler.AssemblyError as exc:
        raise MetadataError(str(exc)) from exc
    declarations = _strict_object(value, "declarations", DECLARATION_KEYS)
    _strict_object(
        declarations["workload"],
        "declarations.workload",
        WORKLOAD_DECLARATION_KEYS,
    )
    implementations = declarations["implementations"]
    if not isinstance(implementations, list) or len(implementations) != 2:
        _fail("declarations.implementations: expected exactly two entries")
    engines: set[str] = set()
    for index, value in enumerate(implementations):
        item = _strict_object(
            value,
            f"declarations.implementations[{index}]",
            IMPLEMENTATION_DECLARATION_KEYS,
        )
        engine = item["id"]
        if engine not in assembler.ENGINES or engine in engines:
            _fail(
                f"declarations.implementations[{index}].id: "
                "expected unique native and sglang entries"
            )
        engines.add(engine)
        if not isinstance(
            item["source_commit"], str
        ) or not assembler.COMMIT_RE.fullmatch(item["source_commit"]):
            _fail(
                f"declarations.implementations[{index}].source_commit: "
                "expected an immutable 40-character lowercase Git commit"
            )
    _reject_placeholders(declarations)
    return declarations


def _load_run_images(
    evidence_root: Path,
    assembler: Any,
) -> dict[str, dict[str, Any]]:
    runs_root = evidence_root / "runs"
    assembler._validate_tree(runs_root, "runs root")
    run_dirs = assembler._discover_run_directories(runs_root)
    if len(run_dirs) != 12:
        _fail(
            f"expected exactly 12 qualifying-run directories, observed {len(run_dirs)}"
        )
    assembler._validate_run_tree_ownership(runs_root, run_dirs)

    observed_keys: set[tuple[str, str, int]] = set()
    images: dict[str, set[tuple[str, str, int]]] = {
        engine: set() for engine in assembler.ENGINES
    }
    for run_dir in run_dirs:
        identities = assembler._parse_checksum_inventory(run_dir)
        required = {
            "provenance/invocation.json",
            "provenance/image-inspect.json",
            "run-resource.json",
        }
        missing = required - set(identities)
        if missing:
            _fail(f"{run_dir}: missing metadata inputs: {sorted(missing)}")
        invocation = assembler._validate_invocation_shape(
            assembler._load_json_file(
                run_dir / "provenance/invocation.json", run_dir, "invocation"
            ),
            f"{run_dir}/provenance/invocation.json",
        )
        engine = invocation["engine"]
        profile = invocation["profile"]
        round_number = invocation["round"]
        key = (engine, profile, round_number)
        if engine not in assembler.ENGINES or profile not in assembler.PROFILES:
            _fail(f"{run_dir}: invalid qualifying-run identity {key}")
        if key in observed_keys:
            _fail(f"duplicate qualifying-run identity: {key}")
        observed_keys.add(key)
        if invocation["schema_version"] != assembler.RUN_SCHEMA_VERSION:
            _fail(f"{run_dir}: unexpected qualifying-run schema")
        if invocation["tooling_repository"].get("tracked_files_clean") is not True:
            _fail(f"{run_dir}: benchmark tooling worktree was not clean")
        tooling_commit = invocation["tooling_repository"].get("commit")
        if not isinstance(tooling_commit, str) or not assembler.COMMIT_RE.fullmatch(
            tooling_commit
        ):
            _fail(f"{run_dir}: benchmark tooling commit is not immutable")

        image = invocation["image"]
        reference = image.get("reference")
        image_id = image.get("resolved_id")
        if not isinstance(reference, str) or not reference:
            _fail(f"{run_dir}: missing local image reference")
        if not isinstance(image_id, str) or not assembler.IMAGE_DIGEST_RE.fullmatch(
            image_id
        ):
            _fail(f"{run_dir}: invalid local Docker image ID")
        inspected = assembler._load_json_file(
            run_dir / "provenance/image-inspect.json", run_dir, "image-inspect"
        )
        if not isinstance(inspected, list) or len(inspected) != 1:
            _fail(f"{run_dir}: image-inspect evidence must contain exactly one image")
        image_record = inspected[0]
        if not isinstance(image_record, dict) or image_record.get("Id") != image_id:
            _fail(f"{run_dir}: image-inspect Id differs from invocation resolved_id")
        size = image_record.get("Size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            _fail(f"{run_dir}: invalid image-inspect Size")
        images[engine].add((reference, image_id, size))

    if observed_keys != assembler.EXPECTED_RUN_KEYS:
        _fail(
            "qualifying-run set mismatch; "
            f"missing={sorted(assembler.EXPECTED_RUN_KEYS - observed_keys)} "
            f"extra={sorted(observed_keys - assembler.EXPECTED_RUN_KEYS)}"
        )
    result: dict[str, dict[str, Any]] = {}
    for engine, candidates in images.items():
        if len(candidates) != 1:
            _fail(f"{engine}: qualifying runs used inconsistent local images")
        reference, image_id, size = next(iter(candidates))
        result[engine] = {
            "reference": reference,
            "id": image_id,
            "unpacked_size_bytes": size,
        }
    return result


def _artifact_configuration(
    evidence_root: Path,
    engine: str,
    image_id: str,
    assembler: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = evidence_root / "artifacts" / engine / "model-artifact.json"
    identity = assembler._hash_regular(path, evidence_root)
    payload = assembler._load_json_file(path, evidence_root, f"{engine} model artifact")
    required = {
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
    }
    payload = _strict_object(payload, f"{engine} model artifact", required)
    if payload["schema_version"] != assembler.MODEL_ARTIFACT_SCHEMA_VERSION:
        _fail(f"{engine} model artifact: unexpected schema")
    if payload["implementation_id"] != engine:
        _fail(f"{engine} model artifact: implementation identity mismatch")
    if payload["local_image_id"] != image_id:
        _fail(f"{engine} model artifact: local image identity mismatch")
    claims = {
        field: payload[field]
        for field in (
            "repository",
            "revision",
            "variant",
            "parameter_count",
            "precision",
            "manifest_sha256",
            "weight_files",
        )
    }
    claims["evidence"] = {
        "path": path.relative_to(evidence_root).as_posix(),
        "sha256": identity.sha256,
    }
    return claims, payload


def _registry_configuration(
    evidence_root: Path,
    engine: str,
    image_id: str,
    assembler: Any,
) -> dict[str, Any] | None:
    path = evidence_root / "artifacts" / engine / "registry-image.json"
    if not path.exists() and not path.is_symlink():
        return None
    identity = assembler._hash_regular(path, evidence_root)
    payload = assembler._load_json_file(path, evidence_root, f"{engine} registry image")
    required = {
        "schema_version",
        "implementation_id",
        "local_image_id",
        "reference",
        "manifest_digest",
    }
    optional = {"compressed_size_bytes"}
    if not isinstance(payload, dict):
        _fail(f"{engine} registry image: expected an object")
    missing = required - set(payload)
    extra = set(payload) - required - optional
    if missing or extra:
        _fail(
            f"{engine} registry image: key mismatch; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    if payload["schema_version"] != assembler.REGISTRY_METADATA_SCHEMA_VERSION:
        _fail(f"{engine} registry image: unexpected schema")
    if payload["implementation_id"] != engine or payload["local_image_id"] != image_id:
        _fail(f"{engine} registry image: tested local image identity mismatch")
    result = {
        field: payload[field]
        for field in ("reference", "manifest_digest", "compressed_size_bytes")
        if field in payload
    }
    result["evidence"] = {
        "path": path.relative_to(evidence_root).as_posix(),
        "sha256": identity.sha256,
    }
    return result


def build_metadata(declarations_path: Path, evidence_root: Path) -> dict[str, Any]:
    """Build fully expanded schema-v1.2 metadata without writing a file."""

    assembler = _load_assembler()
    declarations = _load_declarations(declarations_path, assembler)
    evidence_root = evidence_root.expanduser().absolute()
    assembler._reject_symlink(evidence_root, "evidence root")
    evidence_root = evidence_root.resolve(strict=True)
    if not evidence_root.is_dir():
        _fail(f"evidence root: expected a directory: {evidence_root}")

    workload_path = evidence_root / "workload" / "workload.jsonl"
    workload_identity, ordered_seeds = assembler._validate_workload(
        workload_path, evidence_root
    )
    local_images = _load_run_images(evidence_root, assembler)

    declared_implementations = {
        item["id"]: item for item in declarations["implementations"]
    }
    implementations: list[dict[str, Any]] = []
    common_models: list[dict[str, Any]] = []
    for engine in assembler.ENGINES:
        local_image = local_images[engine]
        artifact, _ = _artifact_configuration(
            evidence_root, engine, local_image["id"], assembler
        )
        common_models.append(
            {field: artifact[field] for field in ("repository", "revision", "variant")}
        )
        declared = declared_implementations[engine]
        implementation = {
            "id": engine,
            "role": engine,
            **{key: declared[key] for key in IMPLEMENTATION_CLAIM_ORDER},
            "local_image": local_image,
            "model_artifact": artifact,
        }
        registry = _registry_configuration(
            evidence_root, engine, local_image["id"], assembler
        )
        if registry is not None:
            implementation["registry_image"] = registry
        implementations.append(implementation)
    if common_models[0] != common_models[1]:
        _fail("Native and stock artifact evidence do not identify the same model")

    workload_declaration = declarations["workload"]
    metadata = {
        "report": declarations["report"],
        "system": declarations["system"],
        "model": common_models[0],
        "workload": {
            "corpus_sha256": workload_identity.sha256,
            "ordered_seeds": list(ordered_seeds),
            "sample_rate_hz": assembler.SAMPLE_RATE_HZ,
            "channels": 1,
            "sample_format": "pcm_s16le",
            "response_mode": "streaming",
            "warmup_requests_per_run": 24,
            "minimum_measured_requests_per_profile": 200,
            "minimum_rounds_per_subject": 2,
            "profiles": workload_declaration["profiles"],
            "language_policy": workload_declaration["language_policy"],
        },
        "implementations": implementations,
        "methodology": declarations["methodology"],
        "limitations": declarations["limitations"],
    }
    try:
        schema = assembler._load_evidence_schema()
        assembler._validate_config_against_schema(metadata, schema)
        assembler._validate_static_config(metadata, workload_identity, ordered_seeds)
        assembler._load_bound_implementation_evidence(metadata, evidence_root)
    except assembler.AssemblyError as exc:
        raise MetadataError(str(exc)) from exc
    return metadata


def encode_metadata(metadata: dict[str, Any]) -> bytes:
    return (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_create_new(path: Path, payload: bytes) -> None:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        _fail(f"output already exists; refusing to overwrite: {path}")
    if not path.parent.is_dir():
        _fail(f"output parent does not exist: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare create-new production metadata from explicit declarations "
            "and checksum-bound qualifying evidence."
        )
    )
    parser.add_argument("--declarations", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        metadata = build_metadata(args.declarations, args.evidence_root)
        write_create_new(args.output, encode_metadata(metadata))
    except (MetadataError, OSError) as exc:
        print(f"metadata preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"created {args.output}: deterministic schema-v1.2 production metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
