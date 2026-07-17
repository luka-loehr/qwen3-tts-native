#!/usr/bin/env python3
"""Finalize manifest, report, and paper data from one production evidence root."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn


class PipelineError(ValueError):
    """Raised when the production evidence pipeline cannot finish safely."""


@dataclass(frozen=True)
class PublishedOutputs:
    metadata: Path
    manifest: Path
    report: Path
    paper_files: tuple[Path, ...]


def _fail(message: str) -> NoReturn:
    raise PipelineError(message)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load required pipeline module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _modules(repository_root: Path) -> tuple[Any, Any, Any, Any]:
    tools = repository_root / "reports" / "tools"
    metadata = _load_module(
        "qwen3_tts_pipeline_metadata", tools / "prepare_production_metadata.py"
    )
    assembler = _load_module(
        "qwen3_tts_pipeline_assembler", tools / "assemble_production_manifest.py"
    )
    report = _load_module(
        "qwen3_tts_pipeline_report", repository_root / "reports" / "generate_report.py"
    )
    paper = _load_module(
        "qwen3_tts_pipeline_paper",
        repository_root / "research" / "paper" / "tools" / "finalize_evidence.py",
    )
    return metadata, assembler, report, paper


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_new(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"{label} already exists; refusing to overwrite: {path}")
    if not path.parent.is_dir():
        _fail(f"{label} parent does not exist: {path.parent}")


def _paper_preflight(paper_root: Path, paper: Any) -> tuple[Path, bytes]:
    paper_root = paper_root.expanduser().resolve(strict=True)
    if (
        not (paper_root / "main.tex").is_file()
        or not (paper_root / "Makefile").is_file()
    ):
        _fail(f"not a paper source root: {paper_root}")
    data_dir = paper_root / "data"
    placeholder = data_dir / "evidence_placeholders.tex"
    if data_dir.is_symlink() or not data_dir.is_dir():
        _fail(f"paper data directory is missing or unsafe: {data_dir}")
    if placeholder.is_symlink() or not placeholder.is_file():
        _fail(f"managed paper evidence boundary is missing or unsafe: {placeholder}")
    original = placeholder.read_bytes()
    if (
        b"\\FinalEvidenceAvailablefalse" not in original
        or b"PENDING_EVIDENCE" not in original
    ):
        _fail("paper evidence boundary is not in its pristine pending state")
    for name in paper.MANAGED_FILENAMES:
        path = data_dir / name
        if name != "evidence_placeholders.tex" and (path.exists() or path.is_symlink()):
            _fail(f"managed paper output already exists; refusing to overwrite: {path}")
    return data_dir, original


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def finalize(
    declarations_path: Path,
    evidence_root: Path,
    paper_root: Path,
) -> PublishedOutputs:
    """Validate and publish the complete production evidence output set once."""

    repository_root = Path(__file__).resolve().parents[2]
    metadata_module, assembler, report, paper = _modules(repository_root)
    evidence_root = evidence_root.expanduser().resolve(strict=True)
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        _fail(f"evidence root is missing or unsafe: {evidence_root}")

    metadata_value = metadata_module.build_metadata(declarations_path, evidence_root)
    benchmark_id = metadata_value["report"]["benchmark_id"]
    metadata_output = evidence_root / "production-metadata.json"
    manifest_output = evidence_root / "manifest.json"
    report_output = evidence_root / f"{benchmark_id}-report.pdf"
    for path, label in (
        (metadata_output, "production metadata"),
        (manifest_output, "production manifest"),
        (report_output, "benchmark report"),
    ):
        _require_new(path, label)
    paper_data_dir, original_paper_boundary = _paper_preflight(paper_root, paper)

    process_token = f"{os.getpid()}"
    metadata_stage = evidence_root / f".production-metadata.{process_token}.stage.json"
    manifest_stage = evidence_root / f".manifest.{process_token}.stage.json"
    report_stage = evidence_root / f".{benchmark_id}.{process_token}.stage.pdf"
    stages = (metadata_stage, manifest_stage, report_stage)
    for stage in stages:
        _require_new(stage, "pipeline staging file")

    committed: list[Path] = []
    paper_commit_started = False
    try:
        metadata_module.write_create_new(
            metadata_stage, metadata_module.encode_metadata(metadata_value)
        )
        assembler.assemble_manifest(
            metadata_stage,
            evidence_root / "workload" / "workload.jsonl",
            evidence_root / "runs",
            manifest_stage,
        )
        try:
            bundle = report.load_bundle(manifest_stage, allow_test_fixture=False)
            report.aggregate(bundle)
            report.build_pdf(bundle, report_stage, overwrite=False)
        except report.EvidenceError as exc:
            raise PipelineError(f"production report validation failed: {exc}") from exc
        try:
            paper_bundle, aggregates = paper._validated_bundle(manifest_stage)
            paper_outputs = paper.build_outputs(paper_bundle, aggregates)
        except paper.FinalizationError as exc:
            raise PipelineError(f"paper evidence finalization failed: {exc}") from exc

        for stage, destination in (
            (metadata_stage, metadata_output),
            (manifest_stage, manifest_output),
            (report_stage, report_output),
        ):
            _require_new(destination, "pipeline output")
            os.replace(stage, destination)
            committed.append(destination)
        paper_commit_started = True
        paper.write_outputs(paper_outputs, paper_root)
    except BaseException:
        for stage in stages:
            _safe_unlink(stage)
        if paper_commit_started:
            try:
                paper._atomic_replace(
                    paper_data_dir / "evidence_placeholders.tex",
                    original_paper_boundary,
                )
            finally:
                for name in ("native-runs.dat", "sglang-runs.dat"):
                    _safe_unlink(paper_data_dir / name)
        for path in reversed(committed):
            _safe_unlink(path)
        raise

    paper_files = tuple(paper_data_dir / name for name in paper.MANAGED_FILENAMES)
    return PublishedOutputs(
        metadata=metadata_output,
        manifest=manifest_output,
        report=report_output,
        paper_files=paper_files,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed one-shot production evidence finalization: metadata, "
            "manifest, PDF report, and paper data."
        )
    )
    parser.add_argument("--declarations", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "research" / "paper",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        outputs = finalize(args.declarations, args.evidence_root, args.paper_root)
    except (PipelineError, OSError, ValueError) as exc:
        print(f"production evidence finalization failed: {exc}", file=sys.stderr)
        return 1
    print(f"production metadata: {outputs.metadata}")
    print(
        f"production manifest: {outputs.manifest} (sha256:{_sha256(outputs.manifest)})"
    )
    print(f"benchmark report: {outputs.report} (sha256:{_sha256(outputs.report)})")
    for path in outputs.paper_files:
        print(f"paper evidence: {path} (sha256:{_sha256(path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
