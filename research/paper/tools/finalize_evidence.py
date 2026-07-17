#!/usr/bin/env python3
"""Finalize the paper's machine-generated evidence boundary from schema v1.2.

The finalizer never accepts hand-entered measurements.  It revalidates the
complete production bundle with the report pipeline, derives every emitted
value from that bundle, and atomically replaces only the three documented
machine-generated paper data files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn


SCHEMA_VERSION = "1.2"
EXPECTED_MODEL_REPOSITORY = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
EXPECTED_MODEL_REVISION = "5ecdb67327fd37bb2e042aab12ff7391903235d3"
EXPECTED_HOST_MODEL = "NVIDIA DGX Spark"
EXPECTED_ACCELERATOR_TOKEN = "GB10"
EXPECTED_ARCHITECTURES = frozenset({"aarch64", "arm64", "linux/arm64"})
EXPECTED_SAMPLE_RATE_HZ = 24_000
EXPECTED_SAMPLING_INTERVAL_MS = 100
EXPECTED_WARMUPS_PER_RUN = 24
PROFILE_ORDER = (("B1", 1), ("B3", 3), ("B6", 6))
ENGINE_ORDER = ("native", "sglang")
ROUND_ORDER = (1, 2)
EXPECTED_RUN_KEYS = {
    (engine, profile, round_number)
    for round_number in ROUND_ORDER
    for engine in ENGINE_ORDER
    for profile, _ in PROFILE_ORDER
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MANAGED_FILENAMES = (
    "evidence_placeholders.tex",
    "native-runs.dat",
    "sglang-runs.dat",
)


class FinalizationError(ValueError):
    """Raised when publishable paper data cannot be derived exactly."""


@dataclass(frozen=True)
class FinalizedOutputs:
    tex: bytes
    native_dat: bytes
    sglang_dat: bytes

    def by_name(self) -> dict[str, bytes]:
        return {
            "evidence_placeholders.tex": self.tex,
            "native-runs.dat": self.native_dat,
            "sglang-runs.dat": self.sglang_dat,
        }


def _fail(message: str) -> NoReturn:
    raise FinalizationError(message)


def _load_report_module() -> Any:
    repository_root = Path(__file__).resolve().parents[3]
    module_path = repository_root / "reports" / "generate_report.py"
    spec = importlib.util.spec_from_file_location(
        "qwen3_tts_paper_report_validator", module_path
    )
    if spec is None or spec.loader is None:
        _fail(f"cannot load report validator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validated_bundle(manifest_path: Path) -> tuple[Any, dict[str, Any]]:
    report = _load_report_module()
    try:
        bundle = report.load_bundle(manifest_path, allow_test_fixture=False)
        aggregates = report.aggregate(bundle)
    except report.EvidenceError as exc:
        raise FinalizationError(
            f"production evidence validation failed: {exc}"
        ) from exc
    manifest = bundle.manifest
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _fail(
            f"paper finalization requires schema {SCHEMA_VERSION} production evidence"
        )
    if manifest.get("evidence_kind") != "production":
        _fail("paper finalization refuses test fixtures")
    return bundle, aggregates


def _implementation_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    implementations = manifest.get("implementations")
    if not isinstance(implementations, list) or len(implementations) != 2:
        _fail("manifest must contain exactly Native and stock SGLang implementations")
    result: dict[str, dict[str, Any]] = {}
    for item in implementations:
        if not isinstance(item, dict):
            _fail("implementation declaration is not an object")
        engine = item.get("id")
        if engine not in ENGINE_ORDER or item.get("role") != engine or engine in result:
            _fail(
                "implementation IDs and roles must uniquely identify native and sglang"
            )
        result[engine] = item
    if set(result) != set(ENGINE_ORDER):
        _fail("both implementation identities are required")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label}: expected an exact lowercase SHA-256")
    return value


def _require_image_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or IMAGE_ID_RE.fullmatch(value) is None:
        _fail(f"{label}: expected sha256:<64 lowercase hex>")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_COMMIT_RE.fullmatch(value) is None:
        _fail(f"{label}: paper identity requires a full 40-character Git commit")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label}: expected a finite number")
    if positive and result <= 0:
        _fail(f"{label}: expected a positive number")
    if not positive and result < 0:
        _fail(f"{label}: expected a non-negative number")
    return result


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label}: expected an integer")
    if positive and value <= 0:
        _fail(f"{label}: expected a positive integer")
    if not positive and value < 0:
        _fail(f"{label}: expected a non-negative integer")
    return value


def _json_number(value: Any, label: str) -> str:
    _finite_number(value, label)
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _tex_number(value: Any, digits: int, label: str) -> str:
    return f"{_finite_number(value, label):.{digits}f}"


def _tex_integer(value: Any, label: str) -> str:
    return str(_integer(value, label))


def _tex_token(value: str) -> str:
    if not value or re.fullmatch(r"[A-Za-z0-9:.+/_-]+", value) is None:
        _fail("unsafe identity token for TeX output")
    pieces = [value[index : index + 8] for index in range(0, len(value), 8)]
    escaped = [piece.replace("_", r"\_") for piece in pieces]
    return r"\code{" + r"\allowbreak{}".join(escaped) + "}"


def _paper_protocol_gate(manifest: dict[str, Any]) -> None:
    system = manifest.get("system")
    model = manifest.get("model")
    workload = manifest.get("workload")
    methodology = manifest.get("methodology")
    if not all(
        isinstance(item, dict) for item in (system, model, workload, methodology)
    ):
        _fail("paper protocol identity is incomplete")
    if system["host_model"] != EXPECTED_HOST_MODEL:
        _fail(f"paper target requires system.host_model={EXPECTED_HOST_MODEL!r}")
    if system["architecture"].lower() not in EXPECTED_ARCHITECTURES:
        _fail("paper target requires an ARM64 system architecture")
    if EXPECTED_ACCELERATOR_TOKEN not in system["accelerator"]:
        _fail("paper target requires an NVIDIA GB10 accelerator")
    if model["repository"] != EXPECTED_MODEL_REPOSITORY:
        _fail(f"paper target requires model.repository={EXPECTED_MODEL_REPOSITORY!r}")
    if model["revision"] != EXPECTED_MODEL_REVISION:
        _fail(f"paper target requires model.revision={EXPECTED_MODEL_REVISION}")
    expected_workload = {
        "sample_rate_hz": EXPECTED_SAMPLE_RATE_HZ,
        "channels": 1,
        "sample_format": "pcm_s16le",
        "response_mode": "streaming",
        "warmup_requests_per_run": EXPECTED_WARMUPS_PER_RUN,
        "minimum_rounds_per_subject": 2,
    }
    for field, expected in expected_workload.items():
        if workload[field] != expected:
            _fail(f"paper protocol requires workload.{field}={expected!r}")
    if methodology["sampling_interval_ms"] != EXPECTED_SAMPLING_INTERVAL_MS:
        _fail(
            "paper protocol requires methodology.sampling_interval_ms="
            f"{EXPECTED_SAMPLING_INTERVAL_MS}"
        )


def _registry_values(implementation: dict[str, Any], label: str) -> tuple[str, str]:
    registry = implementation.get("registry_image")
    if registry is None:
        return "N/A", "N/A"
    if not isinstance(registry, dict):
        _fail(f"{label}.registry_image: expected an object")
    evidence = registry.get("evidence")
    if not isinstance(evidence, dict):
        _fail(f"{label}.registry_image.evidence: missing digest-bound evidence")
    _require_sha256(evidence.get("sha256"), f"{label}.registry_image.evidence.sha256")
    digest = _require_image_id(
        registry.get("manifest_digest"), f"{label}.registry_image.manifest_digest"
    )
    compressed = registry.get("compressed_size_bytes")
    if compressed is None:
        return digest, "N/A"
    compressed_bytes = _integer(
        compressed, f"{label}.registry_image.compressed_size_bytes", positive=True
    )
    return digest, f"{compressed_bytes / (1024**3):.3f}"


def _artifact_identity(
    implementation: dict[str, Any], label: str
) -> tuple[str, int, int]:
    artifact = implementation.get("model_artifact")
    if not isinstance(artifact, dict):
        _fail(f"{label}.model_artifact: missing")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, dict):
        _fail(f"{label}.model_artifact.evidence: missing")
    digest = _require_sha256(
        evidence.get("sha256"), f"{label}.model_artifact.evidence.sha256"
    )
    parameters = _integer(
        artifact.get("parameter_count"),
        f"{label}.model_artifact.parameter_count",
        positive=True,
    )
    weights = artifact.get("weight_files")
    if not isinstance(weights, list) or not weights:
        _fail(f"{label}.model_artifact.weight_files: expected a non-empty array")
    weight_bytes = 0
    parameter_sum = 0
    for index, weight in enumerate(weights):
        if not isinstance(weight, dict):
            _fail(f"{label}.model_artifact.weight_files[{index}]: expected an object")
        weight_bytes += _integer(
            weight.get("bytes"),
            f"{label}.model_artifact.weight_files[{index}].bytes",
            positive=True,
        )
        parameter_sum += _integer(
            weight.get("parameter_count"),
            f"{label}.model_artifact.weight_files[{index}].parameter_count",
            positive=True,
        )
    if parameter_sum != parameters:
        _fail(f"{label}.model_artifact.parameter_count: weight sum mismatch")
    return digest, parameters, weight_bytes


def _run_maps(bundle: Any) -> tuple[dict[Any, Any], dict[Any, Any]]:
    summaries = bundle.run_summaries
    resources = bundle.run_resources
    if set(summaries) != EXPECTED_RUN_KEYS:
        _fail(
            "paper requires exactly twelve run summaries; "
            f"missing={sorted(EXPECTED_RUN_KEYS - set(summaries))} "
            f"extra={sorted(set(summaries) - EXPECTED_RUN_KEYS)}"
        )
    if set(resources) != EXPECTED_RUN_KEYS:
        _fail(
            "paper requires exactly twelve run resources; "
            f"missing={sorted(EXPECTED_RUN_KEYS - set(resources))} "
            f"extra={sorted(set(resources) - EXPECTED_RUN_KEYS)}"
        )
    return summaries, resources


def _performance_rows(summaries: dict[Any, Any]) -> str:
    rows: list[str] = []
    labels = {"native": "Native", "sglang": "Stock SGLang"}
    for round_number in ROUND_ORDER:
        for engine in ENGINE_ORDER:
            for profile, concurrency in PROFILE_ORDER:
                key = (engine, profile, round_number)
                summary = summaries[key]
                distribution = summary.get("ttfa_ms")
                if not isinstance(distribution, dict):
                    _fail(f"run {key}: missing TTFA distribution")
                rows.append(
                    "  "
                    + " & ".join(
                        (
                            labels[engine],
                            str(round_number),
                            profile,
                            str(concurrency),
                            _tex_integer(
                                summary.get("successful_requests"),
                                f"run {key}.successful_requests",
                            ),
                            _tex_number(
                                distribution.get("p50"), 3, f"run {key}.ttfa_ms.p50"
                            ),
                            _tex_number(
                                distribution.get("p95"), 3, f"run {key}.ttfa_ms.p95"
                            ),
                            _tex_number(
                                summary.get("aggregate_rtf"),
                                4,
                                f"run {key}.aggregate_rtf",
                            ),
                        )
                    )
                    + r" \\"
                )
    return "\n".join(rows)


def _resource_rows(summaries: dict[Any, Any], resources: dict[Any, Any]) -> str:
    rows: list[str] = []
    labels = {"native": "Native", "sglang": "Stock SGLang"}
    for round_number in ROUND_ORDER:
        for engine in ENGINE_ORDER:
            for profile, _ in PROFILE_ORDER:
                key = (engine, profile, round_number)
                summary = summaries[key]
                resource = resources[key]
                audio_seconds = _finite_number(
                    summary.get("total_audio_seconds"),
                    f"run {key}.total_audio_seconds",
                    positive=True,
                )
                energy = _finite_number(resource.get("energy_j"), f"run {key}.energy_j")
                energy_per_audio_minute = energy * 60.0 / audio_seconds
                rows.append(
                    "  "
                    + " & ".join(
                        (
                            labels[engine],
                            str(round_number),
                            profile,
                            f"{_integer(resource.get('process_rss_peak_bytes'), f'run {key}.process_rss_peak_bytes', positive=True) / (1024**3):.3f}",
                            f"{_integer(resource.get('gpu_unified_memory_peak_bytes'), f'run {key}.gpu_unified_memory_peak_bytes', positive=True) / (1024**3):.3f}",
                            _tex_number(
                                resource.get("average_power_w"),
                                3,
                                f"run {key}.average_power_w",
                            ),
                            f"{energy_per_audio_minute:.3f}",
                        )
                    )
                    + r" \\"
                )
    return "\n".join(rows)


def _plot_data(engine: str, summaries: dict[Any, Any]) -> bytes:
    lines = ["concurrency round ttfa_p95_ms aggregate_rtf\n"]
    for round_number in ROUND_ORDER:
        for profile, concurrency in PROFILE_ORDER:
            key = (engine, profile, round_number)
            summary = summaries[key]
            distribution = summary.get("ttfa_ms")
            if not isinstance(distribution, dict):
                _fail(f"run {key}: missing TTFA distribution")
            lines.append(
                " ".join(
                    (
                        str(concurrency),
                        str(round_number),
                        _json_number(distribution.get("p95"), f"run {key}.ttfa_ms.p95"),
                        _json_number(
                            summary.get("aggregate_rtf"), f"run {key}.aggregate_rtf"
                        ),
                    )
                )
                + "\n"
            )
    return "".join(lines).encode("ascii")


def _summary_macros(aggregates: dict[str, Any]) -> tuple[str, str, str]:
    for engine in ENGINE_ORDER:
        if set(aggregates.get(engine, {})) != {item[0] for item in PROFILE_ORDER}:
            _fail(f"aggregate output is missing required {engine} profiles")

    def profile_values(engine: str, profile: str) -> tuple[str, str]:
        item = aggregates[engine][profile]
        return (
            _tex_number(item["ttfa_p95"], 3, f"aggregate {engine} {profile} TTFA"),
            _tex_number(item["aggregate_rtf"], 4, f"aggregate {engine} {profile} RTF"),
        )

    native_total = sum(
        _integer(aggregates["native"][p]["total"], "native total")
        for p, _ in PROFILE_ORDER
    )
    native_success = sum(
        _integer(aggregates["native"][p]["success"], "native success")
        for p, _ in PROFILE_ORDER
    )
    stock_total = sum(
        _integer(aggregates["sglang"][p]["total"], "stock total")
        for p, _ in PROFILE_ORDER
    )
    stock_success = sum(
        _integer(aggregates["sglang"][p]["success"], "stock success")
        for p, _ in PROFILE_ORDER
    )
    native_natural = sum(
        _integer(aggregates["native"][p]["natural_eos"], "native EOS")
        for p, _ in PROFILE_ORDER
    )
    stock_unknown = sum(
        _integer(aggregates["sglang"][p]["eos_unknown"], "stock EOS")
        for p, _ in PROFILE_ORDER
    )
    values = {
        profile: {
            "native": profile_values("native", profile),
            "sglang": profile_values("sglang", profile),
        }
        for profile, _ in PROFILE_ORDER
    }
    profile_sentences = " ".join(
        (
            f"For {profile}, combined-round TTFA p95 was "
            f"{values[profile]['native'][0]} ms for Native and "
            f"{values[profile]['sglang'][0]} ms for stock SGLang; aggregate "
            f"RTF was {values[profile]['native'][1]} and "
            f"{values[profile]['sglang'][1]}, respectively."
        )
        for profile, _ in PROFILE_ORDER
    )
    detailed = (
        f"Across the six validated runs per engine, Native completed "
        f"{native_success}/{native_total} measured requests and stock SGLang "
        f"completed {stock_success}/{stock_total}. {profile_sentences} "
        f"Native recorded natural EOS for {native_natural}/{native_success} successful "
        f"requests; stock SGLang retained unknown EOS for "
        f"{stock_unknown}/{stock_success} successful requests."
    )
    abstract = (
        f"In the validated two-round comparison, Native completed "
        f"{native_success}/{native_total} measured requests and stock SGLang "
        f"completed {stock_success}/{stock_total}. Combined-round aggregate RTF "
        + "at B1, B3, and B6 was "
        + ", ".join(values[p]["native"][1] for p, _ in PROFILE_ORDER)
        + " for Native and "
        + ", ".join(values[p]["sglang"][1] for p, _ in PROFILE_ORDER)
        + " for stock SGLang, respectively."
    )
    conclusion = (
        f"The accepted evidence contains {native_success + stock_success} successful "
        f"measured responses across twelve runs. {profile_sentences} The Native "
        f"series preserved explicit natural-EOS classification, while stock SGLang "
        f"retained unknown EOS as required by its transport contract."
    )
    return detailed, abstract, conclusion


def build_outputs(bundle: Any, aggregates: dict[str, Any]) -> FinalizedOutputs:
    manifest = bundle.manifest
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _fail(f"expected schema {SCHEMA_VERSION}")
    if manifest.get("evidence_kind") != "production":
        _fail("expected production evidence")
    _paper_protocol_gate(manifest)
    summaries, resources = _run_maps(bundle)
    implementations = _implementation_map(manifest)
    native = implementations["native"]
    stock = implementations["sglang"]

    native_commit = _require_commit(native.get("source_commit"), "native.source_commit")
    stock_commit = _require_commit(stock.get("source_commit"), "sglang.source_commit")
    native_local = _require_image_id(
        native.get("local_image", {}).get("id"), "native.local_image.id"
    )
    stock_local = _require_image_id(
        stock.get("local_image", {}).get("id"), "sglang.local_image.id"
    )
    native_unpacked = _integer(
        native.get("local_image", {}).get("unpacked_size_bytes"),
        "native.local_image.unpacked_size_bytes",
        positive=True,
    )
    stock_unpacked = _integer(
        stock.get("local_image", {}).get("unpacked_size_bytes"),
        "sglang.local_image.unpacked_size_bytes",
        positive=True,
    )
    native_registry, native_compressed = _registry_values(native, "native")
    stock_registry, stock_compressed = _registry_values(stock, "sglang")
    native_artifact, native_parameters, native_weight_bytes = _artifact_identity(
        native, "native"
    )
    stock_artifact, stock_parameters, stock_weight_bytes = _artifact_identity(
        stock, "sglang"
    )
    manifest_sha = _require_sha256(bundle.manifest_sha256, "manifest SHA-256")
    workload_sha = _require_sha256(
        manifest.get("workload", {}).get("corpus_sha256"), "workload corpus SHA-256"
    )
    generated_at = manifest.get("report", {}).get("generated_at")
    if not isinstance(generated_at, str) or not generated_at:
        _fail("report.generated_at: missing")
    model_revision = manifest["model"]["revision"]

    detailed, abstract, conclusion = _summary_macros(aggregates)
    performance_rows = _performance_rows(summaries)
    resource_rows = _resource_rows(summaries, resources)
    container_rows = "\n".join(
        (
            f"  Native & {native_unpacked / (1024**3):.3f} & {native_compressed} & "
            f"{native_parameters} & {native_weight_bytes / (1024**3):.3f} \\\\",
            f"  Stock SGLang & {stock_unpacked / (1024**3):.3f} & {stock_compressed} & "
            f"{stock_parameters} & {stock_weight_bytes / (1024**3):.3f} \\\\",
        )
    )

    def identity_or_na(value: str) -> str:
        return "N/A" if value == "N/A" else _tex_token(value)

    tex = f"""% Generated deterministically from validated schema-v1.2 production evidence.
% Do not edit by hand. Re-run research/paper/tools/finalize_evidence.py.

\\newif\\ifFinalEvidenceAvailable
\\FinalEvidenceAvailabletrue

\\newcommand{{\\PaperEvidenceState}}{{VALIDATED PRODUCTION}}
\\newcommand{{\\NativeCommit}}{{{_tex_token(native_commit)}}}
\\newcommand{{\\StockCommit}}{{{_tex_token(stock_commit)}}}
\\newcommand{{\\NativeLocalImageId}}{{{_tex_token(native_local)}}}
\\newcommand{{\\NativeImageDigest}}{{{identity_or_na(native_registry)}}}
\\newcommand{{\\StockLocalImageId}}{{{_tex_token(stock_local)}}}
\\newcommand{{\\StockImageDigest}}{{{identity_or_na(stock_registry)}}}
\\newcommand{{\\NativeModelArtifactEvidenceSha}}{{{_tex_token(native_artifact)}}}
\\newcommand{{\\StockModelArtifactEvidenceSha}}{{{_tex_token(stock_artifact)}}}
\\newcommand{{\\EvidenceManifestSha}}{{{_tex_token(manifest_sha)}}}
\\newcommand{{\\WorkloadSha}}{{{_tex_token(workload_sha)}}}
\\newcommand{{\\ModelRevision}}{{{_tex_token(model_revision)}}}
\\newcommand{{\\EvidenceTimestamp}}{{{_tex_token(generated_at)}}}

% BEGIN GENERATED:benchmark-result-rows
\\newcommand{{\\BenchmarkResultRows}}{{%
{performance_rows}
}}
% END GENERATED:benchmark-result-rows

% BEGIN GENERATED:resource-result-rows
\\newcommand{{\\ResourceResultRows}}{{%
{resource_rows}
}}
% END GENERATED:resource-result-rows

% BEGIN GENERATED:container-result-rows
\\newcommand{{\\ContainerResultRows}}{{%
{container_rows}
}}
% END GENERATED:container-result-rows

% BEGIN GENERATED:verified-result-summary
\\newcommand{{\\VerifiedResultSummary}}{{%
  {detailed}}}
\\newcommand{{\\VerifiedAbstractResult}}{{%
  {abstract}}}
\\newcommand{{\\VerifiedConclusionResult}}{{%
  {conclusion}}}
% END GENERATED:verified-result-summary
""".encode("ascii")
    if b"PENDING" in tex:
        _fail("generated TeX unexpectedly contains a pending marker")
    return FinalizedOutputs(
        tex=tex,
        native_dat=_plot_data("native", summaries),
        sglang_dat=_plot_data("sglang", summaries),
    )


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        _fail(f"temporary output already exists: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(outputs: FinalizedOutputs, paper_root: Path) -> None:
    paper_root = paper_root.expanduser().resolve(strict=True)
    if (
        not (paper_root / "main.tex").is_file()
        or not (paper_root / "Makefile").is_file()
    ):
        _fail(f"not a paper source root: {paper_root}")
    data_dir = paper_root / "data"
    if data_dir.is_symlink() or not data_dir.is_dir():
        _fail(f"paper data directory is missing or unsafe: {data_dir}")
    target_tex = data_dir / "evidence_placeholders.tex"
    if target_tex.is_symlink() or not target_tex.is_file():
        _fail(f"managed TeX evidence boundary is missing or unsafe: {target_tex}")
    payloads = outputs.by_name()
    for name in MANAGED_FILENAMES:
        target = data_dir / name
        if target.is_symlink():
            _fail(f"managed output is a symlink: {target}")
        if (
            name != "evidence_placeholders.tex"
            and target.exists()
            and not target.is_file()
        ):
            _fail(f"managed output is not a regular file: {target}")
        if not payloads[name]:
            _fail(f"refusing to write empty managed output: {name}")
    for name in MANAGED_FILENAMES:
        _atomic_replace(data_dir / name, payloads[name])


def finalize(manifest_path: Path, paper_root: Path) -> FinalizedOutputs:
    bundle, aggregates = _validated_bundle(manifest_path.expanduser())
    outputs = build_outputs(bundle, aggregates)
    write_outputs(outputs, paper_root)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize paper data from validated schema-v1.2 production evidence."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Paper source root; defaults to research/paper",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        outputs = finalize(args.manifest, args.paper_root)
    except (FinalizationError, OSError) as exc:
        print(f"paper finalization failed: {exc}", file=sys.stderr)
        return 1
    print(
        "finalized validated paper evidence: "
        f"{len(outputs.native_dat.splitlines()) - 1} Native plot rows, "
        f"{len(outputs.sglang_dat.splitlines()) - 1} stock SGLang plot rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
