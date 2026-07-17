from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


metadata = load_module(
    "prepare_production_metadata_tests", TOOLS_DIR / "prepare_production_metadata.py"
)
fixtures = load_module(
    "assemble_production_manifest_fixtures",
    TOOLS_DIR / "tests" / "test_assemble_production_manifest.py",
)


def declarations_from(config: dict) -> dict:
    return {
        "report": config["report"],
        "system": config["system"],
        "workload": {
            "profiles": config["workload"]["profiles"],
            "language_policy": config["workload"]["language_policy"],
        },
        "implementations": [
            {
                key: implementation[key]
                for key in (
                    "id",
                    "name",
                    "version",
                    "source_commit",
                    "source_url",
                    "api_protocol",
                    "streaming_semantics",
                    "runtime_components",
                )
            }
            for implementation in config["implementations"]
        ],
        "methodology": config["methodology"],
        "limitations": config["limitations"],
    }


class PrepareProductionMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = fixtures.EvidenceFixture(self.root)
        self.declarations = self.root / "declarations.json"
        self.declaration_value = declarations_from(self.fixture.config_value)
        self.declarations.write_text(
            json.dumps(self.declaration_value, indent=2) + "\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_builds_exact_expanded_metadata(self) -> None:
        observed = metadata.build_metadata(self.declarations, self.fixture.evidence)
        self.assertEqual(observed, self.fixture.config_value)
        self.assertEqual(
            metadata.encode_metadata(observed), metadata.encode_metadata(observed)
        )

    def test_rejects_unresolved_declaration_placeholder(self) -> None:
        self.declaration_value["system"]["kernel"] = "PENDING_KERNEL"
        self.declarations.write_text(
            json.dumps(self.declaration_value), encoding="utf-8"
        )
        with self.assertRaisesRegex(metadata.MetadataError, "unresolved placeholder"):
            metadata.build_metadata(self.declarations, self.fixture.evidence)

    def test_rejects_image_identity_drift_across_runs(self) -> None:
        run = self.fixture.run_dir("native", "B1", 1)
        invocation_path = run / "provenance" / "invocation.json"
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        invocation["image"]["reference"] = "fixture/native:different"
        fixtures.write_json(invocation_path, invocation)
        fixtures.refresh_checksums(run)
        with self.assertRaisesRegex(
            metadata.MetadataError, "inconsistent local images"
        ):
            metadata.build_metadata(self.declarations, self.fixture.evidence)

    def test_output_is_create_new(self) -> None:
        output = self.root / "metadata.json"
        metadata.write_create_new(output, b"{}\n")
        with self.assertRaisesRegex(metadata.MetadataError, "refusing to overwrite"):
            metadata.write_create_new(output, b"{}\n")


if __name__ == "__main__":
    unittest.main()
