"""Assembles a portable, source-free artifact package.

discovery_data and dataset_contract never contained raw source rows,
connection info, or credentials to begin with (see each stage's own
docstring) - reference_profile is different: src/profiling/profiler.py's
output deliberately keeps exact minimum/maximum/category_frequencies/
representative_examples for in-pipeline consumers, so this module is the
one place that strips them via src/privacy/profile_sanitizer.py before
anything is written to reference_profile.json. No connection info,
credentials, or absolute training-machine paths are added by this module
either - see tests/negative/test_artifact_negative.py for an explicit
scan proving the built archive never contains them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from synth_platform.engine.training.database.artifact.errors import ArtifactBuildError
from synth_platform.engine.training.database.artifact.manifest import build_manifest
from synth_platform.engine.common.database.privacy.profile_sanitizer import sanitize_profile_for_export

CHECKSUMS_FILENAME = "checksums.sha256"
MANIFEST_FILENAME = "manifest.json"
SELF_TEST_FILENAME = "self_test.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _build_self_test(training_report: dict[str, Any]) -> dict[str, Any]:
    tables = {
        table_name: {
            "expected_columns": table_report["fit_evidence"]["trained_columns"],
            "min_generate_rows": 5,
        }
        for table_name, table_report in training_report["tables"].items()
    }
    return {"self_test_version": "1.0", "tables": tables}


def build_artifact(
    dataset_id: str,
    artifact_version: str,
    discovery_data: dict[str, Any],
    dataset_contract: dict[str, Any],
    reference_profile: dict[str, Any],
    training_report: dict[str, Any],
    training_run_id: str,
    code_version: str,
    output_path: str | Path,
) -> Path:
    """Build a self-contained generator zip at output_path."""
    output_path = Path(output_path)
    if output_path.exists():
        raise ArtifactBuildError(
            f"artifact already exists at {output_path}; a stage output must never be overwritten"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    staging_dir = output_path.parent / f".{output_path.stem}.staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    try:
        (staging_dir / "schema").mkdir()
        _write_json(staging_dir / "schema" / "database_catalog.json", discovery_data)

        (staging_dir / "semantics").mkdir()
        _write_json(staging_dir / "semantics" / "dataset_contract.json", dataset_contract)

        (staging_dir / "validation").mkdir()
        _write_json(
            staging_dir / "validation" / "reference_profile.json",
            sanitize_profile_for_export(reference_profile, dataset_contract),
        )

        models_root = staging_dir / "models" / "tables"
        models_root.mkdir(parents=True)
        for table_name, table_report in training_report["tables"].items():
            table_dir = models_root / table_name
            table_dir.mkdir()
            source_model_path = Path(table_report["model_path"])
            if not source_model_path.is_file():
                raise ArtifactBuildError(f"trained model file not found: {source_model_path}")
            # Destination keeps the source's real extension (.pkl vs
            # .json) rather than a hardcoded "model.pkl" - a JSON model
            # must never be renamed to look like a pickle file, or vice
            # versa; see src/synthesis/base.py's file_extension contract.
            try:
                shutil.copy2(source_model_path, table_dir / f"model{source_model_path.suffix}")
            except OSError as exc:
                raise ArtifactBuildError(
                    f"could not copy trained model file for table {table_name!r}: {exc}"
                ) from exc

            sidecar_path = source_model_path.with_name(source_model_path.name + ".meta.json")
            if sidecar_path.is_file():
                try:
                    shutil.copy2(sidecar_path, table_dir / "model.pkl.meta.json")
                except OSError as exc:
                    raise ArtifactBuildError(
                        f"could not copy model sidecar file for table {table_name!r}: {exc}"
                    ) from exc

        _write_json(staging_dir / SELF_TEST_FILENAME, _build_self_test(training_report))

        manifest = build_manifest(
            dataset_id,
            artifact_version,
            discovery_data,
            dataset_contract,
            training_report,
            training_run_id,
            code_version,
        )
        _write_json(staging_dir / MANIFEST_FILENAME, manifest)

        checksums = {
            file_path.relative_to(staging_dir).as_posix(): _sha256_file(file_path)
            for file_path in staging_dir.rglob("*")
            if file_path.is_file()
        }
        with (staging_dir / CHECKSUMS_FILENAME).open("w", encoding="utf-8") as f:
            for rel_path, digest in sorted(checksums.items()):
                f.write(f"{digest}  {rel_path}\n")

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(staging_dir.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(staging_dir).as_posix())
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return output_path
