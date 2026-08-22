"""Contract: every pipeline stage returns the same StageResult shape.

Checked against the discovery stage now; as profiling, inference, training,
etc. are added, add their StageResult-producing calls to
STAGE_RUNNERS below rather than writing a new bespoke shape check.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.training.database.artifact.service import run_artifact_export, run_generation_from_artifact
from synth_platform.engine.profiling.database.cleaning.service import run_cleaning
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import StageResult
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.service import run_document_profiling
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.training.database.service import run_training_and_sampling
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.contract

REQUIRED_FIELD_TYPES: dict[str, type] = {
    "stage_name": str,
    "status": str,
    "input_references": list,
    "output_references": list,
    "metrics": dict,
    "warnings": list,
    "errors": list,
    "evidence": dict,
}


def assert_valid_stage_result(result: StageResult) -> None:
    assert isinstance(result, StageResult)
    for field_name, expected_type in REQUIRED_FIELD_TYPES.items():
        value = getattr(result, field_name)
        assert isinstance(value, expected_type), (
            f"{field_name} expected {expected_type}, got {type(value)}"
        )
    assert result.status in {"success", "failed"}


def _run_discovery_success(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    return run_discovery(adapter, manifest, config_path="config/project.yaml")


def _run_discovery_failure(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter = SQLiteSourceAdapter({"path": str(tmp_path / "missing.db")})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    return run_discovery(adapter, manifest, config_path="config/project.yaml")


def _run_profiling_success(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    return run_profiling(
        adapter, discovery_data, manifest, discovery_reference="discovery.json", sample_limit=100
    )


def _run_profiling_failure(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    discovery_data["tables"]["__does_not_exist__"] = discovery_data["tables"]["orders"]
    return run_profiling(
        adapter, discovery_data, manifest, discovery_reference="discovery.json", sample_limit=100
    )


def _discover_and_profile(tmp_path: Path, temp_sqlite_db: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    profiling_result = run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())
    return adapter, manifest, discovery_data, profile_data


def _run_inference_success(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter, manifest, discovery_data, profile_data = _discover_and_profile(tmp_path, temp_sqlite_db)
    return run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
    )


def _run_inference_failure(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter, manifest, discovery_data, profile_data = _discover_and_profile(tmp_path, temp_sqlite_db)
    discovery_data["tables"]["__does_not_exist__"] = discovery_data["tables"]["orders"]
    return run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
    )


def _run_contract_approval_success(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter, manifest, discovery_data, profile_data = _discover_and_profile(tmp_path, temp_sqlite_db)
    inference_result = run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
    )
    candidates_data = json.loads(Path(inference_result.output_references[0]).read_text())
    return run_contract_approval(
        dataset_id="ds",
        source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data,
        candidates_by_table=candidates_data["tables"],
        manifest=manifest,
        metadata_dir=tmp_path / "metadata",
        candidates_reference="semantic_candidates.json",
    )


def _discover_profile_infer_approve(tmp_path: Path, temp_sqlite_db: Path):
    adapter, manifest, discovery_data, profile_data = _discover_and_profile(tmp_path, temp_sqlite_db)
    inference_result = run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
    )
    candidates_data = json.loads(Path(inference_result.output_references[0]).read_text())
    metadata_dir = tmp_path / "metadata"
    run_contract_approval(
        dataset_id="ds",
        source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data,
        candidates_by_table=candidates_data["tables"],
        manifest=manifest,
        metadata_dir=metadata_dir,
        candidates_reference="semantic_candidates.json",
        decisions={
            table: {col: cand["semantic_type"] for col, cand in cols.items()}
            for table, cols in candidates_data["tables"].items()
        },
    )
    contract = load_dataset_contract(metadata_dir)
    return adapter, manifest, contract


def _run_cleaning_success(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter, manifest, contract = _discover_profile_infer_approve(tmp_path, temp_sqlite_db)
    return run_cleaning(adapter, contract, manifest, "dataset_contract.json", sample_limit=100)


def _run_cleaning_failure(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter, manifest, contract = _discover_profile_infer_approve(tmp_path, temp_sqlite_db)
    contract["tables"]["__does_not_exist__"] = contract["tables"]["orders"]
    return run_cleaning(adapter, contract, manifest, "dataset_contract.json", sample_limit=100)


def _run_document_profiling_success(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Contact Ada Lovelace at ada@example.com."])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    return run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "doc1", manifest, tmp_path / "metadata"
    )


def _run_document_profiling_failure(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    return run_document_profiling(
        PDFDocumentAdapter(), tmp_path / "does_not_exist.pdf", "doc1", manifest, tmp_path / "metadata"
    )


def _run_training_success(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter, manifest, contract = _discover_profile_infer_approve(tmp_path, temp_sqlite_db)
    return run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json",
        sample_limit=100, num_rows_to_generate=5, seed=1,
    )


def _run_training_failure(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    adapter, manifest, contract = _discover_profile_infer_approve(tmp_path, temp_sqlite_db)
    contract["tables"]["__does_not_exist__"] = contract["tables"]["orders"]
    return run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json",
        sample_limit=100, num_rows_to_generate=5, seed=1,
    )


def _discover_profile_infer_approve_train(tmp_path: Path, temp_sqlite_db: Path):
    adapter, manifest, discovery_data, profile_data = _discover_and_profile(tmp_path, temp_sqlite_db)
    inference_result = run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
    )
    candidates_data = json.loads(Path(inference_result.output_references[0]).read_text())
    metadata_dir = tmp_path / "metadata"
    run_contract_approval(
        dataset_id="ds",
        source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data,
        candidates_by_table=candidates_data["tables"],
        manifest=manifest,
        metadata_dir=metadata_dir,
        candidates_reference="semantic_candidates.json",
        decisions={
            table: {col: cand["semantic_type"] for col, cand in cols.items()}
            for table, cols in candidates_data["tables"].items()
        },
    )
    contract = load_dataset_contract(metadata_dir)
    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=5, seed=1
    )
    training_report = json.loads(Path(training_result.output_references[0]).read_text())
    return manifest, discovery_data, profile_data, contract, training_report


def _run_artifact_export_success(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    manifest, discovery_data, profile_data, contract, training_report = (
        _discover_profile_infer_approve_train(tmp_path, temp_sqlite_db)
    )
    return run_artifact_export(
        "ds", "1.0.0", discovery_data, contract, profile_data, training_report,
        manifest.run_id, manifest.code_version, manifest, "training_report.json",
    )


def _run_artifact_export_failure(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    manifest, discovery_data, profile_data, contract, training_report = (
        _discover_profile_infer_approve_train(tmp_path, temp_sqlite_db)
    )
    training_report["tables"]["customers"]["model_path"] = str(tmp_path / "missing_model.pkl")
    return run_artifact_export(
        "ds", "1.0.0", discovery_data, contract, profile_data, training_report,
        manifest.run_id, manifest.code_version, manifest, "training_report.json",
    )


def _run_generation_success(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    manifest, discovery_data, profile_data, contract, training_report = (
        _discover_profile_infer_approve_train(tmp_path, temp_sqlite_db)
    )
    export_result = run_artifact_export(
        "ds", "1.0.0", discovery_data, contract, profile_data, training_report,
        manifest.run_id, manifest.code_version, manifest, "training_report.json",
    )
    generation_manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    return run_generation_from_artifact(
        export_result.output_references[0], "customers", 5, 1, generation_manifest, allow_cloudpickle_models=True,
    )


def _run_generation_failure(tmp_path: Path, temp_sqlite_db: Path) -> StageResult:
    generation_manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    return run_generation_from_artifact(
        tmp_path / "does_not_exist.zip", "customers", 5, 1, generation_manifest
    )


STAGE_RUNNERS: dict[str, Callable[[Path, Path], StageResult]] = {
    "discovery_success": _run_discovery_success,
    "discovery_failure": _run_discovery_failure,
    "profiling_success": _run_profiling_success,
    "profiling_failure": _run_profiling_failure,
    "inference_success": _run_inference_success,
    "inference_failure": _run_inference_failure,
    "contract_approval_success": _run_contract_approval_success,
    "cleaning_success": _run_cleaning_success,
    "cleaning_failure": _run_cleaning_failure,
    "document_profiling_success": _run_document_profiling_success,
    "document_profiling_failure": _run_document_profiling_failure,
    "training_success": _run_training_success,
    "training_failure": _run_training_failure,
    "artifact_export_success": _run_artifact_export_success,
    "artifact_export_failure": _run_artifact_export_failure,
    "generation_success": _run_generation_success,
    "generation_failure": _run_generation_failure,
}


@pytest.mark.parametrize("stage_runner", STAGE_RUNNERS.values(), ids=STAGE_RUNNERS.keys())
def test_stage_result_shape_is_consistent(
    stage_runner: Callable[[Path, Path], StageResult],
    tmp_path: Path,
    temp_sqlite_db: Path,
):
    result = stage_runner(tmp_path, temp_sqlite_db)
    assert_valid_stage_result(result)
