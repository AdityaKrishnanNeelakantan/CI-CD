"""Integration: discovery -> profiling -> inference -> approval -> cleaning
(implicitly, via training) -> training -> artifact export -> load -> generate.

Verifies the full traceability chain the master spec calls for: generated
dataset -> generation run -> artifact version -> training run -> approved
contract -> source schema fingerprint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.training.database.artifact.loader import load_artifact
from synth_platform.engine.training.database.artifact.service import run_artifact_export, run_generation_from_artifact
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.training.database.service import run_training_and_sampling

pytestmark = pytest.mark.integration


def _train_and_export(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    profiling_result = run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())
    inference_result = run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
    )
    candidates_data = json.loads(Path(inference_result.output_references[0]).read_text())
    decisions = {
        table: {col: cand["semantic_type"] for col, cand in cols.items()}
        for table, cols in candidates_data["tables"].items()
    }
    run_contract_approval(
        dataset_id="ds",
        source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data,
        candidates_by_table=candidates_data["tables"],
        manifest=manifest,
        metadata_dir=metadata_dir,
        candidates_reference="semantic_candidates.json",
        decisions=decisions,
    )
    contract = load_dataset_contract(metadata_dir)

    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100, num_rows_to_generate=10, seed=1
    )
    training_report = json.loads(Path(training_result.output_references[0]).read_text())

    export_result = run_artifact_export(
        "ds", "1.0.0", discovery_data, contract, profile_data, training_report,
        manifest.run_id, manifest.code_version, manifest, "training_report.json",
    )
    return manifest, discovery_data, contract, training_report, export_result


def test_full_chain_is_traceable_from_artifact_alone(temp_sqlite_db: Path, tmp_path: Path):
    manifest, discovery_data, contract, training_report, export_result = _train_and_export(
        temp_sqlite_db, tmp_path
    )
    assert export_result.is_success()
    assert export_result.metrics["self_test_passed"] is True

    artifact_path = export_result.output_references[0]
    loaded = load_artifact(artifact_path)

    # Trace: artifact manifest -> training run -> approved contract -> source fingerprint.
    assert loaded.manifest["training_run_id"] == manifest.run_id
    assert loaded.manifest["contract_revision"] == contract["revision"]
    assert loaded.manifest["source_fingerprint"] == discovery_data["source_fingerprint"]
    assert loaded.dataset_contract["source_fingerprint"] == discovery_data["source_fingerprint"]

    for table_name in training_report["tables"]:
        assert table_name in loaded.manifest["tables"]


def test_generation_from_artifact_produces_recorded_evidence(temp_sqlite_db: Path, tmp_path: Path):
    manifest, discovery_data, contract, training_report, export_result = _train_and_export(
        temp_sqlite_db, tmp_path
    )
    artifact_path = export_result.output_references[0]

    generation_manifest = RunManifest.create(runs_dir=tmp_path / "gen_runs")
    generation_result = run_generation_from_artifact(
        artifact_path, "customers", 15, seed=2, manifest=generation_manifest, allow_cloudpickle_models=True,
    )
    assert generation_result.is_success()
    assert generation_result.metrics["generated_row_count"] == 15

    import pandas as pd

    generated = pd.read_csv(generation_result.output_references[0])
    assert len(generated) == 15

    gen_manifest_data = json.loads((generation_manifest.run_dir / "run_manifest.json").read_text())
    assert gen_manifest_data["stages"][0]["stage_name"] == "generation"
