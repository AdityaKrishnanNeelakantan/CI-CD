"""Integration: discovery -> profiling -> inference -> contract approval ->
training/sampling, verifying the generated data is genuinely synthetic
(never a real training value) and structurally valid against the approved
contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling

pytestmark = pytest.mark.integration


def _approve_everything(adapter, manifest, metadata_dir):
    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    profiling_result = run_profiling(
        adapter, discovery_data, manifest, str(discovery_result.output_references[0]), sample_limit=100
    )
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())
    inference_result = run_inference(
        adapter, discovery_data, profile_data, manifest,
        str(discovery_result.output_references[0]), str(profiling_result.output_references[0]),
        sample_limit=100,
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
        candidates_reference=str(inference_result.output_references[0]),
        decisions=decisions,
    )
    return load_dataset_contract(metadata_dir)


def test_generated_data_is_synthetic_and_structurally_valid(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"
    contract = _approve_everything(adapter, manifest, metadata_dir)

    result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json",
        sample_limit=100, num_rows_to_generate=25, seed=3,
    )
    assert result.is_success()

    report = load_training_report(manifest.output_path("training_report.json"))
    customers_report = report["tables"]["customers"]
    generated = pd.read_csv(customers_report["sample_path"])

    real_df = adapter.read_sample("customers", limit=100)
    assert len(generated) == 25

    # customer_id was approved as identifier and is the table's primary key.
    assert generated["customer_id"].is_unique
    assert set(generated["customer_id"]) & set(real_df["customer_id"]) == set()

    manifest_data = json.loads((manifest.run_dir / "run_manifest.json").read_text())
    stage_names = [s["stage_name"] for s in manifest_data["stages"]]
    assert stage_names == ["discovery", "profiling", "inference", "contract_approval", "training"]


def test_model_can_be_reloaded_and_resampled_in_a_fresh_process(temp_sqlite_db: Path, tmp_path: Path):
    from synth_platform.engine.training.database.adapters.sdv_adapter import SDVSynthesizerAdapter

    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"
    contract = _approve_everything(adapter, manifest, metadata_dir)

    run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json",
        sample_limit=100, num_rows_to_generate=5, seed=3,
    )
    report = load_training_report(manifest.output_path("training_report.json"))
    model_path = report["tables"]["customers"]["model_path"]

    # Simulate a completely fresh process loading only the saved model file.
    reloaded = SDVSynthesizerAdapter.load(model_path)
    generated = reloaded.sample(10, seed=99)
    assert len(generated) == 10
