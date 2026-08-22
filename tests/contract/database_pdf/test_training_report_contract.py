"""Contract: run_training_and_sampling() always writes the same top-level
shape, and every table entry carries the same evidence fields, regardless
of the table's schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling

pytestmark = pytest.mark.contract

REQUIRED_TOP_LEVEL_KEYS = {"trained_at", "tables"}
REQUIRED_TABLE_KEYS = {
    "model_type",
    "seed",
    "data_fingerprint",
    "fit_evidence",
    "pre_input_mask",
    "generated_row_count",
    "model_path",
    "sample_path",
    "serialization_format",
}


def test_training_report_shape(temp_sqlite_db: Path, tmp_path: Path):
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

    result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json",
        sample_limit=100, num_rows_to_generate=5, seed=1,
    )
    assert result.is_success()

    report = load_training_report(manifest.output_path("training_report.json"))
    assert set(report) == REQUIRED_TOP_LEVEL_KEYS
    for table_report in report["tables"].values():
        assert set(table_report) == REQUIRED_TABLE_KEYS
        assert Path(table_report["model_path"]).is_file()
        assert Path(table_report["sample_path"]).is_file()
