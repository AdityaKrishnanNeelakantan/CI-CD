"""End-to-end: discovery -> profiling -> inference -> approval ->
training -> relational generation, fully traceable and re-read entirely
from disk afterward, using the project's real two-table fixture
(customers -> orders) - the database track's relational analogue of the
other tests/e2e/test_*_e2e.py files.
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
from synth_platform.engine.profiling.database.service import PROFILE_FILENAME, run_profiling
from synth_platform.engine.generation.database.relational_service import (
    RELATIONAL_REPORT_FILENAME,
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling

pytestmark = pytest.mark.e2e


def test_full_relational_track_is_fully_traceable_end_to_end(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads((manifest.run_dir / "discovery.json").read_text())

    run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)
    profile_data = json.loads(manifest.output_path(PROFILE_FILENAME).read_text())

    inference_result = run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
    )
    candidates_data = json.loads(Path(inference_result.output_references[0]).read_text())
    decisions = {
        table: {col: cand["semantic_type"] for col, cand in cols.items()}
        for table, cols in candidates_data["tables"].items()
    }
    run_contract_approval(
        dataset_id="ds", source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data, candidates_by_table=candidates_data["tables"],
        manifest=manifest, metadata_dir=metadata_dir, candidates_reference="semantic_candidates.json",
        decisions=decisions,
    )
    contract = load_dataset_contract(metadata_dir)

    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100,
        num_rows_to_generate=5, seed=3, model_type="safe_gaussian_copula",
    )
    training_report = load_training_report(Path(training_result.output_references[0]))

    adapters = {
        table_name: get_synthesizer_adapter_class(table_report["model_type"]).load(table_report["model_path"])
        for table_name, table_report in training_report["tables"].items()
    }

    relational_result = run_relational_generation(
        contract, adapters, {"customers": 15, "orders": 60}, manifest,
        contract_reference="dataset_contract.json", seed=99,
    )
    assert relational_result.is_success()

    # Simulate a fresh process: re-read every artifact from disk only.
    manifest_on_disk = json.loads((manifest.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    stage_names = [s["stage_name"] for s in manifest_on_disk["stages"]]
    assert stage_names[-1] == "relational_generation"
    assert all(s["status"] == "success" for s in manifest_on_disk["stages"])

    report = load_relational_generation_report(manifest.output_path(RELATIONAL_REPORT_FILENAME))
    assert report["fk_validity"]["overall_fk_validity"] == 1.0
    assert report["generation_order"] == ["customers", "orders"]

    customers_df = pd.read_csv(report["tables"]["customers"]["path"])
    orders_df = pd.read_csv(report["tables"]["orders"]["path"])
    assert len(customers_df) == 15
    assert len(orders_df) == 60
    assert set(orders_df["customer_id"]) <= set(customers_df["customer_id"])
