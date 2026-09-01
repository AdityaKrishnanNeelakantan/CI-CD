"""End-to-end: a DP-trained model flows through the *entire* downstream
pipeline unchanged - relational generation (FK linking), QA validation,
and target-database write all work identically regardless of which
SynthesizerAdapter trained the model, proving the DP adapter is a true
drop-in replacement, not a special-cased dead end.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import PROFILE_FILENAME, load_profile, run_profiling
from synth_platform.engine.validation.database.qa_service import QA_REPORT_FILENAME, load_qa_report, run_qa_validation
from synth_platform.engine.generation.database.relational_service import (
    RELATIONAL_REPORT_FILENAME,
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.generation.database.target_write_service import run_target_write
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling

pytestmark = pytest.mark.e2e

_COLUMN_BOUNDS = {"amount": (0.0, 1000.0), "signup_date": ("2000-01-01", "2030-01-01")}


def test_dp_trained_model_flows_through_relational_generation_qa_and_target_write(
    temp_sqlite_db: Path, tmp_path: Path
):
    source_adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    run_discovery(source_adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads((manifest.run_dir / "discovery.json").read_text())
    run_profiling(source_adapter, discovery_data, manifest, "discovery.json", sample_limit=100)
    profile_data = load_profile(manifest.output_path(PROFILE_FILENAME))
    inference_result = run_inference(
        source_adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
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
        source_adapter, contract, manifest, "dataset_contract.json", sample_limit=100,
        num_rows_to_generate=5, seed=1, model_type="dp_gaussian_copula",
        model_kwargs={"epsilon_budget": 3.0, "delta_budget": 1e-5, "column_bounds": _COLUMN_BOUNDS},
    )
    assert training_result.is_success()
    training_report = load_training_report(Path(training_result.output_references[0]))

    adapters = {
        table_name: get_synthesizer_adapter_class(table_report["model_type"]).load(table_report["model_path"])
        for table_name, table_report in training_report["tables"].items()
    }

    relational_result = run_relational_generation(
        contract, adapters, {"customers": 10, "orders": 40}, manifest,
        contract_reference="dataset_contract.json", seed=5,
    )
    assert relational_result.is_success()
    assert relational_result.metrics["overall_fk_validity"] == 1.0
    relational_report = load_relational_generation_report(manifest.output_path(RELATIONAL_REPORT_FILENAME))

    qa_result = run_qa_validation(
        contract, relational_report, manifest, relational_report_reference="relational_generation_report.json",
        reference_profile=profile_data,
    )
    assert qa_result.is_success()
    qa_report = load_qa_report(manifest.output_path(QA_REPORT_FILENAME))
    assert qa_report["hard_checks_passed"] is True

    target_db_path = tmp_path / "database_b.db"
    write_result = run_target_write(
        target_db_path, contract, relational_report, qa_report, manifest, qa_report_reference="qa_report.json"
    )
    assert write_result.is_success()

    conn = sqlite3.connect(str(target_db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 40
    orphaned = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers)"
    ).fetchone()[0]
    assert orphaned == 0
    conn.close()

    # The privacy evidence must be traceable all the way from training
    # report through to whatever a reviewer inspects afterward.
    for table_report in training_report["tables"].values():
        assert table_report["fit_evidence"]["privacy_summary"]["total_epsilon_spent"] <= 3.0 + 1e-9
