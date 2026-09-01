"""End-to-end: the complete database track - discovery -> profiling ->
inference -> approval -> training -> relational_generation ->
qa_validation -> target_write - "Database A -> generate -> Database B"
in full, using the project's real two-table fixture (customers ->
orders), re-read entirely from disk afterward and written into a real,
independent target SQLite database file.
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
from synth_platform.engine.generation.database.target_write_service import (
    TARGET_WRITE_REPORT_FILENAME,
    load_target_write_report,
    run_target_write,
)
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling

pytestmark = pytest.mark.e2e


def test_database_a_to_database_b_is_fully_traceable_end_to_end(temp_sqlite_db: Path, tmp_path: Path):
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
        num_rows_to_generate=5, seed=11, model_type="safe_gaussian_copula",
    )
    training_report = load_training_report(Path(training_result.output_references[0]))

    adapters = {
        table_name: get_synthesizer_adapter_class(table_report["model_type"]).load(table_report["model_path"])
        for table_name, table_report in training_report["tables"].items()
    }

    relational_result = run_relational_generation(
        contract, adapters, {"customers": 12, "orders": 50}, manifest,
        contract_reference="dataset_contract.json", seed=77,
    )
    assert relational_result.is_success()
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
        target_db_path, contract, relational_report, qa_report, manifest,
        qa_report_reference="qa_report.json",
    )
    assert write_result.is_success()

    # Simulate a fresh process: re-read every artifact from disk only.
    manifest_on_disk = json.loads((manifest.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    stage_names = [s["stage_name"] for s in manifest_on_disk["stages"]]
    assert stage_names == [
        "discovery",
        "profiling",
        "inference",
        "contract_approval",
        "training",
        "relational_generation",
        "qa_validation",
        "target_write",
    ]
    assert all(s["status"] == "success" for s in manifest_on_disk["stages"])

    write_report = load_target_write_report(manifest.output_path(TARGET_WRITE_REPORT_FILENAME))
    assert write_report["validation_report"]["all_writes_confirmed"] is True

    # The definitive proof: Database B is a real, independent SQLite
    # database, physically disconnected from Database A, containing the
    # requested row counts with real, enforced foreign-key integrity.
    conn = sqlite3.connect(str(target_db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 12
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 50

    orphaned_orders = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE customer_id NOT IN (SELECT customer_id FROM customers)"
    ).fetchone()[0]
    assert orphaned_orders == 0
    conn.close()
