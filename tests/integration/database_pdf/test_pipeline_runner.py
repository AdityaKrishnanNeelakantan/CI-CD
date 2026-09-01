"""Integration test: the full Database Twin track driven headlessly through
WorkflowOrchestrator via src.core.pipeline_runner, instead of one call per
stage as src/synth_platform/interfaces/streamlit/pages/database_twin.py does from Streamlit session state.

Also exercises the CanonicalDomainModel wiring: the sample fixture's
"customers" table is covered by build_banking_domain_model(), while "orders"
is not, so domain_mapping.json should report exactly one unmapped table.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.domain_model import build_banking_domain_model
from synth_platform.engine.common.database.core.pipeline_runner import DatabaseTwinPipelineConfig, run_database_twin_pipeline
from synth_platform.engine.common.database.core.run_manifest import RunManifest


def test_database_twin_pipeline_runs_all_stages_unattended(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    config = DatabaseTwinPipelineConfig(
        dataset_id="ds-unattended",
        artifact_version="1.0.0",
        config_path="config/project.yaml",
        metadata_dir=tmp_path / "metadata",
        target_db_path=tmp_path / "database_b.db",
        row_counts_by_table={"customers": 10, "orders": 25},
        sample_limit=100,
        domain_model=build_banking_domain_model(),
    )

    context = run_database_twin_pipeline(adapter, manifest, config)

    manifest_on_disk = json.loads((manifest.run_dir / "run_manifest.json").read_text())
    stage_names = [s["stage_name"] for s in manifest_on_disk["stages"]]
    assert stage_names == [
        "discovery",
        "profiling",
        "inference",
        "contract_approval",
        "cleaning",
        "training",
        "artifact_export",
        "relational_generation",
        "qa_validation",
        "target_write",
    ]
    assert all(s["status"] == "success" for s in manifest_on_disk["stages"])

    assert context.artifacts["qa_report"]["hard_checks_passed"] is True

    domain_mapping = context.artifacts["domain_mapping"]
    assert domain_mapping["tables"]["customers"] == "Customer"
    assert domain_mapping["unmapped_tables"] == ["orders"]

    conn = sqlite3.connect(str(config.target_db_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 10
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 25
    finally:
        conn.close()


def test_database_twin_pipeline_without_domain_model_skips_mapping(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    config = DatabaseTwinPipelineConfig(
        dataset_id="ds-no-domain",
        artifact_version="1.0.0",
        config_path="config/project.yaml",
        metadata_dir=tmp_path / "metadata",
        target_db_path=tmp_path / "database_b.db",
        row_counts_by_table={"customers": 5, "orders": 5},
        sample_limit=100,
    )

    context = run_database_twin_pipeline(adapter, manifest, config)

    assert "domain_mapping" not in context.artifacts
    assert not (manifest.run_dir / "domain_mapping.json").exists()
    assert context.artifacts["qa_report"]["hard_checks_passed"] is True
