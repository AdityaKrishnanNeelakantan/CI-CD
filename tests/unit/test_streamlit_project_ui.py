from __future__ import annotations

from pathlib import Path

from synth_platform.infrastructure.persistence.platform_db import PlatformDB
from synth_platform.interfaces.streamlit.project_ui import (
    DEFAULT_SETTINGS,
    load_project_summaries,
    project_run_rows,
    read_product_settings,
    write_product_settings,
)


def test_project_summaries_join_projects_to_latest_run_and_real_record_counts(tmp_path: Path) -> None:
    db = PlatformDB(tmp_path / "platform.db")
    project = db.create_project(name="Orders Twin", workflow_type="schema")
    db.record_run(
        project_id=project.id,
        workflow_type="schema",
        status="failed",
        validation_status="FAIL",
        validation_passed=False,
        output_id="old",
        metadata={"row_counts": {"orders": 3}},
    )
    latest = db.record_run(
        project_id=project.id,
        workflow_type="schema",
        status="completed",
        validation_status="PASS",
        validation_passed=True,
        output_id="new",
        metadata={"row_counts": {"orders": 4, "customers": 2}},
    )
    db.record_transfer_attempt(
        workflow="schema",
        output_id="new",
        allowed=True,
        validation_status="PASS",
        reason="ready",
    )

    [summary] = load_project_summaries(db)

    assert summary.name == "Orders Twin"
    assert summary.workflow_label == "Schema"
    assert summary.records == "6 total"
    assert summary.validation == "Passed"
    assert summary.transfer == "Allowed"
    assert summary.latest_run_id == latest.id
    assert summary.latest_result_id is None


def test_project_filter_and_rename_use_platform_db(tmp_path: Path) -> None:
    db = PlatformDB(tmp_path / "platform.db")
    schema_project = db.create_project(name="Before", workflow_type="schema")
    db.create_project(name="Database", workflow_type="database")

    db.update_project(schema_project.id, name="After")
    summaries = load_project_summaries(db, workflow_type="schema")

    assert [summary.name for summary in summaries] == ["After"]


def test_project_run_rows_include_validation_transfer_and_untracked_records(tmp_path: Path) -> None:
    db = PlatformDB(tmp_path / "platform.db")
    project = db.create_project(name="Document", workflow_type="pdf")
    run = db.record_run(
        project_id=project.id,
        workflow_type="pdf",
        status="completed",
        validation_status="PASS",
        validation_passed=True,
        output_id="synthetic_twin.pdf",
    )
    db.record_transfer_attempt(
        workflow="pdf",
        output_id="synthetic_twin.pdf",
        allowed=False,
        validation_status="FAIL",
        reason="blocked",
    )

    [row] = project_run_rows(db, project.id)

    assert row["Run"] == run.id
    assert row["Validation"] == "Passed"
    assert row["Transfer"] == "Blocked"
    assert row["Records"] == "-"


def test_product_settings_read_defaults_and_persist_supported_keys(tmp_path: Path) -> None:
    db = PlatformDB(tmp_path / "platform.db")

    assert read_product_settings(db) == DEFAULT_SETTINGS

    write_product_settings(
        db,
        {
            "generation_mode": "source_driven",
            "default_record_count": 250,
            "privacy_level": "strict",
            "default_output_format": "parquet",
        },
    )

    assert read_product_settings(db) == {
        "generation_mode": "source_driven",
        "default_record_count": 250,
        "privacy_level": "strict",
        "default_output_format": "parquet",
    }
