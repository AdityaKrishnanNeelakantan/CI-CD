from __future__ import annotations

import sqlite3
from pathlib import Path

from synth_platform.application.services.transfer_service import TransferService
from synth_platform.infrastructure.persistence.platform_db import PlatformDB


def test_platform_db_initializes_schema_on_fresh_file(tmp_path: Path) -> None:
    db_path = tmp_path / "platform.db"
    PlatformDB(db_path)

    conn = sqlite3.connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert {"projects", "runs", "settings"}.issubset(tables)


def test_create_project_record_runs_for_each_workflow_and_update_transfer(tmp_path: Path) -> None:
    db = PlatformDB(tmp_path / "platform.db")
    for workflow in ("schema", "database", "pdf", "interaction"):
        project = db.create_project(name=f"{workflow} project", workflow_type=workflow)
        run = db.record_run(
            project_id=project.id,
            workflow_type=workflow,
            status="completed",
            validation_status="PASS",
            validation_passed=True,
            output_id=f"{workflow}.zip",
        )

        assert run.workflow_type == workflow
        assert run.transfer_status == "not_attempted"

    updated = db.record_transfer_attempt(
        workflow="schema_twin",
        output_id="schema.zip",
        allowed=True,
        validation_status="PASS",
        reason="schema validation and export readiness passed",
        metadata={"file_name": "schema.zip"},
    )

    assert updated is not None
    assert updated.transfer_status == "success"
    assert updated.transfer_allowed is True


def test_settings_round_trip_after_write(tmp_path: Path) -> None:
    db = PlatformDB(tmp_path / "platform.db")

    db.write_settings(
        {
            "generation_mode": "schema_driven",
            "default_record_count": 250,
            "privacy_level": "restricted",
            "default_output_format": "csv",
        }
    )

    assert db.read_settings() == {
        "default_output_format": "csv",
        "default_record_count": 250,
        "generation_mode": "schema_driven",
        "privacy_level": "restricted",
    }


def test_record_run_autogenerates_timestamped_project_name(tmp_path: Path) -> None:
    db = PlatformDB(tmp_path / "platform.db")

    run = db.record_run(workflow_type="schema", validation_status="PASS", validation_passed=True)
    project = db.get_project(run.project_id)

    assert project.name.startswith("Schema Twin - ")
    assert project.name != "Schema Twin Project"


def test_transfer_service_can_update_platform_run_without_changing_gate(tmp_path: Path) -> None:
    db = PlatformDB(tmp_path / "platform.db")
    db.record_run(
        workflow_type="schema",
        project_name="schema",
        status="completed",
        validation_status="PASS",
        validation_passed=True,
        output_id="schema.zip",
    )

    transfer = TransferService(transfer_recorder=db).downloadable_bytes(
        workflow="schema_twin",
        output_id="schema.zip",
        validation_report={"passed": True, "export_ready": True},
        data=b"payload",
    )

    assert transfer.data == b"payload"
    [run] = db.list_runs(workflow_type="schema")
    assert run.transfer_status == "success"
    assert run.transfer_allowed is True
