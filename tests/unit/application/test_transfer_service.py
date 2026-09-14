from __future__ import annotations

import json
import sqlite3

import pytest

from synth_platform.application.services.transfer_service import TransferService
from synth_platform.application.workflows.schema_twin import SchemaModeResult
from synth_platform.application.orchestration.schema.result import PipelineResult
from synth_platform.domain.validation.models import CheckResult, Status, ValidationReport
from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table
from synth_platform.engine.validation.schema.audit import AuditLogger
from synth_platform.errors import TransferBlockedError


def _audit_events(db_path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT operation, status, details FROM audit_log ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [(operation, status, json.loads(details)) for operation, status, details in rows]


def _service(tmp_path):
    audit_db = tmp_path / "audit.db"
    return TransferService(AuditLogger(str(audit_db))), audit_db


def _minimal_schema(name: str) -> SchemaConfig:
    return SchemaConfig(
        name=name,
        tables=[Table(name="records", row_count=1)],
        columns={"records": [Column(name="id", type="int")]},
    )


@pytest.mark.parametrize(
    ("workflow", "failed_report"),
    [
        ("schema_twin", {"passed": False, "export_ready": True}),
        ("pdf_twin", {"hard_checks_passed": False}),
        ("database_twin", {"hard_checks_passed": False, "release": {"decision": "BLOCKED", "reason": "QA failed"}}),
        ("interaction_twin", {"hard_checks_passed": False, "export_ready": True}),
    ],
)
def test_transfer_blocks_failed_and_absent_validation_for_each_workflow(tmp_path, workflow, failed_report):
    service, audit_db = _service(tmp_path)

    with pytest.raises(TransferBlockedError):
        service.downloadable_bytes(
            workflow=workflow,
            output_id="artifact",
            validation_report=None,
            data=b"payload",
        )

    with pytest.raises(TransferBlockedError):
        service.downloadable_bytes(
            workflow=workflow,
            output_id="artifact",
            validation_report=failed_report,
            data=b"payload",
        )

    events = _audit_events(audit_db)
    assert [event[0] for event in events] == ["transfer_attempt", "transfer_attempt"]
    assert [event[1] for event in events] == ["blocked", "blocked"]


@pytest.mark.parametrize(
    ("workflow", "passing_report"),
    [
        ("schema_twin", {"passed": True, "export_ready": True}),
        ("pdf_twin", {"hard_checks_passed": True}),
        ("database_twin", {"hard_checks_passed": True, "release": {"decision": "PASS", "reason": "all hard checks passed"}}),
        ("interaction_twin", {"hard_checks_passed": True, "export_ready": True}),
    ],
)
def test_transfer_allows_passing_validation_and_records_audit(tmp_path, workflow, passing_report):
    service, audit_db = _service(tmp_path)

    transfer = service.downloadable_bytes(
        workflow=workflow,
        output_id="artifact",
        validation_report=passing_report,
        data=b"payload",
        metadata={"file_name": "artifact.bin"},
    )

    assert transfer.data == b"payload"
    events = _audit_events(audit_db)
    assert len(events) == 1
    operation, status, details = events[0]
    assert operation == "transfer_attempt"
    assert status == "success"
    assert details["workflow"] == workflow
    assert details["validation_status"] == "PASS"
    assert details["metadata"]["file_name"] == "artifact.bin"


def test_transfer_reuses_canonical_release_gate_for_validation_report(tmp_path):
    service, audit_db = _service(tmp_path)
    report = ValidationReport(
        checks=[
            CheckResult(
                name="critical_metric",
                dimension="structural",
                status=Status.PASS,
                ran=True,
                required=True,
            )
        ]
    )

    transfer = service.downloadable_bytes(
        workflow="schema_twin",
        output_id="canonical",
        validation_report=report,
        data=b"payload",
    )

    assert transfer.approval.validation_status == "PASS"
    assert _audit_events(audit_db)[0][1] == "success"


def test_transfer_blocks_malformed_database_qa_without_release_decision(tmp_path):
    service, _audit_db = _service(tmp_path)

    with pytest.raises(TransferBlockedError, match="release decision"):
        service.downloadable_bytes(
            workflow="database_twin",
            output_id="database_b.db",
            validation_report={"hard_checks_passed": True},
            data=b"sqlite",
        )


def test_schema_transfer_blocks_when_schema_mode_validation_status_is_unknown(tmp_path):
    service, _audit_db = _service(tmp_path)
    result = SchemaModeResult(
        schema=_minimal_schema("unknown_validation"),
        pipeline=PipelineResult(
            generation_mode="schema_driven",
            validation_report={"summary": "validation report missing pass/fail fields"},
        ),
    )

    assert result.hard_checks_passed is False
    with pytest.raises(TransferBlockedError, match="schema hard validation did not pass"):
        service.downloadable_bytes(
            workflow="schema_twin",
            output_id="schema.zip",
            validation_report={"passed": result.hard_checks_passed, "export_ready": True},
            data=b"zip",
        )
