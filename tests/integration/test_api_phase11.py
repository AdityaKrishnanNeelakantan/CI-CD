from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synth_platform.interfaces.api.app import app


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SP_API_STATE_DIR", str(tmp_path / "api_state"))
    monkeypatch.setenv("SP_PLATFORM_DB_PATH", str(tmp_path / "platform.db"))
    return TestClient(app)


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()["data"]
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish")


def _assert_job_contract(job: dict, *, workflow_type: str, session_id: str) -> None:
    assert job["job_id"] == job["id"]
    assert job["session_id"] == session_id
    assert job["workflow_type"] == workflow_type
    assert job["status"] in {"queued", "running", "succeeded", "failed"}
    assert isinstance(job["stage"], str)
    assert isinstance(job["percent"], float)
    assert isinstance(job["message"], str)
    assert "error" in job
    assert "result_id" in job


def test_database_session_and_source_validation(api_client: TestClient) -> None:
    created = api_client.post("/api/database/sessions", json={"intent": "Database API test"})
    assert created.status_code == 201
    session = created.json()["data"]
    assert session["workflow"] == "database_twin"
    assert session["workflow_type"] == "database_twin"

    unsupported = api_client.post(
        f"/api/database/sessions/{session['id']}/source",
        data={"source_type": "postgresql"},
        files={"file": ("source.db", b"not sqlite", "application/octet-stream")},
    )
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_source_type"

    empty = api_client.post(
        f"/api/database/sessions/{session['id']}/source",
        data={"source_type": "sqlite"},
        files={"file": ("source.db", b"", "application/octet-stream")},
    )
    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "empty_upload"


def test_database_generation_job_contract_on_pipeline_failure(api_client: TestClient, tmp_path: Path) -> None:
    db_path = tmp_path / "tiny.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE customer (id INTEGER PRIMARY KEY, segment TEXT)")
    conn.executemany("INSERT INTO customer VALUES (?, ?)", [(1, "retail"), (2, "smb"), (3, "corp")])
    conn.commit()
    conn.close()

    created = api_client.post("/api/database/sessions", json={"intent": "Database job contract"})
    session_id = created.json()["data"]["id"]
    upload = api_client.post(
        f"/api/database/sessions/{session_id}/source",
        data={"source_type": "sqlite"},
        files={"file": ("tiny.db", db_path.read_bytes(), "application/octet-stream")},
    )
    assert upload.status_code == 200
    configured = api_client.post(
        f"/api/database/sessions/{session_id}/configure",
        json={"row_count": 3, "sample_limit": 3, "seed": 3},
    )
    assert configured.status_code == 200

    started = api_client.post(f"/api/database/sessions/{session_id}/generate")
    assert started.status_code == 202
    initial = started.json()["data"]["job"]
    _assert_job_contract(initial, workflow_type="database_twin", session_id=session_id)
    final = _wait_for_job(api_client, initial["id"])
    _assert_job_contract(final, workflow_type="database_twin", session_id=session_id)
    assert final["status"] in {"succeeded", "failed"}
    if final["status"] == "succeeded":
        result = api_client.get(f"/api/results/{final['result_id']}")
        assert result.status_code == 200
        assert result.json()["data"]["workflow_type"] == "database_twin"
    else:
        assert final["error"]["code"] == "database_generation_failed"


def test_document_session_and_pdf_upload_validation(api_client: TestClient) -> None:
    created = api_client.post("/api/document/sessions", json={"intent": "Document API test"})
    assert created.status_code == 201
    session = created.json()["data"]
    assert session["workflow"] == "pdf_twin"
    assert session["workflow_type"] == "pdf_twin"

    unsupported = api_client.post(
        f"/api/document/sessions/{session['id']}/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_file_type"

    bad_pdf = api_client.post(
        f"/api/document/sessions/{session['id']}/upload",
        files={"file": ("bad.pdf", b"not a pdf", "application/pdf")},
    )
    assert bad_pdf.status_code == 400
    assert bad_pdf.json()["error"]["code"] == "document_validation_error"


def test_document_generation_job_contract_on_invalid_pdf(api_client: TestClient) -> None:
    created = api_client.post("/api/document/sessions", json={"intent": "Document job contract"})
    session_id = created.json()["data"]["id"]
    upload = api_client.post(
        f"/api/document/sessions/{session_id}/upload",
        files={"file": ("emptyish.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")},
    )
    assert upload.status_code == 200
    started = api_client.post(f"/api/document/sessions/{session_id}/generate")
    assert started.status_code == 202
    initial = started.json()["data"]["job"]
    _assert_job_contract(initial, workflow_type="pdf_twin", session_id=session_id)
    final = _wait_for_job(api_client, initial["id"])
    _assert_job_contract(final, workflow_type="pdf_twin", session_id=session_id)
    assert final["status"] == "failed"
    assert final["stage"] == "failed"
    assert final["error"]["code"] == "document_generation_failed"


def test_interaction_upload_generate_job_and_result_bundle(api_client: TestClient) -> None:
    created = api_client.post("/api/interaction/sessions", json={"intent": "Interaction API test"})
    assert created.status_code == 201
    session_id = created.json()["data"]["id"]

    invalid = api_client.post(
        f"/api/interaction/sessions/{session_id}/upload",
        files={"file": ("bad.csv", b"agent,customer", "text/csv")},
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "unsupported_file_type"

    transcript = b"Client: My card ending in 4199 is not working.\nSupport: I can help with that."
    upload = api_client.post(
        f"/api/interaction/sessions/{session_id}/upload",
        files={"file": ("support.log", transcript, "text/plain")},
    )
    assert upload.status_code == 200
    assert upload.json()["data"]["state"]["transcript"]["turn_count"] == 2

    configured = api_client.post(
        f"/api/interaction/sessions/{session_id}/configure",
        json={
            "interaction_type": "Billing Support",
            "output_format": "Structured JSON + Synthetic Logs",
            "remove_sensitive_information": True,
            "seed": 9,
        },
    )
    assert configured.status_code == 200

    started = api_client.post(f"/api/interaction/sessions/{session_id}/generate")
    assert started.status_code == 202
    initial = started.json()["data"]["job"]
    _assert_job_contract(initial, workflow_type="interaction_twin", session_id=session_id)
    final = _wait_for_job(api_client, initial["id"])
    _assert_job_contract(final, workflow_type="interaction_twin", session_id=session_id)
    assert final["status"] == "succeeded"
    assert final["stage"] == "completed"
    assert final["result_id"]

    result = api_client.get(f"/api/results/{final['result_id']}")
    assert result.status_code == 200
    data = result.json()["data"]
    assert data["workflow_type"] == "interaction_twin"
    assert data["session_id"] == session_id
    assert data["quality_report"]["hard_checks_passed"] is True
    assert data["summary"]["synthetic_turn_count"] == 2
    roles = {artifact["role"] for artifact in data["artifacts"]}
    assert {"download", "json", "log", "validation_report", "redaction_report"} <= roles
