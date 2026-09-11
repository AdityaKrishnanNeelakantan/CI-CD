from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synth_platform.infrastructure.persistence.platform_db import PlatformDB
from synth_platform.interfaces.api.app import app
from synth_platform.interfaces.api.store import ApiStateStore


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


def test_health_workflow_sessions_jobs_settings_and_projects(api_client: TestClient) -> None:
    health = api_client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["data"]["feature_flags"]["auth_enabled"] is False

    created = api_client.post(
        "/api/workflows/schema/sessions",
        json={"intent": "Development", "metadata": {"source": "test"}},
    )
    assert created.status_code == 201
    session = created.json()["data"]
    assert session["workflow"] == "schema_twin"

    fetched = api_client.get(f"/api/workflows/schema/sessions/{session['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["data"]["state"]["metadata"] == {"source": "test"}

    job = api_client.post("/api/jobs", json={"kind": "manual", "payload": {"check": True}})
    assert job.status_code == 201
    job_id = job.json()["data"]["id"]
    assert api_client.get(f"/api/jobs/{job_id}").json()["data"]["status"] == "queued"

    updated = api_client.put(
        "/api/settings",
        json={
            "generation_mode": "schema_driven",
            "default_record_count": 12,
            "privacy_level": "standard",
            "default_output_format": "csv",
        },
    )
    assert updated.status_code == 200
    assert api_client.get("/api/settings").json()["data"]["default_record_count"] == 12

    db = PlatformDB()
    project = db.create_project(name="API project", workflow_type="schema")
    db.record_run(
        workflow_type="schema",
        project_id=project.id,
        status="completed",
        validation_status="PASS",
        validation_passed=True,
        output_id="api-project-output",
        metadata={"row_counts": {"users": 12}},
    )
    projects = api_client.get("/api/projects")
    assert projects.status_code == 200
    assert any(row["id"] == project.id for row in projects.json()["data"]["projects"])
    runs = api_client.get(f"/api/projects/{project.id}/runs")
    assert runs.status_code == 200
    assert runs.json()["data"]["runs"][0]["Output"] == "api-project-output"


def test_schema_twin_generate_then_download_flow(api_client: TestClient) -> None:
    created = api_client.post("/api/schema/sessions", json={"intent": "QA / automated tests"})
    assert created.status_code == 201
    session_id = created.json()["data"]["id"]

    fixture = Path("tests/fixtures/schema/schema_twin_minimal.json")
    upload = api_client.post(
        f"/api/schema/sessions/{session_id}/schema-file",
        files={"file": (fixture.name, fixture.read_bytes(), "application/json")},
    )
    assert upload.status_code == 200
    upload_data = upload.json()["data"]
    assert upload_data["summary"]["table_count"] == 2
    assert {row["table"] for row in upload_data["columns"]} == {"users", "orders"}

    generate = api_client.post(
        f"/api/schema/sessions/{session_id}/generate",
        json={"row_count": 5, "preview_rows": 3, "seed": 7, "export_format": "csv"},
    )
    assert generate.status_code == 202
    job = _wait_for_job(api_client, generate.json()["data"]["job"]["id"])
    assert job["status"] == "succeeded"

    preview = api_client.get(f"/api/schema/sessions/{session_id}/preview")
    assert preview.status_code == 200
    assert set(preview.json()["data"]["tables"]) == {"users", "orders"}
    assert preview.json()["data"]["row_counts"] == {"users": 5, "orders": 5}

    validation = api_client.get(f"/api/schema/sessions/{session_id}/validation")
    assert validation.status_code == 200
    assert validation.json()["data"]["highlights"]["hard_checks_passed"] is True

    ticket = api_client.post(f"/api/schema/sessions/{session_id}/download")
    assert ticket.status_code == 201
    download = api_client.get(ticket.json()["data"]["url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert download.content.startswith(b"PK")

    runs = PlatformDB().list_runs(workflow_type="schema")
    assert any(run.transfer_status == "success" and run.transfer_allowed is True for run in runs)


def test_schema_upload_rejects_invalid_payload(api_client: TestClient) -> None:
    created = api_client.post("/api/schema/sessions", json={"intent": "bad upload"})
    session_id = created.json()["data"]["id"]
    upload = api_client.post(
        f"/api/schema/sessions/{session_id}/schema-file",
        files={"file": ("bad.json", json.dumps([1, 2, 3]).encode(), "application/json")},
    )
    assert upload.status_code == 400
    assert upload.json()["error"]["code"] == "schema_parse_error"


def test_download_endpoint_blocks_and_records_failed_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SP_API_STATE_DIR", str(tmp_path / "api_state"))
    monkeypatch.setenv("SP_PLATFORM_DB_PATH", str(tmp_path / "platform.db"))
    client = TestClient(app)
    store = ApiStateStore()
    db = PlatformDB()
    db.record_run(
        workflow_type="schema",
        status="failed",
        validation_status="FAIL",
        validation_passed=False,
        output_id="blocked.zip",
        metadata={"row_counts": {"users": 1}},
    )
    zip_path = store.write_blob("blocked.zip", b"not a valid release")
    ticket = store.create_download(
        {
            "workflow": "schema_twin",
            "path": str(zip_path),
            "filename": "blocked.zip",
            "media_type": "application/zip",
            "output_id": "blocked.zip",
            "validation_report": {"passed": False, "export_ready": False},
            "metadata": {"file_name": "blocked.zip", "intent": "negative test"},
        }
    )

    response = client.get(f"/api/downloads/{ticket['id']}")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "transfer_blocked"

    run = db.list_runs(workflow_type="schema")[0]
    assert run.transfer_status == "blocked"
    assert run.transfer_allowed is False
