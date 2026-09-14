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
    manual_job = job.json()["data"]
    job_id = manual_job["id"]
    assert manual_job["job_id"] == job_id
    assert manual_job["stage"] == "queued"
    assert manual_job["percent"] == 0.0
    fetched_job = api_client.get(f"/api/jobs/{job_id}").json()["data"]
    assert fetched_job["status"] == "queued"
    assert fetched_job["message"] == "queued"

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

    renamed = api_client.patch(f"/api/projects/{project.id}", json={"name": "Renamed API project"})
    assert renamed.status_code == 200
    assert renamed.json()["data"]["name"] == "Renamed API project"
    assert PlatformDB().get_project(project.id).name == "Renamed API project"

    runs = api_client.get(f"/api/projects/{project.id}/runs")
    assert runs.status_code == 200
    assert runs.json()["data"]["runs"][0]["result_id"] is None

    empty_update = api_client.patch(f"/api/projects/{project.id}", json={})
    assert empty_update.status_code == 422
    assert empty_update.json()["error"]["code"] == "validation_error"


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
    initial_job = generate.json()["data"]["job"]
    assert initial_job["job_id"] == initial_job["id"]
    assert initial_job["session_id"] == session_id
    assert initial_job["workflow_type"] == "schema_twin"
    assert initial_job["stage"] == "queued"
    assert initial_job["percent"] == 0.0
    job = _wait_for_job(api_client, initial_job["id"])
    assert job["status"] == "succeeded"
    assert job["job_id"] == job["id"]
    assert job["session_id"] == session_id
    assert job["workflow_type"] == "schema_twin"
    assert job["stage"] == "complete"
    assert job["percent"] == 100.0
    assert job["message"] == "Generation complete"
    assert job["result_id"]
    assert job["result"] == {"session_id": session_id, "result_id": job["result_id"]}

    result_bundle = api_client.get(f"/api/results/{job['result_id']}")
    assert result_bundle.status_code == 200
    result_data = result_bundle.json()["data"]
    assert result_data["result_id"] == job["result_id"]
    assert result_data["session_id"] == session_id
    assert result_data["workflow_type"] == "schema_twin"
    assert result_data["preview"]["row_counts"]["users"] == 5
    assert result_data["preview"]["row_counts"]["orders"] > 5
    assert result_data["summary"]["row_count_mode"] == "fk_aware"
    assert result_data["quality_report"]["passed"] is True
    assert {artifact["role"] for artifact in result_data["artifacts"]} >= {"download", "table_export"}
    downloadable = [artifact for artifact in result_data["artifacts"] if artifact["downloadable"]]
    assert downloadable
    assert all(artifact["download_url"] for artifact in downloadable)

    artifact_download = api_client.get(downloadable[0]["download_url"])
    assert artifact_download.status_code == 200
    assert artifact_download.content

    projects_before_save = api_client.get("/api/projects")
    assert projects_before_save.status_code == 200
    assert projects_before_save.json()["data"]["projects"] == []

    saved = api_client.post(f"/api/results/{job['result_id']}/save")
    assert saved.status_code == 201
    saved_data = saved.json()["data"]
    assert saved_data["project_id"]
    assert saved_data["run_id"]
    assert saved_data["saved"] is True

    saved_again = api_client.post(f"/api/results/{job['result_id']}/save")
    assert saved_again.status_code == 201
    saved_again_data = saved_again.json()["data"]
    assert saved_again_data["project_id"] == saved_data["project_id"]
    assert saved_again_data["run_id"] == saved_data["run_id"]
    assert saved_again_data["saved"] is False

    preview = api_client.get(f"/api/schema/sessions/{session_id}/preview")
    assert preview.status_code == 200
    assert set(preview.json()["data"]["tables"]) == {"users", "orders"}
    assert preview.json()["data"]["row_counts"] == result_data["preview"]["row_counts"]

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


def test_result_artifact_download_returns_410_for_missing_file(api_client: TestClient, tmp_path: Path) -> None:
    store = ApiStateStore()
    bundle = store.create_result_bundle(
        session_id="session-missing-artifact",
        workflow_type="schema_twin",
        artifacts=[
            {
                "id": "missing-report",
                "name": "missing-report.json",
                "path": str(tmp_path / "missing-report.json"),
                "media_type": "application/json",
                "size": None,
                "role": "report",
                "metadata": {},
            }
        ],
    )

    result = api_client.get(f"/api/results/{bundle['id']}")
    assert result.status_code == 200
    artifact = result.json()["data"]["artifacts"][0]
    assert artifact["downloadable"] is False
    assert artifact["download_url"] is None

    response = api_client.get(f"/api/results/{bundle['id']}/artifacts/missing-report/download")
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "artifact_unavailable"


def test_schema_upload_rejects_invalid_payload(api_client: TestClient) -> None:
    created = api_client.post("/api/schema/sessions", json={"intent": "bad upload"})
    session_id = created.json()["data"]["id"]
    upload = api_client.post(
        f"/api/schema/sessions/{session_id}/schema-file",
        files={"file": ("bad.json", json.dumps([1, 2, 3]).encode(), "application/json")},
    )
    assert upload.status_code == 400
    assert upload.json()["error"]["code"] == "schema_parse_error"


def test_schema_upload_rejects_sqlite_database_files(api_client: TestClient) -> None:
    created = api_client.post("/api/schema/sessions", json={"intent": "wrong workflow"})
    session_id = created.json()["data"]["id"]
    upload = api_client.post(
        f"/api/schema/sessions/{session_id}/schema-file",
        files={"file": ("source.sqlite", b"sqlite bytes", "application/x-sqlite3")},
    )
    assert upload.status_code == 400
    assert upload.json()["error"]["code"] == "wrong_workflow"


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
