from __future__ import annotations

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


def test_project_and_run_detail_include_result_links(api_client: TestClient) -> None:
    store = ApiStateStore()
    bundle = store.create_result_bundle(
        session_id="session-project-detail",
        workflow_type="schema_twin",
        preview={"tables": {"users": [{"id": 1}]}},
        quality_report={"passed": True, "status": "passed"},
        summary={"row_counts": {"users": 3}},
        artifacts=[],
        metadata={"job_id": "job-project-detail"},
    )

    saved = api_client.post(f"/api/results/{bundle['id']}/save")
    assert saved.status_code == 201
    saved_data = saved.json()["data"]

    project = api_client.get(f"/api/projects/{saved_data['project_id']}")
    assert project.status_code == 200
    project_data = project.json()["data"]["project"]
    assert project_data["project_id"] == saved_data["project_id"]
    assert project_data["latest_result_id"] == bundle["id"]
    assert project_data["runs"][0]["run_id"] == saved_data["run_id"]

    runs = api_client.get(f"/api/projects/{saved_data['project_id']}/runs")
    assert runs.status_code == 200
    assert runs.json()["data"]["runs"][0]["result_id"] == bundle["id"]

    run = api_client.get(f"/api/projects/{saved_data['project_id']}/runs/{saved_data['run_id']}")
    assert run.status_code == 200
    run_data = run.json()["data"]["run"]
    assert run_data["result_id"] == bundle["id"]
    assert run_data["job_id"] == "job-project-detail"
    assert run_data["quality_report"]["passed"] is True
    assert run_data["summary"]["row_counts"] == {"users": 3}


def test_saved_schema_project_records_show_total_and_per_table_counts(api_client: TestClient) -> None:
    store = ApiStateStore()
    bundle = store.create_result_bundle(
        session_id="session-multi-table-records",
        workflow_type="schema_twin",
        preview={
            "row_counts": {"accounts": 1000, "branches": 1000, "cards": 1000},
            "tables": {
                "accounts": [{"id": 1}],
                "branches": [{"id": 1}],
                "cards": [{"id": 1}],
            },
        },
        quality_report={"passed": True, "status": "passed"},
        summary={
            "row_counts": {"accounts": 1000, "branches": 1000, "cards": 1000},
            "requested_row_count": 1000,
            "row_count_mode": "per_table",
        },
        artifacts=[],
    )

    saved = api_client.post(f"/api/results/{bundle['id']}/save")
    assert saved.status_code == 201

    projects = api_client.get("/api/projects")
    assert projects.status_code == 200
    rows = projects.json()["data"]["projects"]
    saved_row = next(row for row in rows if row["latest_result_id"] == bundle["id"])
    assert saved_row["records"] == "3,000 total / 1,000 per table"


def test_artifact_paths_and_filenames_are_not_exposed_as_result_page_ids(api_client: TestClient) -> None:
    db = PlatformDB()
    path_project = db.create_project(name="Path output", workflow_type="schema")
    path_run = db.record_run(
        workflow_type="schema",
        project_id=path_project.id,
        output_id="/tmp/schema-output",
    )
    file_project = db.create_project(name="File output", workflow_type="database")
    file_run = db.record_run(
        workflow_type="database",
        project_id=file_project.id,
        output_id="database.db",
    )

    projects = api_client.get("/api/projects").json()["data"]["projects"]
    by_id = {row["id"]: row for row in projects}
    assert by_id[path_project.id]["latest_result_id"] is None
    assert by_id[file_project.id]["latest_result_id"] is None

    path_detail = api_client.get(f"/api/projects/{path_project.id}").json()["data"]["project"]
    file_detail = api_client.get(f"/api/projects/{file_project.id}").json()["data"]["project"]
    assert path_detail["runs"][0]["run_id"] == path_run.id
    assert path_detail["runs"][0]["result_id"] is None
    assert file_detail["runs"][0]["run_id"] == file_run.id
    assert file_detail["runs"][0]["result_id"] is None


def test_templates_api_and_schema_template_attach(api_client: TestClient) -> None:
    templates = api_client.get("/api/templates")
    assert templates.status_code == 200
    rows = templates.json()["data"]["templates"]
    assert any(row["template_id"] == "ecommerce" and row["can_generate"] is True for row in rows)
    assert any(row["status"] == "coming_soon" and row["can_generate"] is False for row in rows)

    detail = api_client.get("/api/templates/ecommerce")
    assert detail.status_code == 200
    template = detail.json()["data"]["template"]
    assert template["status"] == "available"
    assert template["fields"]
    assert template["schema_preview"]["table_count"] >= 1

    created = api_client.post("/api/schema/sessions", json={"intent": "Template test"})
    session_id = created.json()["data"]["id"]
    attached = api_client.post(f"/api/schema/sessions/{session_id}/template", json={"template_id": "ecommerce"})
    assert attached.status_code == 200
    attached_data = attached.json()["data"]
    assert attached_data["summary"]["name"] == "E-commerce Platform"
    assert attached_data["columns"]
    assert attached_data["session"]["state"]["template"]["template_id"] == "ecommerce"

    coming_soon = api_client.post(
        f"/api/schema/sessions/{session_id}/template",
        json={"template_id": "support-transcript"},
    )
    assert coming_soon.status_code == 409
    assert coming_soon.json()["error"]["code"] == "template_unavailable"


def test_run_detail_404s_when_project_does_not_match(api_client: TestClient) -> None:
    db = PlatformDB()
    project = db.create_project(name="One", workflow_type="schema")
    other_project = db.create_project(name="Two", workflow_type="schema")
    run = db.record_run(workflow_type="schema", project_id=project.id, output_id="result-x")

    response = api_client.get(f"/api/projects/{other_project.id}/runs/{run.id}")
    assert response.status_code == 404
