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
