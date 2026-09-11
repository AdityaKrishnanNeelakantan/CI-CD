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


def _sqlite_fixture(path: Path) -> bytes:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customer (
            id INTEGER PRIMARY KEY,
            segment TEXT NOT NULL,
            age INTEGER
        );
        CREATE TABLE account (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            balance REAL,
            FOREIGN KEY(customer_id) REFERENCES customer(id)
        );
        """
    )
    conn.executemany(
        "INSERT INTO customer VALUES (?, ?, ?)",
        [(1, "retail", 34), (2, "smb", 45), (3, "corp", 52)],
    )
    conn.executemany(
        "INSERT INTO account VALUES (?, ?, ?)",
        [(1, 1, 125.25), (2, 1, 300.0), (3, 2, 50.5), (4, 3, 700.0)],
    )
    conn.commit()
    conn.close()
    return path.read_bytes()


def _create_session(client: TestClient) -> str:
    response = client.post("/api/database/sessions", json={"intent": "Database Twin regression"})
    assert response.status_code == 201
    return response.json()["data"]["id"]


def _upload_fixture(client: TestClient, session_id: str, db_bytes: bytes) -> dict:
    response = client.post(
        f"/api/database/sessions/{session_id}/source",
        data={"source_type": "sqlite"},
        files={"file": ("fixture.sqlite", db_bytes, "application/x-sqlite3")},
    )
    assert response.status_code == 200
    return response.json()["data"]


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()["data"]
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish")


def test_sqlite_upload_discovers_real_tables_counts_and_columns(api_client: TestClient, tmp_path: Path) -> None:
    session_id = _create_session(api_client)
    session = _upload_fixture(api_client, session_id, _sqlite_fixture(tmp_path / "fixture.sqlite"))

    source = session["state"]["source"]
    assert source["filename"] == "fixture.sqlite"
    assert source["source_type"] == "sqlite"
    assert source["table_names"] == ["account", "customer"]
    assert source["source_rows_by_table"] == {"account": 4, "customer": 3}
    assert source["total_rows"] == 7
    assert {table["name"]: table["column_count"] for table in source["tables"]} == {
        "account": 3,
        "customer": 3,
    }
    assert session["state"]["discovery"]["tables"]["account"]["estimated_row_count"] == 4


def test_schema_twin_rejects_sqlite_uploads(api_client: TestClient, tmp_path: Path) -> None:
    created = api_client.post("/api/schema/sessions", json={"intent": "wrong workflow"})
    assert created.status_code == 201
    session_id = created.json()["data"]["id"]

    response = api_client.post(
        f"/api/schema/sessions/{session_id}/schema-file",
        files={"file": ("fixture.sqlite", _sqlite_fixture(tmp_path / "fixture.sqlite"), "application/x-sqlite3")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "wrong_workflow"


def test_configure_preserve_and_target_counts_use_discovered_tables(api_client: TestClient, tmp_path: Path) -> None:
    session_id = _create_session(api_client)
    _upload_fixture(api_client, session_id, _sqlite_fixture(tmp_path / "fixture.sqlite"))

    defaulted = api_client.post(
        f"/api/database/sessions/{session_id}/configure",
        json={"sample_limit": 3, "seed": 5},
    )
    assert defaulted.status_code == 200
    config = defaulted.json()["data"]["state"]["config"]
    assert config["count_mode"] == "preserve_source_counts"
    assert config["row_counts_by_table"] == {"account": 4, "customer": 3}

    preserve = api_client.post(
        f"/api/database/sessions/{session_id}/configure",
        json={"preserve_source_counts": True, "sample_limit": 3, "seed": 5},
    )
    assert preserve.status_code == 200
    config = preserve.json()["data"]["state"]["config"]
    assert config["count_mode"] == "preserve_source_counts"
    assert config["row_counts_by_table"] == {"account": 4, "customer": 3}

    target = api_client.post(
        f"/api/database/sessions/{session_id}/configure",
        json={"target_record_count": 2, "sample_limit": 3, "seed": 5},
    )
    assert target.status_code == 200
    config = target.json()["data"]["state"]["config"]
    assert config["count_mode"] == "target_record_count"
    assert config["row_counts_by_table"] == {"account": 2, "customer": 2}


def test_database_generation_result_includes_source_and_generated_count_metadata(
    api_client: TestClient, tmp_path: Path
) -> None:
    session_id = _create_session(api_client)
    _upload_fixture(api_client, session_id, _sqlite_fixture(tmp_path / "fixture.sqlite"))
    configured = api_client.post(
        f"/api/database/sessions/{session_id}/configure",
        json={"preserve_source_counts": True, "sample_limit": 4, "seed": 5},
    )
    assert configured.status_code == 200

    started = api_client.post(f"/api/database/sessions/{session_id}/generate")
    assert started.status_code == 202
    final = _wait_for_job(api_client, started.json()["data"]["job"]["id"])
    assert final["status"] == "succeeded", final.get("error")

    result = api_client.get(f"/api/results/{final['result_id']}")
    assert result.status_code == 200
    bundle = result.json()["data"]
    projects_before_save = api_client.get("/api/projects")
    assert projects_before_save.status_code == 200
    assert projects_before_save.json()["data"]["projects"] == []
    assert bundle["workflow_type"] == "database_twin"
    assert bundle["summary"]["source_rows_by_table"] == {"account": 4, "customer": 3}
    assert bundle["summary"]["generated_rows_by_table"] == {"account": 4, "customer": 3}
    assert bundle["summary"]["row_counts"] == {"account": 4, "customer": 3}
    assert bundle["summary"]["source_table_count"] == 2
    assert bundle["summary"]["generated_table_count"] == 2
    assert bundle["summary"]["count_mode"] == "preserve_source_counts"
    assert bundle["summary"]["name"] == "database_twin"
    assert bundle["summary"]["generation_mode"] == "database_source_driven"
    assert bundle["quality_report"]["source_vs_generated_row_counts"]["matches_source"] is True
    assert bundle["quality_report"]["source_vs_generated_row_counts"]["tables"]["account"] == {
        "source_rows": 4,
        "generated_rows": 4,
        "matches": True,
    }
    assert bundle["metadata"]["source"]["filename"] == "fixture.sqlite"
    assert all("schema_outputs" not in artifact["path"] for artifact in bundle["artifacts"])
    assert any("database_outputs" in artifact["path"] for artifact in bundle["artifacts"])


def test_database_generation_without_configure_preserves_source_counts(
    api_client: TestClient, tmp_path: Path
) -> None:
    session_id = _create_session(api_client)
    _upload_fixture(api_client, session_id, _sqlite_fixture(tmp_path / "fixture.sqlite"))

    started = api_client.post(f"/api/database/sessions/{session_id}/generate")
    assert started.status_code == 202
    final = _wait_for_job(api_client, started.json()["data"]["job"]["id"])
    assert final["status"] == "succeeded", final.get("error")

    result = api_client.get(f"/api/results/{final['result_id']}")
    assert result.status_code == 200
    bundle = result.json()["data"]
    assert bundle["summary"]["count_mode"] == "preserve_source_counts"
    assert bundle["summary"]["generated_rows_by_table"] == {"account": 4, "customer": 3}
    assert bundle["metadata"]["config"]["row_counts_by_table"] == {"account": 4, "customer": 3}

    session = api_client.get(f"/api/database/sessions/{session_id}").json()["data"]
    assert session["state"]["config"]["count_mode"] == "preserve_source_counts"
