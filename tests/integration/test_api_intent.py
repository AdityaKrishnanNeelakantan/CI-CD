from __future__ import annotations

from fastapi.testclient import TestClient

from synth_platform.interfaces.api.app import app


def test_intent_routes_pdf_to_document_twin() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/intent/workflow",
        json={"message": "create synthetic pdf from this", "attachments": [{"filename": "report.pdf", "extension": ".pdf"}]},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["workflow_type"] == "document_twin"
    assert data["suggested_route"] == "/document"
    assert data["confidence"] >= 0.8


def test_intent_routes_sqlite_to_database_twin() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/intent/workflow",
        json={"message": "Create a synthetic twin of this database", "attachments": [{"filename": "source.sqlite"}]},
    )

    data = response.json()["data"]
    assert data["workflow_type"] == "database_twin"
    assert data["next_action"] == "configure_required"


def test_intent_routes_txt_and_log_to_interaction_twin() -> None:
    client = TestClient(app)

    for filename in ("support.txt", "support.log"):
        response = client.post(
            "/api/intent/workflow",
            json={"message": "generate synthetic support conversations", "attachments": [{"filename": filename}]},
        )
        data = response.json()["data"]
        assert data["workflow_type"] == "interaction_twin"
        assert data["suggested_route"] == "/interaction"


def test_intent_routes_sql_and_json_to_schema_twin() -> None:
    client = TestClient(app)

    for filename in ("schema.sql", "schema.json"):
        response = client.post(
            "/api/intent/workflow",
            json={"message": "Create synthetic patient data from this schema", "attachments": [{"filename": filename}]},
        )
        data = response.json()["data"]
        assert data["workflow_type"] == "schema_twin"
        assert data["suggested_route"] == "/schema"


def test_prompt_only_customer_chats_routes_to_interaction_twin() -> None:
    client = TestClient(app)
    response = client.post("/api/intent/workflow", json={"message": "generate synthetic customer chats"})

    data = response.json()["data"]
    assert data["workflow_type"] == "interaction_twin"
    assert data["next_action"] == "upload_required"


def test_prompt_only_pdf_report_routes_to_document_twin() -> None:
    client = TestClient(app)
    response = client.post("/api/intent/workflow", json={"message": "create synthetic pdf report"})

    data = response.json()["data"]
    assert data["workflow_type"] == "document_twin"
    assert data["next_action"] == "upload_required"


def test_ambiguous_csv_returns_low_confidence_alternatives() -> None:
    client = TestClient(app)
    response = client.post("/api/intent/workflow", json={"message": "use this csv", "attachments": [{"filename": "data.csv"}]})

    data = response.json()["data"]
    assert data["next_action"] == "choose_workflow"
    assert data["confidence"] < 0.55
    assert {alternative["workflow_type"] for alternative in data["alternatives"]} >= {"schema_twin", "database_twin"}


def test_unsupported_source_warns_without_claiming_support() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/intent/workflow",
        json={"message": "create a synthetic database", "attachments": [{"filename": "data.csv"}]},
    )

    data = response.json()["data"]
    assert data["next_action"] == "unsupported"
    assert any("SQLite" in warning for warning in data["warnings"])


def test_llm_unavailable_falls_back_to_rules() -> None:
    client = TestClient(app)
    response = client.post("/api/intent/workflow", json={"message": "Generate 10,000 fake insurance claim records."})

    data = response.json()["data"]
    assert data["workflow_type"] == "schema_twin"
    assert data["prefill"]["record_count"] == 10000
