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


def test_plain_data_count_prefills_record_count() -> None:
    client = TestClient(app)
    response = client.post("/api/intent/workflow", json={"message": "generate 1000 data"})

    data = response.json()["data"]
    assert data["prefill"]["record_count"] == 1000


def test_prompt_configurable_options_are_prefilled() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/intent/workflow",
        json={
            "message": "Generate 1000 data as parquet for Indian locale with strict privacy seed 77 sample 25 using OCR and LLM text"
        },
    )

    data = response.json()["data"]
    assert data["prefill"]["record_count"] == 1000
    assert data["prefill"]["output_format"] == "parquet"
    assert data["prefill"]["privacy_level"] == "strict"
    assert data["prefill"]["other"]["locale"] == "en_IN"
    assert data["prefill"]["other"]["seed"] == 77
    assert data["prefill"]["other"]["sample_limit"] == 25
    assert data["prefill"]["other"]["extraction_method"] == "ocr"
    assert data["prefill"]["other"]["llm_text_enabled"] is True


def test_prompt_redaction_options_are_prefilled() -> None:
    client = TestClient(app)
    response = client.post("/api/intent/workflow", json={"message": "Generate customer chats as logs with no redaction seed 12"})

    data = response.json()["data"]
    assert data["workflow_type"] == "interaction_twin"
    assert data["prefill"]["output_format"] == "logs"
    assert data["prefill"]["other"]["remove_sensitive_information"] is False
    assert data["prefill"]["other"]["seed"] == 12


def test_demo_placeholder_run_returns_completed_job() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/demo/placeholder-run",
        json={
            "message": "show me a demo database twin",
            "workflow_type": "database_twin",
            "intent": {"workflow_type": "database_twin", "prefill": {"record_count": 1000}},
        },
    )

    assert response.status_code == 201
    data = response.json()["data"]
    assert data["job"]["status"] == "succeeded"
    assert data["job"]["result_id"]
    result = client.get(f"/api/results/{data['job']['result_id']}").json()["data"]
    assert result["summary"]["requested_row_count"] == 1000


def test_database_sample_source_creates_readable_sqlite_session() -> None:
    client = TestClient(app)
    session_response = client.post("/api/database/sessions", json={"intent": "sample database demo"})
    session_id = session_response.json()["data"]["id"]

    response = client.post(
        f"/api/database/sessions/{session_id}/sample-source",
        json={"customer_count": 50, "seed": 42},
    )

    assert response.status_code == 200
    source = response.json()["data"]["state"]["source"]
    assert source["filename"] == "database_a.db"
    assert source["source_type"] == "sqlite"
    assert set(source["tables"]) == {"customer", "account", "transaction"}
    assert source["sample"] == {"customer_count": 50, "seed": 42}
