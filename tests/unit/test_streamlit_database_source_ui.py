from __future__ import annotations

from synth_platform.interfaces.streamlit.database_source_ui import (
    PostgresConnectionDetails,
    postgres_connection_config,
)
from synth_platform.infrastructure.persistence.platform_db import PlatformDB


def test_postgres_connection_config_builds_url_and_allowlists_without_persisted_metadata():
    config = postgres_connection_config(
        PostgresConnectionDetails(
            host="db.example.com",
            port=5432,
            database="app",
            username="reader@example.com",
            password="secret value",
            sslmode="require",
            allowed_schemas=("public", "analytics"),
            allowed_tables=("public.customer",),
        )
    )

    assert config["url"] == (
        "postgresql://reader%40example.com:secret%20value@db.example.com:5432/app?sslmode=require"
    )
    assert config["allowed_schemas"] == ("public", "analytics")
    assert config["allowed_tables"] == ("public.customer",)


def test_postgres_source_run_metadata_records_source_type_without_credentials(tmp_path):
    db = PlatformDB(tmp_path / "platform.db")
    db.record_run(
        workflow_type="database",
        project_name="Development & testing",
        status="completed",
        validation_status="PASS",
        validation_passed=True,
        output_id="database_b.db",
        metadata={
            "source_type": "postgresql",
            "row_counts_by_table": {"customer": 10},
            "manifest_run_id": "run-1",
        },
    )

    run = db.list_runs()[0]
    assert run.metadata["source_type"] == "postgresql"
    assert "secret" not in str(run.metadata)
    assert "postgresql://" not in str(run.metadata)
