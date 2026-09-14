from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.domain.schema.models import (
    ColumnSchema,
    ConnectionHealth,
    DatabaseSchema,
    ForeignKey,
    TableSchema,
)
from synth_platform.engine.discovery.database.adapters.errors import SourceConnectionError
from synth_platform.interfaces.streamlit.postgres_source_adapter import PostgresSourceAdapter


class FakePostgresSource:
    def __init__(self, url, allowed_schemas, allowed_tables):
        self.url = url
        self.allowed_schemas = allowed_schemas
        self.allowed_tables = allowed_tables

    def health_check(self):
        return ConnectionHealth(
            server_version="16.0",
            current_database="app",
            current_user="reader",
            transaction_read_only=True,
            latency_ms=2.5,
        )

    def assert_read_only(self):
        return None

    def discover_schema(self):
        return DatabaseSchema(
            source_kind="postgresql_ro",
            tables={
                "customer": TableSchema(
                    name="customer",
                    schema_name="public",
                    primary_key="id",
                    primary_key_columns=["id"],
                    columns=[
                        ColumnSchema(name="id", physical_type="INTEGER", nullable=False),
                        ColumnSchema(name="email", physical_type="TEXT", nullable=True),
                    ],
                    estimated_row_count=3,
                ),
                "orders": TableSchema(
                    name="orders",
                    schema_name="public",
                    primary_key="id",
                    primary_key_columns=["id"],
                    columns=[ColumnSchema(name="id", physical_type="INTEGER", nullable=False)],
                    estimated_row_count=5,
                ),
            },
            foreign_keys=[
                ForeignKey(
                    parent_table="customer",
                    parent_column="id",
                    child_table="orders",
                    child_column="customer_id",
                )
            ],
        )

    def sample_table(self, table_name, max_rows, seed):
        return pd.DataFrame({"id": [1, 2], "email": ["a@example.com", "b@example.com"]})

    def close(self):
        return None


def test_postgres_adapter_wraps_existing_source(monkeypatch):
    monkeypatch.setattr(
        "synth_platform.interfaces.streamlit.postgres_source_adapter.PostgresSource",
        FakePostgresSource,
    )

    adapter = PostgresSourceAdapter(
        {
            "url": "postgresql://reader:secret@localhost/app?sslmode=disable",
            "allowed_schemas": ("public",),
            "allowed_tables": ("customer", "orders"),
        },
    )

    assert isinstance(adapter, PostgresSourceAdapter)
    assert adapter.test_connection()["healthy"] is True
    assert adapter.list_tables() == ["customer", "orders"]
    assert adapter.get_primary_keys("customer") == ["id"]
    assert adapter.get_foreign_keys("orders") == [
        {"column": "customer_id", "references_table": "customer", "references_column": "id"}
    ]
    assert adapter.read_sample("customer", limit=2).shape == (2, 2)


def test_postgres_adapter_surfaces_source_validation_errors():
    with pytest.raises(SourceConnectionError, match="TLS is required"):
        PostgresSourceAdapter(
            {
                "url": "postgresql://reader:secret@db.example.com/app?sslmode=disable",
                "allowed_schemas": ("public",),
            }
        )
