from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.discovery.database.adapters.errors import SourceConfigError, SourceConnectionError, UnknownTableError
from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter

pytestmark = pytest.mark.unit


def make_adapter(db_path: Path) -> SQLiteSourceAdapter:
    return SQLiteSourceAdapter({"path": str(db_path)})


def test_requires_path_in_config():
    with pytest.raises(SourceConfigError):
        SQLiteSourceAdapter({})


def test_connection_success(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    health = adapter.test_connection()
    assert health["healthy"] is True
    assert health["source_type"] == "sqlite"


def test_connection_missing_file_reports_unhealthy(tmp_path: Path):
    adapter = make_adapter(tmp_path / "does_not_exist.db")
    health = adapter.test_connection()
    assert health["healthy"] is False
    assert "error" in health


def test_operations_raise_on_missing_file(tmp_path: Path):
    adapter = make_adapter(tmp_path / "does_not_exist.db")
    with pytest.raises(SourceConnectionError):
        adapter.list_tables()


def test_list_tables_discovers_all_tables(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    assert adapter.list_tables() == ["customers", "orders"]


def test_list_tables_empty_database(empty_sqlite_db: Path):
    adapter = make_adapter(empty_sqlite_db)
    assert adapter.list_tables() == []


def test_get_schema_reports_columns_and_nullability(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    columns = {col["name"]: col for col in adapter.get_schema("customers")}
    assert set(columns) == {"customer_id", "name", "email", "signup_date"}
    # SQLite does not imply NOT NULL for non-INTEGER primary keys unless it
    # is declared explicitly, so this reflects the fixture's real DDL, not
    # an adapter inference - the pragma is reported verbatim.
    assert columns["customer_id"]["nullable"] is True
    assert columns["name"]["nullable"] is False
    assert columns["email"]["nullable"] is True
    assert columns["customer_id"]["database_type"] == "TEXT"


def test_get_schema_unknown_table_raises(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    with pytest.raises(UnknownTableError):
        adapter.get_schema("nonexistent_table")


def test_read_sample_chunks_yields_dataframes_whose_concatenation_matches_read_sample(
    temp_sqlite_db: Path,
):
    adapter = make_adapter(temp_sqlite_db)
    whole = adapter.read_sample("orders", limit=100)
    chunks = list(adapter.read_sample_chunks("orders", limit=100, chunk_size=2))

    assert [len(c) for c in chunks] == [2, 2, 1]
    concatenated = pd.concat(chunks, ignore_index=True)
    assert concatenated["order_id"].tolist() == whole["order_id"].tolist()


def test_read_sample_chunks_respects_limit(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    chunks = list(adapter.read_sample_chunks("orders", limit=3, chunk_size=2))
    assert sum(len(c) for c in chunks) == 3


def test_read_sample_chunks_rejects_non_positive_limit(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    with pytest.raises(ValueError):
        list(adapter.read_sample_chunks("orders", limit=0, chunk_size=2))


def test_read_sample_chunks_rejects_non_positive_chunk_size(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    with pytest.raises(ValueError):
        list(adapter.read_sample_chunks("orders", limit=10, chunk_size=0))


def test_read_sample_chunks_unknown_table_raises(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    with pytest.raises(UnknownTableError):
        list(adapter.read_sample_chunks("nonexistent_table", limit=10, chunk_size=2))


def test_get_primary_keys(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    assert adapter.get_primary_keys("customers") == ["customer_id"]
    assert adapter.get_primary_keys("orders") == ["order_id"]


def test_get_foreign_keys(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    fks = adapter.get_foreign_keys("orders")
    assert fks == [
        {
            "column": "customer_id",
            "references_table": "customers",
            "references_column": "customer_id",
        }
    ]
    assert adapter.get_foreign_keys("customers") == []


def test_get_foreign_keys_unknown_table_raises(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    with pytest.raises(UnknownTableError):
        adapter.get_foreign_keys("nonexistent_table")


def test_get_indexes(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    indexes = adapter.get_indexes("orders")
    assert any(idx["columns"] == ["customer_id"] for idx in indexes)


def test_estimate_row_count(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    assert adapter.estimate_row_count("customers") == 3
    assert adapter.estimate_row_count("orders") == 5


def test_estimate_row_count_unknown_table_raises(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    with pytest.raises(UnknownTableError):
        adapter.estimate_row_count("nonexistent_table")


def test_read_sample_returns_dataframe_with_expected_columns(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    df = adapter.read_sample("customers", limit=100)
    assert list(df.columns) == ["customer_id", "name", "email", "signup_date"]
    assert len(df) == 3


def test_read_sample_respects_limit(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    df = adapter.read_sample("orders", limit=2)
    assert len(df) == 2


def test_read_sample_unknown_table_raises(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    with pytest.raises(UnknownTableError):
        adapter.read_sample("nonexistent_table", limit=10)


def test_read_sample_rejects_non_positive_limit(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    with pytest.raises(ValueError):
        adapter.read_sample("customers", limit=0)


def test_connection_is_read_only(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    with adapter._connect() as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO customers (customer_id, name) VALUES ('x', 'y')")


def test_discover_produces_generic_contract_with_no_special_casing(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    result = adapter.discover()

    assert result["source_type"] == "sqlite"
    assert set(result["tables"]) == {"customers", "orders"}
    assert result["tables"]["orders"]["primary_key"] == ["order_id"]
    assert result["tables"]["orders"]["foreign_keys"][0]["references_table"] == "customers"
    assert result["source_fingerprint"].startswith("sha256:")


def test_discover_fingerprint_is_stable_across_calls(temp_sqlite_db: Path):
    adapter = make_adapter(temp_sqlite_db)
    first = adapter.discover()["source_fingerprint"]
    second = adapter.discover()["source_fingerprint"]
    assert first == second


def test_discover_fingerprint_excludes_row_data(temp_sqlite_db: Path, tmp_path: Path):
    """Two schemas that are structurally identical but hold different data
    must fingerprint identically - the fingerprint is schema-only evidence.
    """
    other_db = tmp_path / "other.db"
    conn = sqlite3.connect(str(other_db))
    conn.executescript(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            signup_date TEXT
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE INDEX idx_orders_customer_id ON orders(customer_id);

        INSERT INTO customers (customer_id, name, email, signup_date)
            VALUES ('completely-different-row', 'Someone Else', NULL, '1999-01-01');
        """
    )
    conn.commit()
    conn.close()

    original_fingerprint = make_adapter(temp_sqlite_db).discover()["source_fingerprint"]
    other_fingerprint = make_adapter(other_db).discover()["source_fingerprint"]
    assert original_fingerprint == other_fingerprint
