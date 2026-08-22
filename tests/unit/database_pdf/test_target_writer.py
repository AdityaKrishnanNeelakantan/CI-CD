from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.generation.database.target_writer import (
    TargetSchemaMismatchError,
    TargetWriterError,
    validate_write,
    write_dataset,
)

pytestmark = pytest.mark.unit


def _contract():
    return {
        "tables": {
            "customers": {
                "primary_key": ["customer_id"],
                "foreign_keys": [],
                "columns": {
                    "customer_id": {"physical_type": "TEXT", "nullable": False},
                    "name": {"physical_type": "TEXT", "nullable": True},
                },
            },
            "orders": {
                "primary_key": ["order_id"],
                "foreign_keys": [{"column": "customer_id", "references_table": "customers", "references_column": "customer_id"}],
                "columns": {
                    "order_id": {"physical_type": "INTEGER", "nullable": False},
                    "customer_id": {"physical_type": "TEXT", "nullable": False},
                    "amount": {"physical_type": "REAL", "nullable": False},
                },
            },
        }
    }


def _tables():
    return {
        "customers": pd.DataFrame({"customer_id": ["a", "b"], "name": ["A", "B"]}),
        "orders": pd.DataFrame({"order_id": [1, 2], "customer_id": ["a", "b"], "amount": [10.0, 20.0]}),
    }


def test_write_to_empty_target_creates_tables_and_inserts_rows(tmp_path: Path):
    target_path = tmp_path / "target.db"
    report = write_dataset(target_path, _tables(), _contract(), ["customers", "orders"])
    assert report["tables"]["customers"]["mode"] == "created"
    assert report["tables"]["orders"]["mode"] == "created"

    conn = sqlite3.connect(str(target_path))
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 2
    conn.close()


def test_write_wraps_connection_failure_as_target_writer_error(tmp_path: Path):
    """A target path whose parent directory doesn't exist can never be
    opened by sqlite3.connect() - this must surface as TargetWriterError
    (the only exception type target_write_service.py's run_target_write
    catches to produce a clean failed StageResult), not a raw sqlite3
    exception that would crash the whole pipeline stage.
    """
    target_path = tmp_path / "does_not_exist" / "target.db"
    with pytest.raises(TargetWriterError):
        write_dataset(target_path, _tables(), _contract(), ["customers", "orders"])


def test_write_respects_foreign_key_constraints(tmp_path: Path):
    """A real, physical FK constraint check by SQLite itself - a second,
    independent guarantee beyond src/relational/relational_generator.py's
    own compute_fk_validity().
    """
    target_path = tmp_path / "target.db"
    tables = _tables()
    tables["orders"] = pd.DataFrame({"order_id": [1], "customer_id": ["does-not-exist"], "amount": [5.0]})
    with pytest.raises(sqlite3.IntegrityError):
        write_dataset(target_path, tables, _contract(), ["customers", "orders"])


def test_write_to_existing_compatible_table_appends_rows(tmp_path: Path):
    target_path = tmp_path / "target.db"
    write_dataset(target_path, _tables(), _contract(), ["customers", "orders"])

    more_customers = {"customers": pd.DataFrame({"customer_id": ["c"], "name": ["C"]}), "orders": pd.DataFrame(columns=["order_id", "customer_id", "amount"])}
    report = write_dataset(target_path, more_customers, _contract(), ["customers", "orders"])
    assert report["tables"]["customers"]["mode"] == "existing_compatible"

    conn = sqlite3.connect(str(target_path))
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 3
    conn.close()


def test_write_to_incompatible_existing_table_fails_explicitly(tmp_path: Path):
    target_path = tmp_path / "target.db"
    conn = sqlite3.connect(str(target_path))
    conn.execute("CREATE TABLE customers (totally_unrelated_column TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(TargetSchemaMismatchError):
        write_dataset(target_path, _tables(), _contract(), ["customers", "orders"])


def test_failed_write_rolls_back_the_entire_transaction(tmp_path: Path):
    """customers succeeds, orders violates its FK - customers' insert
    must not remain committed either; the whole batch is one transaction.
    """
    target_path = tmp_path / "target.db"
    tables = _tables()
    tables["orders"] = pd.DataFrame({"order_id": [1], "customer_id": ["does-not-exist"], "amount": [5.0]})

    with pytest.raises(sqlite3.IntegrityError):
        write_dataset(target_path, tables, _contract(), ["customers", "orders"])

    conn = sqlite3.connect(str(target_path))
    tables_present = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    assert tables_present == []  # nothing committed, not even the customers table's CREATE TABLE


def test_validate_write_confirms_row_counts(tmp_path: Path):
    target_path = tmp_path / "target.db"
    tables = _tables()
    write_dataset(target_path, tables, _contract(), ["customers", "orders"])
    validation = validate_write(target_path, tables)
    assert validation["all_writes_confirmed"] is True
    assert validation["tables"]["customers"]["actual_total_rows"] == 2
