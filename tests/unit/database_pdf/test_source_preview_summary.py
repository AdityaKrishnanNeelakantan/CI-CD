from __future__ import annotations

import sqlite3

import pytest

from synth_platform.application.workflows.database_twin import SQLiteSourceAdapter, summarize_source_preview

pytestmark = pytest.mark.unit


def test_source_preview_summary_reflects_live_rows_columns_and_samples(tmp_path):
    db_path = tmp_path / "source.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE customers (customer_id TEXT PRIMARY KEY, email TEXT)")
    conn.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer_id TEXT, amount REAL)")
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?)",
        [("c1", "one@example.com"), ("c2", "two@example.com")],
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)",
        [("o1", "c1", 10.0), ("o2", "c1", 20.0), ("o3", "c2", 30.0)],
    )
    conn.commit()
    conn.close()

    preview = summarize_source_preview(SQLiteSourceAdapter({"path": str(db_path)}), sample_rows=2)

    assert preview["table_count"] == 2
    assert preview["total_rows"] == 5
    assert preview["total_columns"] == 5
    assert preview["tables"]["customers"]["row_count"] == 2
    assert preview["tables"]["orders"]["column_count"] == 3
    assert len(preview["tables"]["orders"]["sample_rows"]) == 2
