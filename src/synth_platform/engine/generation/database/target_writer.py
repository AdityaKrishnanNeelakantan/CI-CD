"""Writes a validated, relationally-generated synthetic dataset into a
target SQLite database (Database B) - the final stage of the database
track's "Database A -> generate -> Database B" flow.

Three cases, matching this project's reference design:
1. Target has no table of that name yet -> create it from the canonical
   dataset_contract schema, then bulk insert.
2. Target already has a table whose columns are a superset of the
   generated columns -> bulk insert into the existing table as-is.
3. Target has a table whose columns don't cover the generated dataset ->
   fail explicitly (TargetSchemaMismatchError). This writer never
   guesses a column mapping or silently drops/renames anything.

All tables in one write_dataset() call share a single transaction: if
any table fails validation or insertion, every table's write in that
call is rolled back, never partially committed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class TargetWriterError(Exception):
    """Base class for target-database write errors."""


class TargetSchemaMismatchError(TargetWriterError):
    """Raised when a target table exists but its columns are not
    compatible with the dataset being written.
    """


def _sqlite_affinity(physical_type: str) -> str:
    physical_type = (physical_type or "").upper()
    if "INT" in physical_type:
        return "INTEGER"
    if any(token in physical_type for token in ("REAL", "FLOA", "DOUB", "NUMERIC", "DECIMAL")):
        return "REAL"
    return "TEXT"


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_create_table_sql(table_name: str, table_contract: dict[str, Any]) -> str:
    primary_key = table_contract.get("primary_key", [])
    column_defs = [
        f"{_quote_ident(column_name)} {_sqlite_affinity(column_entry.get('physical_type', ''))}"
        + ("" if column_entry.get("nullable", True) else " NOT NULL")
        for column_name, column_entry in table_contract["columns"].items()
    ]
    for fk in table_contract.get("foreign_keys", []):
        column_defs.append(
            f"FOREIGN KEY ({_quote_ident(fk['column'])}) REFERENCES "
            f"{_quote_ident(fk['references_table'])}({_quote_ident(fk['references_column'])})"
        )
    pk_clause = f", PRIMARY KEY ({', '.join(_quote_ident(c) for c in primary_key)})" if primary_key else ""
    return f"CREATE TABLE {_quote_ident(table_name)} (\n  " + ",\n  ".join(column_defs) + pk_clause + "\n)"


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
    return row is not None


def _existing_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_ident(table_name)})").fetchall()
    return {row[1] for row in rows}


def _insert_rows(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    # Deliberately not pandas.DataFrame.to_sql(): it issues its own
    # internal commit on a raw sqlite3 connection, silently ending any
    # outer transaction the caller already started - verified
    # empirically (conn.in_transaction flips to False immediately after
    # a to_sql() call, even mid-transaction), which broke the "one
    # transaction for the whole write" guarantee this module exists to
    # provide. executemany() has no such side effect.
    if df.empty:
        return
    columns = list(df.columns)
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(_quote_ident(c) for c in columns)
    conn.executemany(
        f"INSERT INTO {_quote_ident(table_name)} ({column_list}) VALUES ({placeholders})",
        df.itertuples(index=False, name=None),
    )


def _validate_schema_compatibility(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    existing = _existing_columns(conn, table_name)
    incoming = set(df.columns)
    missing_in_target = incoming - existing
    if missing_in_target:
        raise TargetSchemaMismatchError(
            f"target table {table_name!r} does not have columns {sorted(missing_in_target)} "
            "present in the generated dataset. This writer never creates a mapping automatically - "
            "the target table's existing schema must already be compatible, or the table must not exist yet."
        )


def write_dataset(
    target_db_path: str | Path,
    tables: dict[str, pd.DataFrame],
    dataset_contract: dict[str, Any],
    generation_order: list[str],
) -> dict[str, Any]:
    """Write every table in generation_order (parents before children, so
    FK constraints are satisfiable row by row) inside one transaction.
    """
    try:
        conn = sqlite3.connect(str(target_db_path))
    except sqlite3.Error as exc:
        raise TargetWriterError(f"could not open target database {target_db_path}: {exc}") from exc

    write_report: dict[str, Any] = {"tables": {}}

    try:
        conn.execute("PRAGMA foreign_keys = ON")
        # Python's sqlite3 module only auto-opens an implicit transaction
        # before INSERT/UPDATE/DELETE, never before DDL - without this
        # explicit BEGIN, CREATE TABLE commits immediately regardless of a
        # later rollback() (verified empirically: a failed write left the
        # already-created table behind). SQLite itself fully supports
        # transactional DDL once inside an explicit transaction.
        conn.execute("BEGIN")
        for table_name in generation_order:
            df = tables[table_name]
            table_contract = dataset_contract["tables"][table_name]

            if not _table_exists(conn, table_name):
                conn.execute(build_create_table_sql(table_name, table_contract))
                mode = "created"
            else:
                _validate_schema_compatibility(conn, table_name, df)
                mode = "existing_compatible"

            _insert_rows(conn, table_name, df)
            write_report["tables"][table_name] = {"mode": mode, "rows_written": len(df)}

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return write_report


def validate_write(target_db_path: str | Path, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Post-write validation: recount rows actually present in the target
    for each table just written, confirming the write really landed.
    """
    conn = sqlite3.connect(str(target_db_path))
    try:
        results: dict[str, Any] = {}
        all_confirmed = True
        for table_name, df in tables.items():
            actual_total = conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table_name)}").fetchone()[0]
            expected_new_rows = len(df)
            confirmed = actual_total >= expected_new_rows
            results[table_name] = {"expected_new_rows": expected_new_rows, "actual_total_rows": actual_total}
            all_confirmed = all_confirmed and confirmed
        return {"tables": results, "all_writes_confirmed": all_confirmed}
    finally:
        conn.close()
