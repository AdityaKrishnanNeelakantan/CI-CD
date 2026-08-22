"""SQLite implementation of SourceAdapter.

All SQLite-specific behaviour (pragma queries, file:// read-only URIs,
sqlite_master introspection) is isolated in this module. Nothing outside
this file should know it is talking to SQLite.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.engine.discovery.database.adapters.errors import (
    SourceConfigError,
    SourceConnectionError,
    UnknownTableError,
)


class SQLiteSourceAdapter(SourceAdapter):
    source_type = "sqlite"

    def __init__(self, connection_config: dict[str, Any]) -> None:
        path = connection_config.get("path")
        if not path or not isinstance(path, str):
            raise SourceConfigError(
                "SQLite source config requires a non-empty string 'path'."
            )
        self._db_path = Path(path)

    def _read_only_uri(self) -> str:
        if not self._db_path.is_file():
            raise SourceConnectionError(
                f"SQLite database file not found: {self._db_path}"
            )
        return f"{self._db_path.resolve().as_uri()}?mode=ro"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        uri = self._read_only_uri()
        try:
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.OperationalError as exc:
            raise SourceConnectionError(f"Failed to open SQLite database: {exc}") from exc
        try:
            yield conn
        finally:
            conn.close()

    def test_connection(self) -> dict[str, Any]:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return {"healthy": True, "source_type": self.source_type}
        except SourceConnectionError as exc:
            return {"healthy": False, "source_type": self.source_type, "error": str(exc)}
        except sqlite3.Error as exc:
            return {"healthy": False, "source_type": self.source_type, "error": str(exc)}

    def list_tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        return [row[0] for row in rows]

    def _require_known_table(self, conn: sqlite3.Connection, table_name: str) -> None:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if row is None:
            raise UnknownTableError(f"Unknown table: {table_name!r}")

    def get_schema(self, table_name: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self._require_known_table(conn, table_name)
            rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        columns = []
        for cid, name, decl_type, notnull, default_value, pk in rows:
            columns.append(
                {
                    "name": name,
                    "database_type": decl_type or "UNKNOWN",
                    "nullable": not bool(notnull),
                    "default": default_value,
                    "ordinal_position": cid,
                }
            )
        return columns

    def get_primary_keys(self, table_name: str) -> list[str]:
        with self._connect() as conn:
            self._require_known_table(conn, table_name)
            rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        pk_columns = [(pk, name) for _cid, name, _t, _nn, _dflt, pk in rows if pk]
        pk_columns.sort(key=lambda item: item[0])
        return [name for _pk_order, name in pk_columns]

    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self._require_known_table(conn, table_name)
            rows = conn.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()
        foreign_keys = []
        for row in rows:
            # id, seq, table, from, to, on_update, on_delete, match
            _id, _seq, ref_table, from_col, to_col, *_rest = row
            foreign_keys.append(
                {
                    "column": from_col,
                    "references_table": ref_table,
                    "references_column": to_col,
                }
            )
        return foreign_keys

    def get_indexes(self, table_name: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self._require_known_table(conn, table_name)
            index_rows = conn.execute(f'PRAGMA index_list("{table_name}")').fetchall()
            indexes = []
            for row in index_rows:
                _seq, index_name, is_unique, origin, _partial = row
                col_rows = conn.execute(f'PRAGMA index_info("{index_name}")').fetchall()
                columns = [col_row[2] for col_row in sorted(col_rows, key=lambda r: r[0])]
                indexes.append(
                    {
                        "name": index_name,
                        "columns": columns,
                        "unique": bool(is_unique),
                        "origin": origin,
                    }
                )
        return indexes

    def estimate_row_count(self, table_name: str) -> int:
        with self._connect() as conn:
            self._require_known_table(conn, table_name)
            row = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
        return int(row[0])

    def read_sample(self, table_name: str, limit: int) -> pd.DataFrame:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        with self._connect() as conn:
            self._require_known_table(conn, table_name)
            return pd.read_sql_query(
                f'SELECT * FROM "{table_name}" LIMIT ?', conn, params=(limit,)
            )

    def read_sample_chunks(
        self, table_name: str, limit: int, chunk_size: int = 10_000
    ) -> Iterator[pd.DataFrame]:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        with self._connect() as conn:
            self._require_known_table(conn, table_name)
            yield from pd.read_sql_query(
                f'SELECT * FROM "{table_name}" LIMIT ?', conn, params=(limit,), chunksize=chunk_size
            )
