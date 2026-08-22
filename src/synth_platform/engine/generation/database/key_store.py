"""Disk-backed store of synthetic parent primary keys for streaming FK assignment.

Only synthetic keys are stored — never source PII. Used by the streaming
relational generator so child batches can sample parent keys without holding
full parent DataFrames in memory.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def _normalize_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value == value and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        head = text[:-2]
        if head.lstrip("-").isdigit():
            return head
    return text


def _infer_kind(values: Sequence[Any]) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return "int"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float) and value == value and value.is_integer():
            return "int"
        return "str"
    return "str"


def _coerce(value: str, kind: str) -> Any:
    if kind == "int":
        try:
            return int(value)
        except ValueError:
            return value
    return value


class ParentKeyStore:
    """SQLite-backed registry of parent PK values for FK sampling."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_keys (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                key_value TEXT NOT NULL,
                PRIMARY KEY (table_name, column_name, key_value)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS key_meta (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                value_kind TEXT NOT NULL,
                PRIMARY KEY (table_name, column_name)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ParentKeyStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _value_kind(self, table: str, column: str) -> str:
        row = self._conn.execute(
            "SELECT value_kind FROM key_meta WHERE table_name = ? AND column_name = ?",
            (table, column),
        ).fetchone()
        return str(row[0]) if row else "str"

    def register_keys(
        self,
        table: str,
        column: str,
        values: Iterable[Any],
    ) -> int:
        """Insert synthetic key values for ``table.column``. Returns rows inserted."""
        materialised = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
        if not materialised:
            return 0

        if self._conn.execute(
            "SELECT 1 FROM key_meta WHERE table_name = ? AND column_name = ? LIMIT 1",
            (table, column),
        ).fetchone() is None:
            kind = _infer_kind(materialised)
            self._conn.execute(
                "INSERT INTO key_meta (table_name, column_name, value_kind) VALUES (?, ?, ?)",
                (table, column, kind),
            )

        rows = [(table, column, _normalize_key(v)) for v in materialised if _normalize_key(v) != ""]
        if not rows:
            return 0
        before = self.key_set_count(table, column)
        self._conn.executemany(
            "INSERT OR IGNORE INTO parent_keys (table_name, column_name, key_value) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return self.key_set_count(table, column) - before

    def key_set_count(self, table: str, column: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM parent_keys WHERE table_name = ? AND column_name = ?",
            (table, column),
        ).fetchone()
        return int(row[0]) if row else 0

    def sample_keys(
        self,
        table: str,
        column: str,
        n: int,
        rng,
    ) -> list[Any]:
        """Uniformly sample ``n`` keys (with replacement) from the registered set.

        ``rng`` must expose ``choice(seq)`` (e.g. ``random.Random``).
        """
        if n <= 0:
            return []
        count = self.key_set_count(table, column)
        if count == 0:
            raise ValueError(f"no keys registered for {table}.{column}")

        kind = self._value_kind(table, column)
        keys = [
            _coerce(row[0], kind)
            for row in self._conn.execute(
                "SELECT key_value FROM parent_keys WHERE table_name = ? AND column_name = ?",
                (table, column),
            )
        ]
        return [rng.choice(keys) for _ in range(n)]

    def contains(self, table: str, column: str, value: Any) -> bool:
        if value is None or (isinstance(value, float) and value != value):
            return False
        row = self._conn.execute(
            "SELECT 1 FROM parent_keys WHERE table_name = ? AND column_name = ? AND key_value = ? LIMIT 1",
            (table, column, _normalize_key(value)),
        ).fetchone()
        return row is not None

    def contains_all(self, table: str, column: str, values: Sequence[Any]) -> tuple[int, int]:
        """Return (valid_count, total_count) for membership of ``values`` in the key set."""
        total = len(values)
        if total == 0:
            return 0, 0
        valid = sum(1 for value in values if self.contains(table, column, value))
        return valid, total
