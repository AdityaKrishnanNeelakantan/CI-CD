"""SqliteExtractionPass: the trivial (self-describing) extraction case.

Read-only, PRAGMA-driven schema discovery + bounded deterministic sampling,
lifted into the canonical RelationalDataset. No PK guessing.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from synth_platform.infrastructure.sources.sqlalchemy_base import validate_identifier
from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.domain.schema.models import (
    ColumnSchema, DatabaseSchema, ForeignKey, TableSchema,
)
from synth_platform.errors import ReadOnlyViolationError, SourceUnavailableError


class SqliteExtractionPass:
    kind = "sqlite_ro"

    def __init__(self, db_path: str | Path, max_rows: int = 50_000, seed: int = 0):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise SourceUnavailableError(f"database not found: {self.db_path}")
        self.max_rows = max_rows
        self.seed = seed
        self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def assert_read_only(self) -> None:
        try:
            self._conn.execute("CREATE TABLE _rw_probe (x INTEGER)")
        except sqlite3.OperationalError:
            return
        raise ReadOnlyViolationError("connection accepted a write")

    def _table_names(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def extract(self) -> RelationalDataset:
        tables_schema: dict[str, TableSchema] = {}
        fks: list[ForeignKey] = []
        frames: dict[str, pd.DataFrame] = {}
        for t in self._table_names():
            validate_identifier(t)
            info = self._conn.execute(f'PRAGMA table_info("{t}")').fetchall()
            cols = [ColumnSchema(name=c[1], physical_type=str(c[2] or "unknown"),
                                 nullable=not c[3]) for c in info]
            pk_cols = [c[1] for c in info if c[5]]
            pk = pk_cols[0] if len(pk_cols) == 1 else None
            frames[t] = pd.read_sql_query(
                f'SELECT * FROM "{t}" ORDER BY rowid LIMIT {int(self.max_rows)}', self._conn)
            tables_schema[t] = TableSchema(
                name=t, primary_key=pk, primary_key_confirmed=pk is not None,
                columns=cols, row_count=int(len(frames[t])))
            for fk in self._conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall():
                fks.append(ForeignKey(parent_table=fk[2], parent_column=fk[4] or "id",
                                      child_table=t, child_column=fk[3],
                                      confirmed=True, evidence="declared_pragma"))
        schema = DatabaseSchema(source_kind=self.kind, tables=tables_schema, foreign_keys=fks)
        return RelationalDataset(schema=schema, tables=frames).finalize_counts()

    def close(self) -> None:
        self._conn.close()
