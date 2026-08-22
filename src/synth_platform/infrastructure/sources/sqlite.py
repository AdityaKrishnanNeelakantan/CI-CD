"""Read-only SQLite connector with bounded reproducible seeded sampling."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

from synth_platform.infrastructure.sources.sqlalchemy_base import validate_identifier
from synth_platform.domain.profiling.models import SamplingRecord
from synth_platform.domain.schema.models import (
    CheckConstraint, ColumnSchema, CompositeKey, ConnectionHealth, DatabaseSchema, ForeignKey,
    IndexSchema, TableSchema, UniqueConstraint,
)
from synth_platform.errors import ReadOnlyViolationError, SourceUnavailableError


class SqliteSource:
    kind = "sqlite_ro"

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise SourceUnavailableError(f"database not found: {self.db_path}")
        self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self._conn.create_function(
            "sp_seed_hash",
            2,
            self._stable_row_hash,
            deterministic=True,
        )
        self._sampling: dict[str, SamplingRecord] = {}

    def health_check(self) -> ConnectionHealth:
        if self._conn.execute("SELECT 1").fetchone() != (1,):
            raise SourceUnavailableError("health check failed")
        return ConnectionHealth(
            server_version=sqlite3.sqlite_version,
            current_database=self.db_path.name,
            current_user="read_only_local",
            transaction_read_only=True,
            latency_ms=0.0,
        )

    def assert_read_only(self) -> None:
        try:
            self._conn.execute("CREATE TABLE _rw_probe (x INTEGER)")
        except sqlite3.OperationalError:
            return
        raise ReadOnlyViolationError("connection accepted a write")

    def _tables(self):
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def discover_schema(self) -> DatabaseSchema:
        tables, fks, composite = {}, [], []
        for t in self._tables():
            validate_identifier(t)
            info = self._conn.execute(f'PRAGMA table_info("{t}")').fetchall()
            cols = [ColumnSchema(name=c[1], physical_type=str(c[2] or "unknown"),
                                 nullable=not bool(c[3]), default=c[4]) for c in info]
            pk_cols = [c[1] for c in sorted(info, key=lambda row: row[5]) if c[5]]
            pk = pk_cols[0] if len(pk_cols) == 1 else None
            if len(pk_cols) > 1:
                composite.append(CompositeKey(table=t, columns=pk_cols))
            count = self._conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            indexes, uniques = [], []
            for idx in self._conn.execute(f'PRAGMA index_list("{t}")').fetchall():
                idx_name, unique = idx[1], bool(idx[2])
                idx_cols = [r[2] for r in self._conn.execute(
                    f'PRAGMA index_info("{idx_name}")').fetchall()]
                indexes.append(IndexSchema(name=idx_name, columns=idx_cols, unique=unique))
                if unique:
                    uniques.append(UniqueConstraint(name=idx_name, columns=idx_cols))
            tables[t] = TableSchema(
                name=t, primary_key=pk, primary_key_columns=pk_cols,
                primary_key_confirmed=bool(pk_cols), columns=cols,
                unique_constraints=uniques, indexes=indexes,
                row_count=int(count), estimated_row_count=int(count))
            for fk in self._conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall():
                fks.append(ForeignKey(
                    parent_table=fk[2], parent_column=fk[4] or "id",
                    child_table=t, child_column=fk[3], confirmed=True,
                    evidence="declared_pragma"))
        return DatabaseSchema(source_kind=self.kind, tables=tables,
                              foreign_keys=fks, composite_keys=composite)

    def count_rows(self, table: str) -> int:
        validate_identifier(table)
        return int(self._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

    @staticmethod
    def _stable_row_hash(rowid: int, seed: int) -> int:
        payload = f"{int(seed)}:{int(rowid)}".encode("ascii")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        # SQLite INTEGER is signed 64-bit. Keep the value positive and sortable.
        return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF

    def sample_table(self, table: str, max_rows: int, seed: int = 0) -> pd.DataFrame:
        validate_identifier(table)
        limit = max(0, int(max_rows))
        population = self.count_rows(table)
        query = (
            f'SELECT * FROM "{table}" '
            'ORDER BY sp_seed_hash(rowid, ?), rowid LIMIT ?'
        )
        frame = pd.read_sql_query(query, self._conn, params=(int(seed), limit))
        self._sampling[table] = SamplingRecord(
            strategy="sqlite_seeded_rowid", seed=seed,
            population_count=population, sample_count=len(frame),
            bias_warning=("" if len(frame) >= population
                          else "rowid-based deterministic sample; may reflect insertion order"))
        return frame

    def last_sampling_record(self, table: str) -> SamplingRecord | None:
        return self._sampling.get(table)

    def close(self) -> None:
        self._conn.close()
