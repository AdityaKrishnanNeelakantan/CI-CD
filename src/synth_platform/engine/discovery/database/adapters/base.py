"""Generic source-access contract.

All source-specific behaviour (SQLite, PostgreSQL, CSV, ...) must live
inside a SourceAdapter implementation. Nothing outside this module and its
concrete adapter implementations may branch on source type - the rest of
the pipeline (profiler, semantic inference, trainer, validator) consumes
only the generic dict returned by discover() and the pandas.DataFrame
returned by read_sample().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from synth_platform.engine.common.database.core.fingerprint import compute_fingerprint
from synth_platform.engine.common.database.core.scaling import DEFAULT_CHUNK_SIZE, iter_dataframe_chunks


class SourceAdapter(ABC):
    """Read-only access to a single structured data source."""

    #: Short identifier for this adapter, e.g. "sqlite". Must match the
    #: `source.type` value in project.yaml that selects this adapter.
    source_type: str

    @abstractmethod
    def __init__(self, connection_config: dict[str, Any]) -> None:
        """Every adapter is constructed from the `source.connection` dict
        in project.yaml (or the equivalent programmatic config). This is
        the entire plug-in contract a new connector must satisfy - the
        registry (src/adapters/registry.py) only ever calls
        `adapter_cls(connection_config)`, never a source-specific
        constructor."""

    @abstractmethod
    def test_connection(self) -> dict[str, Any]:
        """Run a lightweight health check and return its result.

        Must never include credentials or a full connection string in the
        returned dict.
        """

    @abstractmethod
    def list_tables(self) -> list[str]:
        """Return the names of tables visible to this adapter."""

    @abstractmethod
    def get_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Return column metadata for a table: name, database_type, nullable, default."""

    @abstractmethod
    def get_primary_keys(self, table_name: str) -> list[str]:
        """Return the ordered primary-key column names for a table."""

    @abstractmethod
    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        """Return foreign keys as dicts with column, references_table, references_column."""

    @abstractmethod
    def estimate_row_count(self, table_name: str) -> int:
        """Return an approximate or exact row count for a table."""

    @abstractmethod
    def read_sample(self, table_name: str, limit: int) -> pd.DataFrame:
        """Read a bounded sample from an approved table. Never reads the full table by default."""

    def read_sample_chunks(
        self, table_name: str, limit: int, chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> Iterator[pd.DataFrame]:
        """Yield the bounded sample in memory-bounded pieces.

        Default implementation reads the full sample once and slices it in
        memory, so it bounds downstream processing memory but not the read
        itself. Adapters that can stream directly from their source (e.g.
        a SQL adapter using a server-side cursor) should override this for
        a genuinely memory-bounded read.
        """
        yield from iter_dataframe_chunks(self.read_sample(table_name, limit), chunk_size)

    def get_indexes(self, table_name: str) -> list[dict[str, Any]]:
        """Return index metadata for a table. Default: unsupported, returns []."""
        return []

    def discover(self) -> dict[str, Any]:
        """Compose the granular metadata methods into the generic discovery contract.

        This method is intentionally concrete (not overridable per adapter):
        genericity comes from every adapter implementing the same granular
        methods, not from each adapter formatting its own discovery output.
        """
        tables: dict[str, Any] = {}
        structural_tables: dict[str, Any] = {}
        for table_name in sorted(self.list_tables()):
            columns = self.get_schema(table_name)
            primary_key = self.get_primary_keys(table_name)
            foreign_keys = self.get_foreign_keys(table_name)
            indexes = self.get_indexes(table_name)

            tables[table_name] = {
                "columns": columns,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
                "estimated_row_count": self.estimate_row_count(table_name),
            }
            # Row counts and other data statistics are deliberately excluded
            # from the fingerprint payload: the fingerprint identifies the
            # *shape* of the source, and must stay stable as rows are
            # inserted or deleted so it can be compared across runs.
            structural_tables[table_name] = {
                "columns": columns,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
            }

        schema_payload = {"source_type": self.source_type, "tables": structural_tables}
        source_fingerprint = compute_fingerprint(schema_payload)

        return {
            "source_type": self.source_type,
            "discovered_at": datetime.now(UTC).isoformat(),
            "tables": tables,
            "source_fingerprint": source_fingerprint,
        }
