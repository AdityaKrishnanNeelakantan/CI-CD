"""Streamlit-facing PostgreSQL adapter for the Database Twin pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.engine.discovery.database.adapters.errors import (
    SourceConfigError,
    SourceConnectionError,
    UnknownTableError,
)
from synth_platform.errors import SourceUnavailableError, UnsafeIdentifierError
from synth_platform.infrastructure.sources.postgres import PostgresSource


class PostgresSourceAdapter(SourceAdapter):
    source_type = "postgresql"

    def __init__(self, connection_config: dict[str, Any]) -> None:
        url = connection_config.get("url")
        if not url or not isinstance(url, str):
            raise SourceConfigError("PostgreSQL source config requires a non-empty string 'url'.")

        allowed_schemas = tuple(connection_config.get("allowed_schemas") or ("public",))
        allowed_tables = connection_config.get("allowed_tables")
        if isinstance(allowed_tables, str):
            allowed_tables = tuple(
                table.strip() for table in allowed_tables.split(",") if table.strip()
            )
        elif allowed_tables is not None:
            allowed_tables = tuple(allowed_tables)

        try:
            self._source = PostgresSource(
                url,
                allowed_schemas=allowed_schemas,
                allowed_tables=allowed_tables,
            )
        except (SourceUnavailableError, UnsafeIdentifierError, ValueError) as exc:
            raise SourceConnectionError(str(exc)) from exc

    def test_connection(self) -> dict[str, Any]:
        try:
            health = self._source.health_check()
            self._source.assert_read_only()
            return {
                "healthy": True,
                "source_type": self.source_type,
                "server_version": health.server_version,
                "current_database": health.current_database,
                "current_user": health.current_user,
                "transaction_read_only": health.transaction_read_only,
                "latency_ms": health.latency_ms,
            }
        except Exception as exc:
            return {"healthy": False, "source_type": self.source_type, "error": str(exc)}

    def _schema(self):
        return self._source.discover_schema()

    def _require_table(self, table_name: str):
        schema = self._schema()
        table = schema.tables.get(table_name)
        if table is None:
            raise UnknownTableError(f"Unknown table: {table_name!r}")
        return table, schema

    def list_tables(self) -> list[str]:
        return sorted(self._schema().tables)

    def get_schema(self, table_name: str) -> list[dict[str, Any]]:
        table, _schema = self._require_table(table_name)
        return [
            {
                "name": column.name,
                "database_type": column.physical_type,
                "nullable": column.nullable,
                "default": column.default,
                "ordinal_position": index,
            }
            for index, column in enumerate(table.columns)
        ]

    def get_primary_keys(self, table_name: str) -> list[str]:
        table, _schema = self._require_table(table_name)
        return list(table.primary_key_columns)

    def get_foreign_keys(self, table_name: str) -> list[dict[str, Any]]:
        _table, schema = self._require_table(table_name)
        return [
            {
                "column": fk.child_column,
                "references_table": fk.parent_table,
                "references_column": fk.parent_column,
            }
            for fk in schema.foreign_keys
            if fk.child_table == table_name
        ]

    def get_indexes(self, table_name: str) -> list[dict[str, Any]]:
        table, _schema = self._require_table(table_name)
        return [
            {
                "name": index.name,
                "columns": list(index.columns),
                "unique": index.unique,
                "expression": index.expression,
            }
            for index in table.indexes
        ]

    def estimate_row_count(self, table_name: str) -> int:
        table, _schema = self._require_table(table_name)
        return int(table.estimated_row_count if table.estimated_row_count is not None else table.row_count)

    def read_sample(self, table_name: str, limit: int) -> pd.DataFrame:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        try:
            return self._source.sample_table(table_name, max_rows=limit, seed=42)
        except UnsafeIdentifierError as exc:
            raise UnknownTableError(str(exc)) from exc
        except SourceUnavailableError as exc:
            raise SourceConnectionError(str(exc)) from exc

    def read_sample_chunks(
        self, table_name: str, limit: int, chunk_size: int = 10_000
    ) -> Iterator[pd.DataFrame]:
        sample = self.read_sample(table_name, limit)
        for start in range(0, len(sample), chunk_size):
            yield sample.iloc[start : start + chunk_size]

    def close(self) -> None:
        self._source.close()
