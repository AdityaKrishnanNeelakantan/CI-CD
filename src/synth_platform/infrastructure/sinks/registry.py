"""Strict output sink registry."""
from __future__ import annotations

from synth_platform.infrastructure.sinks.csv_sink import CsvSink
from synth_platform.infrastructure.sinks.database_sink import SqliteSink
from synth_platform.infrastructure.sinks.parquet_sink import ParquetSink
from synth_platform.infrastructure.sinks.postgres_sink import PostgresSink


def build_output_sink(output_format: str, destination: str, **kwargs):
    fmt = output_format.lower()
    if fmt == "csv":
        return CsvSink(destination)
    if fmt == "sqlite":
        return SqliteSink(destination)
    if fmt == "parquet":
        return ParquetSink(destination)
    if fmt in {"postgres", "postgresql"}:
        target_schema = kwargs.get("target_schema")
        if not target_schema:
            raise ValueError("target_schema is required for PostgreSQL output")
        return PostgresSink(destination, target_schema=target_schema,
                            if_exists=kwargs.get("if_exists", "fail"))
    raise ValueError(f"unsupported output format: {output_format!r}")
