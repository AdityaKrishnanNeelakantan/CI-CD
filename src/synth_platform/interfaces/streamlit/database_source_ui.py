"""Helpers for Database Twin source selection in Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.interfaces.streamlit.postgres_source_adapter import PostgresSourceAdapter


@dataclass(frozen=True)
class PostgresConnectionDetails:
    host: str
    port: int
    database: str
    username: str
    password: str
    sslmode: str
    allowed_schemas: tuple[str, ...] = ("public",)
    allowed_tables: tuple[str, ...] | None = None


def postgres_url(details: PostgresConnectionDetails) -> str:
    user = quote(details.username, safe="")
    password = quote(details.password, safe="")
    host = details.host.strip()
    database = quote(details.database.strip(), safe="")
    sslmode = quote(details.sslmode.strip(), safe="")
    return f"postgresql://{user}:{password}@{host}:{int(details.port)}/{database}?sslmode={sslmode}"


def postgres_connection_config(details: PostgresConnectionDetails) -> dict:
    return {
        "url": postgres_url(details),
        "allowed_schemas": details.allowed_schemas,
        "allowed_tables": details.allowed_tables,
    }


def build_source_adapter(source_type: str, connection_config: dict) -> SourceAdapter:
    if source_type == "sqlite":
        return SQLiteSourceAdapter(connection_config)
    if source_type == "postgresql":
        return PostgresSourceAdapter(connection_config)
    raise ValueError(f"unsupported Database Twin source type: {source_type!r}")
