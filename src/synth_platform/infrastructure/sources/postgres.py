"""Read-only PostgreSQL source connector with strict allowlists and sampling."""
from __future__ import annotations

import re
import time
from collections.abc import Iterable
from contextlib import contextmanager
from urllib.parse import urlsplit

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine, make_url

from synth_platform.domain.profiling.models import SamplingRecord
from synth_platform.domain.schema.models import (
    CheckConstraint, ColumnSchema, CompositeKey, ConnectionHealth, DatabaseSchema,
    ForeignKey, IndexSchema, TableRef, TableSchema, UniqueConstraint,
)
from synth_platform.errors import (
    ReadOnlyViolationError, SourceUnavailableError, UnsafeIdentifierError,
)


class PostgresSource:
    kind = "postgresql_ro"

    def __init__(
        self,
        url: str,
        allowed_schemas: tuple[str, ...],
        allowed_tables: tuple[str, ...] | None,
        statement_timeout_ms: int = 120_000,
        lock_timeout_ms: int = 5_000,
    ):
        if not allowed_schemas:
            raise ValueError("at least one allowed schema is required")
        self.allowed_schemas = tuple(self._validate_identifier(x) for x in allowed_schemas)
        self.allowed_tables = (None if allowed_tables is None else
                               tuple(self._normalise_allowed_table(x) for x in allowed_tables))
        self.statement_timeout_ms = int(statement_timeout_ms)
        self.lock_timeout_ms = int(lock_timeout_ms)
        parsed = make_url(url)
        if parsed.drivername == "postgresql":
            parsed = parsed.set(drivername="postgresql+psycopg")
            url = parsed.render_as_string(hide_password=False)
        host = (parsed.host or "").lower()
        local = host in {"", "localhost", "127.0.0.1", "::1"}
        query_ssl = str(parsed.query.get("sslmode", "")).lower()
        if not local and query_ssl in {"disable", "allow", "prefer"}:
            raise SourceUnavailableError("TLS is required for non-local PostgreSQL hosts")
        connect_args = {"application_name": "synth_platform_training"}
        if not local and not query_ssl:
            connect_args["sslmode"] = "require"
        try:
            self._engine: Engine = create_engine(
                url, pool_pre_ping=True, connect_args=connect_args,
            )
        except (ModuleNotFoundError, ImportError) as exc:
            raise SourceUnavailableError(
                "PostgreSQL driver unavailable; install the postgresql optional dependency"
            ) from exc
        self._sampling: dict[str, SamplingRecord] = {}

    @staticmethod
    def _validate_identifier(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", value or ""):
            raise UnsafeIdentifierError(f"unsafe SQL identifier: {value!r}")
        return value

    def _normalise_allowed_table(self, value: str) -> str:
        bits = value.split(".", 1)
        if len(bits) == 2:
            schema, name = map(self._validate_identifier, bits)
            return f"{schema}.{name}"
        return self._validate_identifier(value)

    def _resolve(self, table: str | TableRef) -> TableRef:
        if isinstance(table, TableRef):
            ref = table
        elif "." in table:
            schema, name = table.split(".", 1)
            ref = TableRef(schema=schema, name=name)
        elif len(self.allowed_schemas) == 1:
            ref = TableRef(schema=self.allowed_schemas[0], name=table)
        else:
            matches = [t for t in (self.allowed_tables or ()) if t.endswith(f".{table}")]
            if len(matches) != 1:
                raise UnsafeIdentifierError(
                    f"ambiguous table {table!r}; use schema.table")
            schema, name = matches[0].split(".", 1)
            ref = TableRef(schema=schema, name=name)
        schema = self._validate_identifier(ref.schema or "")
        name = self._validate_identifier(ref.name)
        if schema not in self.allowed_schemas:
            raise UnsafeIdentifierError(f"schema not allowlisted: {schema!r}")
        if self.allowed_tables is not None:
            allowed = set(self.allowed_tables)
            if name not in allowed and f"{schema}.{name}" not in allowed:
                raise UnsafeIdentifierError(f"table not allowlisted: {schema}.{name}")
        return TableRef(schema=schema, name=name)

    def _quote(self, value: str) -> str:
        return self._engine.dialect.identifier_preparer.quote(value)

    def _qualified(self, ref: TableRef) -> str:
        return f"{self._quote(ref.schema or '')}.{self._quote(ref.name)}"

    @contextmanager
    def _readonly_connection(self):
        with self._engine.connect() as conn:
            tx = conn.begin()
            try:
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                conn.exec_driver_sql(
                    f"SET LOCAL statement_timeout = '{self.statement_timeout_ms}ms'")
                conn.exec_driver_sql(
                    f"SET LOCAL lock_timeout = '{self.lock_timeout_ms}ms'")
                yield conn
            finally:
                tx.rollback()

    def health_check(self) -> ConnectionHealth:
        started = time.perf_counter()
        try:
            with self._readonly_connection() as conn:
                row = conn.execute(text(
                    "SELECT current_setting('server_version'), current_database(), "
                    "current_user, current_setting('transaction_read_only')"
                )).one()
        except Exception as exc:
            raise SourceUnavailableError("PostgreSQL source health check failed") from exc
        return ConnectionHealth(
            server_version=str(row[0]), current_database=str(row[1]),
            current_user=str(row[2]),
            transaction_read_only=str(row[3]).lower() in {"on", "true", "1"},
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def assert_read_only(self) -> None:
        health = self.health_check()
        if not health.transaction_read_only:
            raise ReadOnlyViolationError("source transaction is not read-only")

    def _estimated_count(self, conn: Connection, ref: TableRef) -> int:
        value = conn.execute(text(
            "SELECT COALESCE(c.reltuples, 0)::bigint "
            "FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=:schema AND c.relname=:table"
        ), {"schema": ref.schema, "table": ref.name}).scalar_one_or_none()
        return max(0, int(value or 0))

    def discover_schema(self) -> DatabaseSchema:
        tables: dict[str, TableSchema] = {}
        fks: list[ForeignKey] = []
        composites: list[CompositeKey] = []
        warnings: list[str] = []
        with self._readonly_connection() as conn:
            inspector = inspect(conn)
            for schema in self.allowed_schemas:
                for name in inspector.get_table_names(schema=schema):
                    try:
                        ref = self._resolve(TableRef(schema=schema, name=name))
                    except UnsafeIdentifierError:
                        continue
                    key = name if len(self.allowed_schemas) == 1 else f"{schema}.{name}"
                    cols = []
                    for c in inspector.get_columns(name, schema=schema):
                        cols.append(ColumnSchema(
                            name=c["name"], physical_type=str(c.get("type", "unknown")),
                            logical_type=(type(c.get("type")).__name__.lower()
                                          if c.get("type") is not None else None),
                            nullable=bool(c.get("nullable", True)),
                            default=(str(c["default"]) if c.get("default") is not None else None),
                        ))
                    pk_info = inspector.get_pk_constraint(name, schema=schema) or {}
                    pk_cols = list(pk_info.get("constrained_columns") or [])
                    if len(pk_cols) > 1:
                        composites.append(CompositeKey(table=key, columns=pk_cols))
                    uniques = [UniqueConstraint(
                        name=u.get("name"), columns=list(u.get("column_names") or []))
                        for u in inspector.get_unique_constraints(name, schema=schema)]
                    checks = [CheckConstraint(
                        name=c.get("name"), expression=str(c.get("sqltext") or ""))
                        for c in inspector.get_check_constraints(name, schema=schema)]
                    indexes = [IndexSchema(
                        name=i.get("name") or "unnamed_index",
                        columns=list(i.get("column_names") or []),
                        unique=bool(i.get("unique", False)),
                        expression=(str(i.get("expressions")) if i.get("expressions") else None),
                    ) for i in inspector.get_indexes(name, schema=schema)]
                    estimated = self._estimated_count(conn, ref)
                    tables[key] = TableSchema(
                        name=key, schema_name=schema,
                        primary_key=(pk_cols[0] if len(pk_cols) == 1 else None),
                        primary_key_columns=pk_cols, primary_key_confirmed=bool(pk_cols),
                        columns=cols, unique_constraints=uniques,
                        check_constraints=checks, indexes=indexes,
                        row_count=estimated, estimated_row_count=estimated,
                    )
                    for fk in inspector.get_foreign_keys(name, schema=schema):
                        child_cols = list(fk.get("constrained_columns") or [])
                        parent_cols = list(fk.get("referred_columns") or [])
                        parent_schema = fk.get("referred_schema") or schema
                        parent_name = fk.get("referred_table")
                        parent_key = (parent_name if len(self.allowed_schemas) == 1
                                      else f"{parent_schema}.{parent_name}")
                        if len(child_cols) != len(parent_cols) or not child_cols:
                            warnings.append(f"unsupported malformed FK {fk.get('name')} on {key}")
                            continue
                        fks.append(ForeignKey(
                            name=fk.get("name"), parent_table=parent_key,
                            parent_column=parent_cols[0], child_table=key,
                            child_column=child_cols[0], parent_columns=parent_cols,
                            child_columns=child_cols, confirmed=True,
                            evidence="declared_postgresql"))
        return DatabaseSchema(source_kind=self.kind, tables=tables,
                              foreign_keys=fks, composite_keys=composites,
                              warnings=warnings)

    def count_rows(self, table: str | TableRef) -> int:
        ref = self._resolve(table)
        with self._readonly_connection() as conn:
            return int(conn.exec_driver_sql(
                f"SELECT COUNT(*) FROM {self._qualified(ref)}").scalar_one())

    def sample_table(self, table: str | TableRef, max_rows: int, seed: int) -> pd.DataFrame:
        ref = self._resolve(table)
        limit = max(0, int(max_rows))
        with self._readonly_connection() as conn:
            inspector = inspect(conn)
            pk = inspector.get_pk_constraint(ref.name, schema=ref.schema) or {}
            pk_cols = list(pk.get("constrained_columns") or [])
            population = self._estimated_count(conn, ref)
            qualified = self._qualified(ref)
            if pk_cols:
                pieces = ", ".join(
                    f"COALESCE({self._quote(c)}::text, '')" for c in pk_cols)
                sql = text(
                    f"SELECT * FROM {qualified} "
                    f"ORDER BY hashtextextended(concat_ws('|', {pieces}), :seed) "
                    f"LIMIT :limit")
                frame = pd.read_sql_query(sql, conn, params={"seed": int(seed), "limit": limit})
                strategy, warning = "postgres_hash_pk", ""
            else:
                # SYSTEM percentage is bounded by LIMIT. A conservative percentage
                # avoids scanning the complete table while retaining reproducibility.
                pct = 100.0 if population <= limit or population <= 0 else min(
                    100.0, max(0.1, 150.0 * limit / population))
                sql = text(
                    f"SELECT * FROM {qualified} TABLESAMPLE SYSTEM (:pct) "
                    f"REPEATABLE (:seed) LIMIT :limit")
                frame = pd.read_sql_query(sql, conn, params={
                    "pct": pct, "seed": int(seed), "limit": limit})
                strategy = "postgres_tablesample"
                warning = "physical-page sampling may be biased; table has no stable primary key"
        key = ref.name if len(self.allowed_schemas) == 1 else ref.qualified_name
        self._sampling[key] = SamplingRecord(
            strategy=strategy, seed=seed, population_count=population,
            sample_count=len(frame), bias_warning=warning)
        return frame

    def last_sampling_record(self, table: str) -> SamplingRecord | None:
        return self._sampling.get(table)

    def close(self) -> None:
        self._engine.dispose()
