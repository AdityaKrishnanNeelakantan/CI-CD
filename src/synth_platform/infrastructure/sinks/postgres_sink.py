"""Transactional PostgreSQL sink using run-scoped staging schemas."""
from __future__ import annotations

import re
import uuid

import pandas as pd
from sqlalchemy import create_engine, literal, text

from synth_platform.errors import SourceUnavailableError

_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresSink:
    def __init__(self, url: str, target_schema: str, if_exists: str = "fail"):
        if not _SAFE.fullmatch(target_schema):
            raise ValueError(f"unsafe target schema: {target_schema!r}")
        if not (target_schema.startswith("db_track_") or target_schema.startswith("pdf_track_")):
            raise ValueError("target schema must be run-scoped db_track_* or pdf_track_*")
        if if_exists not in {"fail", "replace_run_schema"}:
            raise ValueError("if_exists must be fail or replace_run_schema")
        try:
            self.engine = create_engine(url, pool_pre_ping=True)
        except (ImportError, ModuleNotFoundError) as exc:
            raise SourceUnavailableError(
                "PostgreSQL driver unavailable; install the postgresql optional dependency") from exc
        self.target_schema = target_schema
        self.if_exists = if_exists
        self.artifact = None
        self.conn = None
        self.tx = None
        self.staging_schema = None
        self._written: dict[str, int] = {}

    def configure_artifact(self, artifact) -> None:
        self.artifact = artifact

    def _q(self, value: str) -> str:
        return self.engine.dialect.identifier_preparer.quote(value)

    @staticmethod
    def _target_table(name: str) -> str:
        return name.replace(".", "__")


    @staticmethod
    def _safe_default(value: str | None) -> str:
        if value is None:
            return ""
        raw = str(value).strip()
        if re.fullmatch(r"[-+]?[0-9]+(?:\.[0-9]+)?", raw):
            return f" DEFAULT {raw}"
        if re.fullmatch(r"'(?:[^']|'')*'(?:::[A-Za-z0-9_ ]+)?", raw):
            return f" DEFAULT {raw}"
        if raw.upper() in {"CURRENT_DATE", "CURRENT_TIMESTAMP", "NOW()"}:
            return f" DEFAULT {raw}"
        return ""

    @staticmethod
    def _safe_check(expression: str) -> str:
        low = expression.lower()
        forbidden = (";", "--", "/*", "drop ", "alter ", "create ",
                     "insert ", "update ", "delete ", "execute ", "copy ")
        if any(token in low for token in forbidden):
            raise ValueError(f"unsafe CHECK expression: {expression!r}")
        return expression

    @staticmethod
    def _sql_type(physical: str) -> str:
        low = physical.lower()
        if any(token in low for token in ("int", "serial")):
            return "BIGINT"
        if any(token in low for token in ("real", "double", "float", "numeric", "decimal")):
            return "DOUBLE PRECISION"
        if "bool" in low:
            return "BOOLEAN"
        if "date" in low and "time" not in low:
            return "DATE"
        if any(token in low for token in ("timestamp", "datetime")):
            return "TIMESTAMP"
        if "json" in low:
            return "JSONB"
        return "TEXT"

    def begin(self, dataset_id: str) -> None:
        if self.artifact is None:
            raise RuntimeError("configure_artifact must be called before begin")
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", dataset_id)[-20:] or uuid.uuid4().hex[:8]
        self.staging_schema = f"_staging_{self.target_schema}_{suffix}"
        if len(self.staging_schema) > 63:
            self.staging_schema = self.staging_schema[:54] + "_" + uuid.uuid4().hex[:8]
        self.conn = self.engine.connect()
        self.tx = self.conn.begin()
        exists = self.conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name=:name)"
        ), {"name": self.target_schema}).scalar_one()
        if exists and self.if_exists == "fail":
            self.tx.rollback(); self.conn.close()
            self.tx = self.conn = None
            raise FileExistsError(f"target schema already exists: {self.target_schema}")
        self.conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {self._q(self.staging_schema)} CASCADE")
        self.conn.exec_driver_sql(f"CREATE SCHEMA {self._q(self.staging_schema)}")

    def write_table(self, table_name: str, frame: pd.DataFrame) -> None:
        if self.conn is None or self.staging_schema is None:
            raise RuntimeError("sink transaction not started")
        schema_table = self.artifact.schema_.tables[table_name]
        target_table = self._target_table(table_name)
        definitions = []
        for column in schema_table.columns:
            nullable = "" if column.nullable else " NOT NULL"
            default = self._safe_default(column.default)
            definitions.append(
                f"{self._q(column.name)} {self._sql_type(column.physical_type)}{default}{nullable}")
        ddl = (f"CREATE TABLE {self._q(self.staging_schema)}.{self._q(target_table)} "
               f"({', '.join(definitions)})")
        self.conn.exec_driver_sql(ddl)
        if len(frame):
            columns = list(schema_table.column_names)
            placeholders = ", ".join(f":v{i}" for i in range(len(columns)))
            col_sql = ", ".join(self._q(c) for c in columns)
            insert = text(
                f"INSERT INTO {self._q(self.staging_schema)}.{self._q(target_table)} "
                f"({col_sql}) VALUES ({placeholders})")
            rows = []
            for values in frame[columns].itertuples(index=False, name=None):
                row = {}
                for i, value in enumerate(values):
                    if pd.isna(value):
                        value = None
                    elif hasattr(value, "item"):
                        value = value.item()
                    row[f"v{i}"] = value
                rows.append(row)
            self.conn.execute(insert, rows)
        self._written[table_name] = len(frame)

    def _add_constraints(self) -> None:
        for table_name in self.artifact.relational_plan.order:
            table = self.artifact.schema_.tables[table_name]
            target = self._target_table(table_name)
            qualified = f"{self._q(self.staging_schema)}.{self._q(target)}"
            pk_cols = table.primary_key_columns or ([table.primary_key] if table.primary_key else [])
            if pk_cols:
                cols = ", ".join(self._q(c) for c in pk_cols)
                self.conn.exec_driver_sql(f"ALTER TABLE {qualified} ADD PRIMARY KEY ({cols})")
            for index, unique in enumerate(table.unique_constraints):
                if unique.columns:
                    cols = ", ".join(self._q(c) for c in unique.columns)
                    name = unique.name if unique.name and _SAFE.fullmatch(unique.name) else f"uq_{target}_{index}"
                    self.conn.exec_driver_sql(
                        f"ALTER TABLE {qualified} ADD CONSTRAINT {self._q(name)} UNIQUE ({cols})")
            for index, check in enumerate(table.check_constraints):
                expression = self._safe_check(check.expression)
                name = check.name if check.name and _SAFE.fullmatch(check.name) else f"ck_{target}_{index}"
                self.conn.exec_driver_sql(
                    f"ALTER TABLE {qualified} ADD CONSTRAINT {self._q(name)} CHECK ({expression})")
        # Add safe compiled range/enum constraints so target DDL enforces the
        # same rules as generation and validation.
        from synth_platform.domain.constraints.models import ConstraintKind
        for index, rule in enumerate(self.artifact.constraints.constraints):
            if rule.kind not in {ConstraintKind.RANGE, ConstraintKind.ENUM} or not rule.column:
                continue
            target = self._target_table(rule.table)
            qualified = f"{self._q(self.staging_schema)}.{self._q(target)}"
            if rule.kind == ConstraintKind.RANGE:
                clauses = []
                if rule.minimum is not None:
                    clauses.append(f"{self._q(rule.column)} >= {float(rule.minimum)}")
                if rule.maximum is not None:
                    clauses.append(f"{self._q(rule.column)} <= {float(rule.maximum)}")
                expression = " AND ".join(clauses)
            else:
                rendered = [str(literal(value).compile(
                    dialect=self.engine.dialect,
                    compile_kwargs={"literal_binds": True})) for value in rule.allowed_values]
                expression = f"{self._q(rule.column)} IN ({', '.join(rendered)})" if rendered else "TRUE"
            self.conn.exec_driver_sql(
                f"ALTER TABLE {qualified} ADD CONSTRAINT {self._q(f'compiled_ck_{index}')} CHECK ({expression})")
        for index, fk in enumerate(self.artifact.schema_.foreign_keys):
            child = self._target_table(fk.child_table)
            parent = self._target_table(fk.parent_table)
            child_cols = fk.child_columns or [fk.child_column]
            parent_cols = fk.parent_columns or [fk.parent_column]
            if len(child_cols) != len(parent_cols):
                raise ValueError(f"malformed FK {fk.name or index}")
            name = fk.name if fk.name and _SAFE.fullmatch(fk.name) else f"fk_{child}_{index}"
            child_sql = ", ".join(self._q(c) for c in child_cols)
            parent_sql = ", ".join(self._q(c) for c in parent_cols)
            self.conn.exec_driver_sql(
                f"ALTER TABLE {self._q(self.staging_schema)}.{self._q(child)} "
                f"ADD CONSTRAINT {self._q(name)} FOREIGN KEY ({child_sql}) "
                f"REFERENCES {self._q(self.staging_schema)}.{self._q(parent)} ({parent_sql})")

    def _verify(self) -> None:
        for table_name, expected in self._written.items():
            target = self._target_table(table_name)
            actual = self.conn.exec_driver_sql(
                f"SELECT COUNT(*) FROM {self._q(self.staging_schema)}.{self._q(target)}"
            ).scalar_one()
            if int(actual) != expected:
                raise RuntimeError(f"row count mismatch for {table_name}: {actual} != {expected}")
        for table_name in self.artifact.relational_plan.order:
            table = self.artifact.schema_.tables[table_name]
            key_cols = table.primary_key_columns or ([table.primary_key] if table.primary_key else [])
            if key_cols:
                target = self._target_table(table_name)
                group = ", ".join(self._q(c) for c in key_cols)
                duplicate_groups = self.conn.exec_driver_sql(
                    f"SELECT COUNT(*) FROM (SELECT {group}, COUNT(*) n "
                    f"FROM {self._q(self.staging_schema)}.{self._q(target)} "
                    f"GROUP BY {group} HAVING COUNT(*) > 1) d").scalar_one()
                if int(duplicate_groups):
                    raise RuntimeError(f"key uniqueness verification failed for {table_name}")
        for fk in self.artifact.schema_.foreign_keys:
            child = self._target_table(fk.child_table)
            parent = self._target_table(fk.parent_table)
            child_cols = fk.child_columns or [fk.child_column]
            parent_cols = fk.parent_columns or [fk.parent_column]
            joins = " AND ".join(
                f"c.{self._q(c)} = p.{self._q(p)}" for c, p in zip(child_cols, parent_cols))
            child_present = " AND ".join(f"c.{self._q(c)} IS NOT NULL" for c in child_cols)
            parent_missing = f"p.{self._q(parent_cols[0])} IS NULL"
            orphan_count = self.conn.exec_driver_sql(
                f"SELECT COUNT(*) FROM {self._q(self.staging_schema)}.{self._q(child)} c "
                f"LEFT JOIN {self._q(self.staging_schema)}.{self._q(parent)} p ON {joins} "
                f"WHERE {child_present} AND {parent_missing}"
            ).scalar_one()
            if int(orphan_count):
                raise RuntimeError(f"FK orphan verification failed for {fk.child_table}")

    def commit(self) -> None:
        try:
            self._add_constraints()
            self._verify()
            if self.if_exists == "replace_run_schema":
                self.conn.exec_driver_sql(
                    f"DROP SCHEMA IF EXISTS {self._q(self.target_schema)} CASCADE")
            self.conn.exec_driver_sql(
                f"ALTER SCHEMA {self._q(self.staging_schema)} RENAME TO {self._q(self.target_schema)}")
            self.tx.commit()
        finally:
            self.conn.close()
            self.conn = self.tx = None

    def rollback(self) -> None:
        staging = self.staging_schema
        if self.tx is not None:
            self.tx.rollback()
        if self.conn is not None:
            self.conn.close()
        self.conn = self.tx = None
        if staging:
            try:
                with self.engine.begin() as conn:
                    conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {self._q(staging)} CASCADE")
            except Exception:
                pass
