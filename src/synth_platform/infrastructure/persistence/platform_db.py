"""Local SQLite persistence for product projects, runs, and defaults."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from synth_platform.settings import Settings

WORKFLOW_TYPES = {"schema", "database", "pdf", "interaction"}
PROJECT_STATUSES = {"in_progress", "completed", "failed"}
RUN_STATUSES = {"in_progress", "completed", "failed"}
TRANSFER_STATUSES = {"not_attempted", "success", "blocked"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_project_name(workflow_type: str) -> str:
    labels = {
        "schema": "Schema Twin",
        "database": "Database Twin",
        "pdf": "PDF Twin",
        "interaction": "Customer Interaction Twin",
    }
    timestamp = datetime.now().astimezone().strftime("%b %d, %Y %I:%M %p")
    return f"{labels.get(workflow_type, workflow_type.title())} - {timestamp}"


def _json_dumps(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, sort_keys=True, default=str)


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    data = json.loads(value)
    return data if isinstance(data, dict) else {}


def _workflow_type(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "schema_twin": "schema",
        "database_twin": "database",
        "pdf_twin": "pdf",
        "interaction_twin": "interaction",
        "customer_interaction_twin": "interaction",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in WORKFLOW_TYPES:
        raise ValueError(f"unknown workflow type: {value!r}")
    return normalized


def platform_db_path(settings: Settings | None = None) -> Path:
    """Resolve the platform database path from config, with a test-friendly override."""
    override = os.environ.get("SP_PLATFORM_DB_PATH")
    if override:
        return Path(override)
    config = settings or Settings.from_env()
    return Path(config.staging_root) / "platform.db"


@dataclass(frozen=True)
class ProjectRecord:
    id: str
    name: str
    workflow_type: str
    created_at: str
    updated_at: str
    status: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RunRecord:
    id: str
    project_id: str
    workflow_type: str
    created_at: str
    updated_at: str
    status: str
    validation_status: str | None
    validation_passed: bool | None
    transfer_status: str
    transfer_allowed: bool | None
    transfer_attempted_at: str | None
    output_id: str | None
    metadata: dict[str, Any]


class PlatformDB:
    """Small sqlite3-backed data-access layer for local persistence."""

    def __init__(self, path: str | Path | None = None, *, settings: Settings | None = None):
        self.path = Path(path) if path is not None else platform_db_path(settings)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    workflow_type TEXT NOT NULL CHECK (workflow_type IN ('schema', 'database', 'pdf', 'interaction')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('in_progress', 'completed', 'failed')),
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    workflow_type TEXT NOT NULL CHECK (workflow_type IN ('schema', 'database', 'pdf', 'interaction')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('in_progress', 'completed', 'failed')),
                    validation_status TEXT,
                    validation_passed INTEGER CHECK (validation_passed IN (0, 1) OR validation_passed IS NULL),
                    transfer_status TEXT NOT NULL DEFAULT 'not_attempted'
                        CHECK (transfer_status IN ('not_attempted', 'success', 'blocked')),
                    transfer_allowed INTEGER CHECK (transfer_allowed IN (0, 1) OR transfer_allowed IS NULL),
                    transfer_attempted_at TEXT,
                    output_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_projects_workflow_created
                    ON projects(workflow_type, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_projects_status
                    ON projects(status);
                CREATE INDEX IF NOT EXISTS idx_runs_project_created
                    ON runs(project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_workflow_created
                    ON runs(workflow_type, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_output
                    ON runs(workflow_type, output_id);
                """
            )
            self._migrate_workflow_check_constraints(conn)

    def _migrate_workflow_check_constraints(self, conn: sqlite3.Connection) -> None:
        project_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'projects'"
        ).fetchone()
        run_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
        ).fetchone()
        if (
            project_sql is None
            or run_sql is None
            or "interaction" in str(project_sql["sql"])
            and "interaction" in str(run_sql["sql"])
        ):
            return

        conn.executescript(
            """
            PRAGMA foreign_keys = OFF;

            ALTER TABLE projects RENAME TO projects_legacy_workflow_check;
            ALTER TABLE runs RENAME TO runs_legacy_workflow_check;

            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                workflow_type TEXT NOT NULL CHECK (workflow_type IN ('schema', 'database', 'pdf', 'interaction')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('in_progress', 'completed', 'failed')),
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                workflow_type TEXT NOT NULL CHECK (workflow_type IN ('schema', 'database', 'pdf', 'interaction')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('in_progress', 'completed', 'failed')),
                validation_status TEXT,
                validation_passed INTEGER CHECK (validation_passed IN (0, 1) OR validation_passed IS NULL),
                transfer_status TEXT NOT NULL DEFAULT 'not_attempted'
                    CHECK (transfer_status IN ('not_attempted', 'success', 'blocked')),
                transfer_allowed INTEGER CHECK (transfer_allowed IN (0, 1) OR transfer_allowed IS NULL),
                transfer_attempted_at TEXT,
                output_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            INSERT INTO projects
                SELECT * FROM projects_legacy_workflow_check;
            INSERT INTO runs
                SELECT * FROM runs_legacy_workflow_check;

            DROP TABLE runs_legacy_workflow_check;
            DROP TABLE projects_legacy_workflow_check;

            CREATE INDEX IF NOT EXISTS idx_projects_workflow_created
                ON projects(workflow_type, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_projects_status
                ON projects(status);
            CREATE INDEX IF NOT EXISTS idx_runs_project_created
                ON runs(project_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_runs_workflow_created
                ON runs(workflow_type, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_runs_output
                ON runs(workflow_type, output_id);

            PRAGMA foreign_keys = ON;
            """
        )

    def create_project(
        self,
        *,
        name: str,
        workflow_type: str,
        status: str = "in_progress",
        metadata: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> ProjectRecord:
        workflow = _workflow_type(workflow_type)
        if status not in PROJECT_STATUSES:
            raise ValueError(f"unknown project status: {status!r}")
        now = _utc_now()
        pid = project_id or uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (id, name, workflow_type, created_at, updated_at, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (pid, name, workflow, now, now, status, _json_dumps(metadata)),
            )
        return self.get_project(pid)

    def get_project(self, project_id: str) -> ProjectRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"project not found: {project_id}")
        return self._project_from_row(row)

    def list_projects(self, *, workflow_type: str | None = None) -> list[ProjectRecord]:
        params: tuple[Any, ...] = ()
        query = "SELECT * FROM projects"
        if workflow_type is not None:
            query += " WHERE workflow_type = ?"
            params = (_workflow_type(workflow_type),)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            return [self._project_from_row(row) for row in conn.execute(query, params).fetchall()]

    def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProjectRecord:
        current = self.get_project(project_id)
        next_status = status or current.status
        if next_status not in PROJECT_STATUSES:
            raise ValueError(f"unknown project status: {next_status!r}")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE projects
                SET name = ?, status = ?, metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    name or current.name,
                    next_status,
                    _json_dumps(metadata if metadata is not None else current.metadata),
                    _utc_now(),
                    project_id,
                ),
            )
        return self.get_project(project_id)

    def record_run(
        self,
        *,
        workflow_type: str,
        project_id: str | None = None,
        project_name: str | None = None,
        status: str = "completed",
        validation_status: str | None = None,
        validation_passed: bool | None = None,
        output_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunRecord:
        workflow = _workflow_type(workflow_type)
        if status not in RUN_STATUSES:
            raise ValueError(f"unknown run status: {status!r}")
        if project_id is None:
            project = self.create_project(
                name=project_name or _default_project_name(workflow),
                workflow_type=workflow,
                status="completed" if status == "completed" else "failed" if status == "failed" else "in_progress",
                metadata={"created_by": "platform_db_record_run"},
            )
            project_id = project.id
        now = _utc_now()
        rid = run_id or uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, project_id, workflow_type, created_at, updated_at, status,
                    validation_status, validation_passed, transfer_status, transfer_allowed,
                    transfer_attempted_at, output_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'not_attempted', NULL, NULL, ?, ?)
                """,
                (
                    rid,
                    project_id,
                    workflow,
                    now,
                    now,
                    status,
                    validation_status,
                    None if validation_passed is None else int(validation_passed),
                    output_id,
                    _json_dumps(metadata),
                ),
            )
            conn.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                ("completed" if status == "completed" else "failed" if status == "failed" else "in_progress", now, project_id),
            )
        return self.get_run(rid)

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id}")
        return self._run_from_row(row)

    def list_runs(self, *, workflow_type: str | None = None, project_id: str | None = None) -> list[RunRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if workflow_type is not None:
            clauses.append("workflow_type = ?")
            params.append(_workflow_type(workflow_type))
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        query = "SELECT * FROM runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            return [self._run_from_row(row) for row in conn.execute(query, tuple(params)).fetchall()]

    def record_transfer_attempt(
        self,
        *,
        workflow: str,
        output_id: str,
        allowed: bool,
        validation_status: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord | None:
        workflow_type = _workflow_type(workflow)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM runs
                WHERE workflow_type = ? AND output_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (workflow_type, output_id),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    """
                    SELECT * FROM runs
                    WHERE workflow_type = ? AND transfer_status = 'not_attempted'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (workflow_type,),
                ).fetchone()
            if row is None:
                return None
            current = self._run_from_row(row)
            merged_metadata = {
                **current.metadata,
                "transfer": {
                    "reason": reason,
                    "metadata": metadata or {},
                },
            }
            now = _utc_now()
            conn.execute(
                """
                UPDATE runs
                SET transfer_status = ?, transfer_allowed = ?, transfer_attempted_at = ?,
                    validation_status = COALESCE(validation_status, ?),
                    metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "success" if allowed else "blocked",
                    int(allowed),
                    now,
                    validation_status,
                    _json_dumps(merged_metadata),
                    now,
                    current.id,
                ),
            )
            return self._run_from_row(conn.execute("SELECT * FROM runs WHERE id = ?", (current.id,)).fetchone())

    def write_setting(self, key: str, value: Any) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                """,
                (key, json.dumps(value, sort_keys=True, default=str), now),
            )

    def read_setting(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value_json"])

    def write_settings(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self.write_setting(key, value)

    def read_settings(self, keys: Iterable[str] | None = None) -> dict[str, Any]:
        if keys is None:
            with self._connect() as conn:
                rows = conn.execute("SELECT key, value_json FROM settings ORDER BY key").fetchall()
            return {row["key"]: json.loads(row["value_json"]) for row in rows}
        return {key: self.read_setting(key) for key in keys}

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            id=row["id"],
            name=row["name"],
            workflow_type=row["workflow_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            metadata=_json_loads(row["metadata_json"]),
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunRecord:
        validation_passed = row["validation_passed"]
        transfer_allowed = row["transfer_allowed"]
        return RunRecord(
            id=row["id"],
            project_id=row["project_id"],
            workflow_type=row["workflow_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            validation_status=row["validation_status"],
            validation_passed=None if validation_passed is None else bool(validation_passed),
            transfer_status=row["transfer_status"],
            transfer_allowed=None if transfer_allowed is None else bool(transfer_allowed),
            transfer_attempted_at=row["transfer_attempted_at"],
            output_id=row["output_id"],
            metadata=_json_loads(row["metadata_json"]),
        )


def get_platform_db() -> PlatformDB:
    return PlatformDB()


def record_run_best_effort(**kwargs: Any) -> RunRecord | None:
    try:
        return get_platform_db().record_run(**kwargs)
    except Exception:
        return None


def record_transfer_best_effort(**kwargs: Any) -> RunRecord | None:
    try:
        return get_platform_db().record_transfer_attempt(**kwargs)
    except Exception:
        return None
