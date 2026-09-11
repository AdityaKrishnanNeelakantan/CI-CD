"""Database Twin API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from synth_platform.application.workflows.database_twin import (
    DatabaseTwinPipelineConfig,
    RunManifest,
    SQLiteSourceAdapter,
    run_database_twin_pipeline,
)
from synth_platform.infrastructure.jobs.inline import LocalJobRunner
from synth_platform.infrastructure.persistence.platform_db import PlatformDB
from synth_platform.interfaces.api.routes.shared import (
    get_workflow_session_or_404,
    job_response,
    ok,
    raise_api,
    session_response,
)
from synth_platform.interfaces.api.store import get_api_store

router = APIRouter(prefix="/api/database", tags=["database"])
WORKFLOW_TYPE = "database_twin"


class DatabaseSessionRequest(BaseModel):
    intent: str = "Development & testing"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatabaseConfigureRequest(BaseModel):
    row_count: int | None = None
    row_counts_by_table: dict[str, int] | None = None
    sample_limit: int = 100
    seed: int = 11
    model_type: str = "safe_gaussian_copula"


@router.post("/sessions", status_code=201)
def create_database_session(body: DatabaseSessionRequest) -> dict[str, Any]:
    record = get_api_store().create_session(
        WORKFLOW_TYPE,
        {"intent": body.intent, "metadata": body.metadata, "stage": "created"},
    )
    return ok(session_response(record))


@router.get("/sessions/{session_id}")
def get_database_session(session_id: str) -> dict[str, Any]:
    return ok(session_response(get_workflow_session_or_404(session_id, WORKFLOW_TYPE)))


@router.post("/sessions/{session_id}/source")
async def upload_database_source(
    session_id: str,
    source_type: str = Form("sqlite"),
    file: UploadFile | None = File(None),
) -> dict[str, Any]:
    store = get_api_store()
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    normalized_source_type = source_type.strip().lower()
    if normalized_source_type != "sqlite":
        raise_api(400, "unsupported_source_type", "Database Twin API currently supports SQLite uploads only.")
    if file is None:
        raise_api(422, "validation_error", "Upload a SQLite database file.")
    raw = await file.read()
    if not raw:
        raise_api(400, "empty_upload", "Uploaded database file is empty.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".db", ".sqlite", ".sqlite3"}:
        raise_api(400, "unsupported_file_type", "Upload a .db, .sqlite, or .sqlite3 file.")

    source_path = store.write_blob(file.filename or "database.sqlite", raw)
    adapter = SQLiteSourceAdapter({"path": str(source_path)})
    health = adapter.test_connection()
    if not health.get("healthy"):
        raise_api(400, "source_validation_error", str(health.get("error") or "SQLite validation failed."))

    tables = adapter.list_tables()
    state = dict(record.get("state") or {})
    state.update(
        {
            "stage": "source_uploaded",
            "source": {
                "source_type": normalized_source_type,
                "filename": file.filename,
                "path": str(source_path),
                "size": len(raw),
                "tables": tables,
            },
        }
    )
    record["state"] = state
    return ok(session_response(store.save_session(record)))


@router.post("/sessions/{session_id}/configure")
def configure_database_session(session_id: str, body: DatabaseConfigureRequest) -> dict[str, Any]:
    store = get_api_store()
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    state = dict(record.get("state") or {})
    config = body.model_dump(mode="json", exclude_none=True)
    if body.row_count is not None and not body.row_counts_by_table:
        tables = ((state.get("source") or {}).get("tables") or [])
        config["row_counts_by_table"] = {table: max(1, int(body.row_count)) for table in tables}
    state.update({"stage": "configured", "config": config})
    record["state"] = state
    return ok(session_response(store.save_session(record)))


@router.post("/sessions/{session_id}/generate", status_code=202)
def generate_database(session_id: str) -> dict[str, Any]:
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    source = (record.get("state") or {}).get("source")
    if not isinstance(source, dict):
        raise_api(409, "source_required", "Upload a SQLite source before starting Database Twin generation.")
    job = LocalJobRunner(get_api_store()).submit(
        kind="database.generate",
        session_id=session_id,
        workflow_type=WORKFLOW_TYPE,
        payload={"session_id": session_id},
        handler=_run_database_job,
    )
    return ok({"job": job_response(job)})


def _run_database_job(job_id: str) -> None:
    store = get_api_store()
    job = store.advance_job(job_id, status="running", stage="analyzing_input", percent=5, message="Analyzing input")
    session_id = str(job["payload"]["session_id"])
    try:
        session = store.get_session(session_id)
        state = dict(session.get("state") or {})
        source = state.get("source") or {}
        source_path = source.get("path")
        if not source_path:
            raise ValueError("Database source is missing.")

        config_state = dict(state.get("config") or {})
        workdir = store.root / "database_outputs" / session_id / job_id
        manifest = RunManifest.create(runs_dir=workdir / "runs")
        metadata_dir = workdir / "metadata"
        target_path = workdir / "database_b.db"

        store.advance_job(job_id, stage="learning_patterns", percent=25, message="Learning data patterns")
        adapter = SQLiteSourceAdapter({"path": str(source_path)})
        row_counts = config_state.get("row_counts_by_table")
        pipeline_config = DatabaseTwinPipelineConfig(
            dataset_id="database_twin",
            artifact_version="1.0.0",
            config_path="config/project.yaml",
            metadata_dir=metadata_dir,
            target_db_path=target_path,
            row_counts_by_table=row_counts if isinstance(row_counts, dict) else None,
            sample_limit=int(config_state.get("sample_limit") or 100),
            num_rows_to_generate=int(config_state.get("row_count") or 10),
            seed=int(config_state.get("seed") or 11),
            model_type=str(config_state.get("model_type") or "safe_gaussian_copula"),
        )
        store.advance_job(job_id, stage="generating_synthetic_data", percent=55, message="Generating synthetic data")
        context = run_database_twin_pipeline(adapter, manifest, pipeline_config, history=PlatformDB())

        failed = [stage for stage in manifest.stages if stage.get("status") != "success"]
        if failed:
            raise RuntimeError(f"Database Twin pipeline failed at {failed[-1].get('stage_name')}: {failed[-1].get('errors')}")

        store.advance_job(job_id, stage="validating_quality", percent=80, message="Validating data quality")
        qa_report = context.artifacts.get("qa_report") or _read_json_if_exists(manifest.output_path("qa_report.json"))
        relational_report = context.artifacts.get("relational_report") or _read_json_if_exists(
            manifest.output_path("relational_generation_report.json")
        )
        store.advance_job(job_id, stage="preparing_files", percent=92, message="Preparing files")
        artifacts = _database_artifacts(manifest.run_dir, target_path, relational_report)
        preview = _database_preview(relational_report)
        bundle = store.create_result_bundle(
            session_id=session_id,
            workflow_type=WORKFLOW_TYPE,
            preview=preview,
            quality_report=qa_report,
            summary={
                "tables": list((relational_report.get("tables") or {}).keys()) if isinstance(relational_report, dict) else [],
                "target_database": str(target_path),
                "run_id": manifest.run_id,
            },
            artifacts=artifacts,
            metadata={"job_id": job_id, "manifest_run_id": manifest.run_id},
        )
        state.update({"stage": "completed", "result_id": bundle["id"], "result_job_id": job_id})
        session["state"] = state
        store.save_session(session)
        store.update_job(
            job_id,
            status="succeeded",
            stage="completed",
            percent=100.0,
            message="Generation complete",
            result_id=bundle["id"],
            result={"session_id": session_id, "result_id": bundle["id"]},
            error=None,
        )
    except Exception as exc:
        store.update_job(
            job_id,
            status="failed",
            stage="failed",
            percent=100.0,
            message="Generation failed",
            error={"code": "database_generation_failed", "message": str(exc)},
            result=None,
        )


def _database_artifacts(run_dir: Path, target_path: Path, relational_report: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if target_path.is_file():
        artifacts.append(_artifact(target_path, role="target_database", media_type="application/x-sqlite3"))
    if isinstance(relational_report, dict):
        for table, data in (relational_report.get("tables") or {}).items():
            path = Path(str(data.get("path") or ""))
            if path.is_file():
                artifacts.append(_artifact(path, role="table_export", metadata={"table": table}))
    for path in sorted(run_dir.rglob("*.json")):
        artifacts.append(_artifact(path, role="report", media_type="application/json"))
    return artifacts


def _database_preview(relational_report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(relational_report, dict):
        return {}
    previews: dict[str, list[dict[str, Any]]] = {}
    row_counts: dict[str, int] = {}
    for table, data in (relational_report.get("tables") or {}).items():
        row_counts[table] = int(data.get("row_count") or 0)
        path = Path(str(data.get("path") or ""))
        if path.is_file():
            import pandas as pd

            previews[table] = pd.read_csv(path, nrows=25).to_dict(orient="records")
    return {"tables": previews, "row_counts": row_counts}


def _artifact(path: Path, *, role: str, media_type: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    suffix = path.suffix.lower()
    resolved_media_type = media_type or ("text/csv" if suffix == ".csv" else "application/octet-stream")
    return {
        "id": path.stem,
        "name": path.name,
        "path": str(path),
        "media_type": resolved_media_type,
        "size": path.stat().st_size if path.is_file() else None,
        "role": role,
        "metadata": metadata or {},
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}

