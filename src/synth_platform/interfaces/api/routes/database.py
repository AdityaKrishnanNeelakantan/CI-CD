"""Database Twin API routes."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from synth_platform.application.workflows.database_twin import (
    DatabaseTwinPipelineConfig,
    RunManifest,
    SQLiteSourceAdapter,
    build_sample_database,
    run_database_twin_pipeline,
)
from synth_platform.domain.product_settings import read_generation_defaults
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
    target_record_count: int | None = None
    row_counts_by_table: dict[str, int] | None = None
    preserve_source_counts: bool | None = None
    scale_factor: float | None = Field(default=None, gt=0)
    sample_limit: int = 5000
    seed: int = 11


class SampleDatabaseRequest(BaseModel):
    customer_count: int = Field(400, ge=50, le=2000)
    seed: int = 42


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

    discovery = adapter.discover()
    source_summary = _source_summary_from_discovery(
        discovery,
        filename=file.filename or source_path.name,
        source_path=source_path,
        size=len(raw),
    )
    state = dict(record.get("state") or {})
    state.update(
        {
            "stage": "source_uploaded",
            "source": source_summary,
            "discovery": discovery,
        }
    )
    record["state"] = state
    return ok(session_response(store.save_session(record)))


@router.post("/sessions/{session_id}/sample-source")
def create_sample_database_source(session_id: str, body: SampleDatabaseRequest) -> dict[str, Any]:
    store = get_api_store()
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    source_path = store.blobs_dir / f"{uuid.uuid4().hex}_database_a.db"
    build_sample_database(source_path, seed=body.seed, customer_count=body.customer_count)

    adapter = SQLiteSourceAdapter({"path": str(source_path)})
    health = adapter.test_connection()
    if not health.get("healthy"):
        raise_api(400, "source_validation_error", str(health.get("error") or "Sample SQLite validation failed."))

    discovery = adapter.discover()
    source_summary = _source_summary_from_discovery(
        discovery,
        filename="database_a.db",
        source_path=source_path,
        size=source_path.stat().st_size,
    )
    source_summary["sample"] = {"customer_count": body.customer_count, "seed": body.seed}
    state = dict(record.get("state") or {})
    state.update(
        {
            "stage": "source_uploaded",
            "source": source_summary,
            "discovery": discovery,
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
    source = state.get("source") if isinstance(state.get("source"), dict) else {}
    table_names = _source_table_names(source or {})
    source_rows = _source_rows_by_table(source or {})
    target_count = body.target_record_count if body.target_record_count is not None else body.row_count
    if body.row_counts_by_table:
        config["row_counts_by_table"] = {
            table: max(1, int(count))
            for table, count in body.row_counts_by_table.items()
            if not table_names or table in table_names
        }
    elif body.preserve_source_counts:
        config["row_counts_by_table"] = {table: max(1, int(count)) for table, count in source_rows.items()}
        config["count_mode"] = "preserve_source_counts"
    elif body.scale_factor is not None:
        config["row_counts_by_table"] = {
            table: max(1, int(round(count * body.scale_factor)))
            for table, count in source_rows.items()
        }
        config["count_mode"] = "scale_factor"
    elif target_count is not None:
        config["target_record_count"] = max(1, int(target_count))
        config["row_counts_by_table"] = {table: max(1, int(target_count)) for table in table_names}
        config["count_mode"] = "target_record_count"
    else:
        config["row_counts_by_table"] = {table: max(1, int(count)) for table, count in source_rows.items()}
        config["count_mode"] = "preserve_source_counts"
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
        source_rows_by_table = _source_rows_by_table(source) if isinstance(source, dict) else {}
        row_counts = config_state.get("row_counts_by_table")
        if not isinstance(row_counts, dict) and source_rows_by_table:
            row_counts = {table: max(1, int(count)) for table, count in source_rows_by_table.items()}
            config_state["row_counts_by_table"] = row_counts
            config_state["count_mode"] = "preserve_source_counts"
        product_settings = PlatformDB()
        defaults = read_generation_defaults(product_settings)
        pipeline_config = DatabaseTwinPipelineConfig(
            dataset_id="database_twin",
            artifact_version="1.0.0",
            config_path="config/project.yaml",
            metadata_dir=metadata_dir,
            target_db_path=target_path,
            row_counts_by_table=row_counts if isinstance(row_counts, dict) else None,
            sample_limit=int(config_state.get("sample_limit") or 5000),
            num_rows_to_generate=int(config_state.get("target_record_count") or config_state.get("row_count") or defaults["default_record_count"]),
            seed=int(config_state.get("seed") or 11),
            model_type="safe_gaussian_copula",
            product_settings=product_settings,
        )
        store.advance_job(job_id, stage="generating_synthetic_data", percent=55, message="Generating synthetic data")
        context = run_database_twin_pipeline(adapter, manifest, pipeline_config, history=None)

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
        generated_rows_by_table = preview.get("row_counts") if isinstance(preview, dict) else {}
        qa_report = _with_database_count_validation(
            qa_report,
            source_rows_by_table=source_rows_by_table,
            generated_rows_by_table=generated_rows_by_table if isinstance(generated_rows_by_table, dict) else {},
        )
        source_metadata = _database_source_metadata(source, source_rows_by_table)
        generated_metadata = _database_generated_metadata(relational_report, generated_rows_by_table)
        bundle = store.create_result_bundle(
            session_id=session_id,
            workflow_type=WORKFLOW_TYPE,
            preview=preview,
            quality_report=qa_report,
            summary={
                "name": "database_twin",
                "workflow_type": WORKFLOW_TYPE,
                "generation_mode": "database_source_driven",
                "tables": list((relational_report.get("tables") or {}).keys()) if isinstance(relational_report, dict) else [],
                "source_table_count": len(source_rows_by_table),
                "generated_table_count": len(generated_rows_by_table or {}),
                "row_counts": generated_rows_by_table or {},
                "source_rows_by_table": source_rows_by_table,
                "generated_rows_by_table": generated_rows_by_table or {},
                "source_total_rows": sum(source_rows_by_table.values()),
                "generated_total_rows": sum((generated_rows_by_table or {}).values()),
                "count_mode": config_state.get("count_mode") or ("per_table" if row_counts else "default_record_count"),
                "source_metadata": source_metadata,
                "generated_metadata": generated_metadata,
                "validation": _database_validation_summary(qa_report),
                "target_database": str(target_path),
                "run_id": manifest.run_id,
            },
            artifacts=artifacts,
            metadata={
                "job_id": job_id,
                "manifest_run_id": manifest.run_id,
                "source": {
                    "filename": source.get("filename"),
                    "source_type": source.get("source_type"),
                    "path": source_path,
                    "tables": source.get("tables") or [],
                    "source_rows_by_table": source_rows_by_table,
                },
                "config": config_state,
            },
        )
        state.update(
            {
                "stage": "completed",
                "config": config_state,
                "result_id": bundle["id"],
                "result_job_id": job_id,
                "last_generated_rows_by_table": generated_rows_by_table or {},
            }
        )
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


def _source_summary_from_discovery(
    discovery: dict[str, Any],
    *,
    filename: str,
    source_path: Path,
    size: int,
) -> dict[str, Any]:
    tables = []
    warnings: list[str] = []
    for table_name, table in (discovery.get("tables") or {}).items():
        columns = table.get("columns") or []
        column_names = [
            str(column.get("name"))
            for column in columns
            if isinstance(column, dict) and column.get("name")
        ]
        row_count = int(table.get("estimated_row_count") or 0)
        tables.append(
            {
                "name": table_name,
                "row_count": row_count,
                "columns": column_names,
                "column_count": len(column_names),
                "primary_key": table.get("primary_key") or [],
                "foreign_keys": table.get("foreign_keys") or [],
            }
        )
        if not column_names:
            warnings.append(f"Table {table_name!r} has no discoverable columns.")
    return {
        "source_type": discovery.get("source_type") or "sqlite",
        "filename": filename,
        "path": str(source_path),
        "size": size,
        "tables": tables,
        "table_names": [table["name"] for table in tables],
        "source_rows_by_table": {table["name"]: table["row_count"] for table in tables},
        "total_rows": sum(table["row_count"] for table in tables),
        "source_fingerprint": discovery.get("source_fingerprint"),
        "warnings": warnings,
        "next_action": "configure_generation",
    }


def _source_table_names(source: dict[str, Any]) -> list[str]:
    table_names = source.get("table_names")
    if isinstance(table_names, list):
        return [str(table) for table in table_names]
    tables = source.get("tables")
    if not isinstance(tables, list):
        return []
    names: list[str] = []
    for table in tables:
        if isinstance(table, dict) and table.get("name"):
            names.append(str(table["name"]))
        elif isinstance(table, str):
            names.append(table)
    return names


def _source_rows_by_table(source: dict[str, Any]) -> dict[str, int]:
    source_rows = source.get("source_rows_by_table")
    if isinstance(source_rows, dict):
        return {str(table): int(count) for table, count in source_rows.items()}
    rows: dict[str, int] = {}
    tables = source.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if isinstance(table, dict) and table.get("name"):
                rows[str(table["name"])] = int(table.get("row_count") or 0)
    return rows


def _database_validation_summary(qa_report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(qa_report, dict):
        return {}
    integrity = (qa_report.get("report") or {}).get("integrity") or {}
    fk_validity = (integrity.get("fk_validity") or {}).get("overall_fk_validity")
    count_validation = qa_report.get("source_vs_generated_row_counts") or {}
    return {
        "hard_checks_passed": bool(qa_report.get("hard_checks_passed")),
        "fk_validity": fk_validity,
        "row_counts_match_source": count_validation.get("matches_source"),
        "release": qa_report.get("release") or {},
    }


def _with_database_count_validation(
    qa_report: dict[str, Any],
    *,
    source_rows_by_table: dict[str, int],
    generated_rows_by_table: dict[str, int],
) -> dict[str, Any]:
    if not isinstance(qa_report, dict):
        qa_report = {}
    exact_counts = {
        table: {
            "source_rows": int(source_rows_by_table.get(table, 0)),
            "generated_rows": int(generated_rows_by_table.get(table, 0)),
            "matches": int(source_rows_by_table.get(table, 0)) == int(generated_rows_by_table.get(table, 0)),
        }
        for table in sorted(set(source_rows_by_table) | set(generated_rows_by_table))
    }
    missing_source_tables = sorted(set(generated_rows_by_table) - set(source_rows_by_table))
    qa_report = dict(qa_report)
    qa_report["source_vs_generated_row_counts"] = {
        "matches_source": bool(exact_counts) and all(row["matches"] for row in exact_counts.values()),
        "tables": exact_counts,
        "source_total_rows": sum(source_rows_by_table.values()),
        "generated_total_rows": sum(generated_rows_by_table.values()),
        "warnings": [
            "No source table row-count metadata was available."
        ] if not source_rows_by_table else [
            f"Generated table {table!r} has no source row-count metadata."
            for table in missing_source_tables
        ],
    }
    return qa_report


def _database_source_metadata(source: dict[str, Any], source_rows_by_table: dict[str, int]) -> dict[str, Any]:
    return {
        "source_type": source.get("source_type") or "sqlite",
        "filename": source.get("filename"),
        "table_names": _source_table_names(source),
        "row_counts_by_table": source_rows_by_table,
        "total_rows": sum(source_rows_by_table.values()),
        "source_fingerprint": source.get("source_fingerprint"),
    }


def _database_generated_metadata(
    relational_report: dict[str, Any],
    generated_rows_by_table: dict[str, int] | Any,
) -> dict[str, Any]:
    row_counts = generated_rows_by_table if isinstance(generated_rows_by_table, dict) else {}
    tables = relational_report.get("tables") if isinstance(relational_report, dict) else {}
    return {
        "table_names": list(tables or row_counts),
        "row_counts_by_table": row_counts,
        "total_rows": sum(int(count) for count in row_counts.values()),
        "generation_order": relational_report.get("generation_order") if isinstance(relational_report, dict) else [],
    }


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
