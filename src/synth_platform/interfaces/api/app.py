"""FastAPI application for the local API migration proof-of-concept."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import synth_platform
from synth_platform.application.services.transfer_service import TransferService
from synth_platform.application.workflows.schema_twin import (
    generate_from_schema,
    list_llm_text_columns,
    load_schema,
    load_schema_bytes,
    package_download,
    schema_column_details,
    summarize_schema,
    validation_highlights,
)
from synth_platform.domain.privacy.llm_policy import LlmPolicyError
from synth_platform.engine.documents.pdf.docling_engine import is_docling_available
from synth_platform.errors import TransferBlockedError
from synth_platform.infrastructure.persistence.platform_db import PlatformDB, get_platform_db
from synth_platform.infrastructure.persistence.project_views import (
    load_project_summaries,
    project_run_rows,
    read_product_settings,
    write_product_settings,
)
from synth_platform.interfaces.api.store import get_api_store

JOB_STATUSES = {"queued", "running", "succeeded", "failed"}


class WorkflowSessionRequest(BaseModel):
    intent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobRequest(BaseModel):
    kind: str = "manual"
    payload: dict[str, Any] = Field(default_factory=dict)


class SchemaSessionRequest(BaseModel):
    intent: str = "Development & testing"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SchemaGenerateRequest(BaseModel):
    row_count: int | None = None
    seed: int = 42
    locale: str = "en_US"
    export_format: str | None = None
    preview_rows: int | None = None
    llm_text_enabled: bool = False
    max_llm_rows: int = 50


class SettingsRequest(BaseModel):
    generation_mode: str | None = None
    default_record_count: int | None = None
    privacy_level: str | None = None
    default_output_format: str | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="Synth Platform API", version=getattr(synth_platform, "__version__", "0.0.0"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _register_error_handlers(app)
    _register_routes(app)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        code = str(detail.get("code") or "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": detail.get("message") or str(exc.detail)}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "Request validation failed", "details": exc.errors()}},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": str(exc)}},
        )


def _register_routes(app: FastAPI) -> None:
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return _ok(
            {
                "status": "ok",
                "version": getattr(synth_platform, "__version__", "0.0.0"),
                "feature_flags": {
                    "docling_available": is_docling_available(),
                    "auth_enabled": False,
                    "job_progress": "polling",
                },
            }
        )

    @app.post("/api/workflows/{workflow}/sessions", status_code=201)
    def create_workflow_session(workflow: str, body: WorkflowSessionRequest) -> dict[str, Any]:
        record = get_api_store().create_session(
            _normalize_workflow(workflow),
            {"intent": body.intent, "metadata": body.metadata},
        )
        return _ok(_session_response(record))

    @app.get("/api/workflows/{workflow}/sessions/{session_id}")
    def get_workflow_session(workflow: str, session_id: str) -> dict[str, Any]:
        record = _get_session(session_id)
        expected = _normalize_workflow(workflow)
        if record["workflow"] != expected:
            _raise(404, "not_found", f"Session {session_id!r} was not found for workflow {expected!r}.")
        return _ok(_session_response(record))

    @app.post("/api/jobs", status_code=201)
    def create_job(body: JobRequest) -> dict[str, Any]:
        record = get_api_store().create_job(kind=body.kind, payload=body.payload)
        return _ok(_job_response(record))

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return _ok(_job_response(_get_job(job_id)))

    @app.get("/api/downloads/{download_id}")
    def download(download_id: str) -> Response:
        record = _get_download(download_id)
        path = Path(record["path"])
        data = path.read_bytes() if path.is_file() else b""
        try:
            transfer = TransferService(transfer_recorder=get_platform_db()).downloadable_bytes(
                workflow=str(record["workflow"]),
                output_id=str(record["output_id"]),
                validation_report=record.get("validation_report"),
                data=data,
                metadata=dict(record.get("metadata") or {}),
            )
        except TransferBlockedError as exc:
            _raise(403, "transfer_blocked", str(exc))
        return Response(
            content=transfer.data,
            media_type=str(record.get("media_type") or "application/octet-stream"),
            headers={"Content-Disposition": f"attachment; filename=\"{record.get('filename') or download_id}\""},
        )

    @app.post("/api/schema/sessions", status_code=201)
    def create_schema_session(body: SchemaSessionRequest) -> dict[str, Any]:
        record = get_api_store().create_session(
            "schema_twin",
            {"intent": body.intent, "metadata": body.metadata, "stage": "intent"},
        )
        return _ok(_session_response(record))

    @app.post("/api/schema/sessions/{session_id}/schema-file")
    async def upload_schema_file(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        store = get_api_store()
        record = _get_schema_session(session_id)
        raw = await file.read()
        try:
            schema = load_schema_bytes(raw, file.filename or "schema.json")
        except Exception as exc:
            _raise(400, "schema_parse_error", str(exc))
        schema_blob = store.write_blob(file.filename or "schema.json", raw)
        summary = summarize_schema(schema)
        state = dict(record.get("state") or {})
        state.update(
            {
                "stage": "schema_review",
                "schema_file": {"filename": file.filename, "path": str(schema_blob), "size": len(raw)},
                "schema": schema.model_dump(mode="json"),
                "summary": summary.__dict__,
                "columns": schema_column_details(schema),
                "llm_text_columns": list_llm_text_columns(schema),
                "result_session_id": None,
            }
        )
        record["state"] = state
        record = store.save_session(record)
        return _ok(
            {
                "session": _session_response(record),
                "summary": state["summary"],
                "columns": state["columns"],
                "llm_text_columns": state["llm_text_columns"],
            }
        )

    @app.post("/api/schema/sessions/{session_id}/generate", status_code=202)
    def generate_schema(session_id: str, body: SchemaGenerateRequest) -> dict[str, Any]:
        record = _get_schema_session(session_id)
        if not (record.get("state") or {}).get("schema"):
            _raise(409, "schema_required", "Upload a schema file before starting generation.")
        job = get_api_store().create_job(
            kind="schema.generate",
            payload={"session_id": session_id, "request": body.model_dump(mode="json")},
        )
        threading.Thread(target=_run_schema_generation_job, args=(job["id"],), daemon=True).start()
        return _ok({"job": _job_response(job)})

    @app.get("/api/schema/sessions/{session_id}/preview")
    def schema_preview(session_id: str) -> dict[str, Any]:
        _get_schema_session(session_id)
        result = _get_schema_result(session_id)
        return _ok({"tables": result["preview_tables"], "row_counts": result["row_counts"]})

    @app.get("/api/schema/sessions/{session_id}/validation")
    def schema_validation(session_id: str) -> dict[str, Any]:
        _get_schema_session(session_id)
        result = _get_schema_result(session_id)
        return _ok({"report": result["validation_report"], "highlights": result["validation_highlights"]})

    @app.post("/api/schema/sessions/{session_id}/download", status_code=201)
    def create_schema_download(session_id: str) -> dict[str, Any]:
        session = _get_schema_session(session_id)
        result = _get_schema_result(session_id)
        summary = (session.get("state") or {}).get("summary") or {}
        base_name = str(summary.get("name") or "schema_twin").replace(" ", "_").lower()
        filename = f"{base_name}_synthetic.zip"
        download = get_api_store().create_download(
            {
                "workflow": "schema_twin",
                "session_id": session_id,
                "path": result["download_path"],
                "filename": filename,
                "media_type": "application/zip",
                "output_id": filename,
                "validation_report": result["validation_report"],
                "metadata": {"file_name": filename, "intent": (session.get("state") or {}).get("intent")},
            }
        )
        return _ok({"download_id": download["id"], "url": f"/api/downloads/{download['id']}"})

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return _ok(read_product_settings(get_platform_db()))

    @app.put("/api/settings")
    def put_settings(body: SettingsRequest) -> dict[str, Any]:
        db = get_platform_db()
        write_product_settings(db, body.model_dump(exclude_none=True))
        return _ok(read_product_settings(db))

    @app.get("/api/projects")
    def get_projects(workflow_type: str | None = None) -> dict[str, Any]:
        summaries = load_project_summaries(get_platform_db(), workflow_type=workflow_type)
        return _ok({"projects": [summary.__dict__ for summary in summaries]})

    @app.get("/api/projects/{project_id}/runs")
    def get_project_runs(project_id: str) -> dict[str, Any]:
        try:
            get_platform_db().get_project(project_id)
        except KeyError:
            _raise(404, "not_found", f"Project {project_id!r} was not found.")
        return _ok({"runs": project_run_rows(get_platform_db(), project_id)})


def _run_schema_generation_job(job_id: str) -> None:
    store = get_api_store()
    job = store.update_job(job_id, status="running", progress=["queued", "running"])
    session_id = str(job["payload"]["session_id"])
    request = dict(job["payload"].get("request") or {})
    try:
        session = store.get_session(session_id)
        schema_payload = (session.get("state") or {}).get("schema")
        if not isinstance(schema_payload, dict):
            raise ValueError("session does not contain an uploaded schema")
        store.append_progress(job_id, "loading schema")
        schema = load_schema(schema_payload)
        output_dir = store.root / "schema_outputs" / session_id / job_id
        store.append_progress(job_id, "running schema-driven pipeline")
        result = generate_from_schema(
            schema,
            row_count=request.get("row_count"),
            seed=int(request.get("seed") or 42),
            locale=str(request.get("locale") or "en_US"),
            output_dir=output_dir,
            export_format=request.get("export_format"),
            preview_rows=request.get("preview_rows"),
            llm_text_enabled=bool(request.get("llm_text_enabled")),
            max_llm_rows=int(request.get("max_llm_rows") or 50),
            history=PlatformDB(),
            product_settings=PlatformDB(),
        )
        store.append_progress(job_id, "packaging download")
        zip_path = store.write_blob(f"{session_id}.zip", package_download(result))
        result_payload = _schema_result_payload(result, zip_path)
        store.write_result(session_id, result_payload)
        state = dict(session.get("state") or {})
        state.update({"stage": "validation", "result_job_id": job_id})
        session["state"] = state
        store.save_session(session)
        store.update_job(job_id, status="succeeded", result={"session_id": session_id}, error=None)
    except LlmPolicyError as exc:
        store.update_job(
            job_id,
            status="failed",
            error={"code": "llm_policy_error", "message": str(exc)},
            result=None,
        )
    except Exception as exc:
        store.update_job(job_id, status="failed", error={"code": "generation_failed", "message": str(exc)}, result=None)


def _schema_result_payload(result, zip_path: Path) -> dict[str, Any]:
    return {
        "row_counts": dict(result.row_counts),
        "preview_tables": {
            name: frame.head(100).to_dict(orient="records") for name, frame in result.preview_tables.items()
        },
        "validation_report": result.validation_report,
        "validation_highlights": validation_highlights(result),
        "export_paths": {name: str(path) for name, path in result.export_paths.items()},
        "download_path": str(zip_path),
    }


def _ok(data: Any) -> dict[str, Any]:
    return {"data": data}


def _normalize_workflow(workflow: str) -> str:
    aliases = {"schema": "schema_twin", "database": "database_twin", "pdf": "pdf_twin", "interaction": "interaction_twin"}
    return aliases.get(workflow.strip().lower(), workflow.strip().lower())


def _session_response(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "workflow": record["workflow"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "state": record.get("state") or {},
    }


def _job_response(record: dict[str, Any]) -> dict[str, Any]:
    status = str(record.get("status"))
    if status not in JOB_STATUSES:
        status = "failed"
    return {
        "id": record["id"],
        "kind": record.get("kind"),
        "status": status,
        "progress": record.get("progress") or [],
        "error": record.get("error"),
        "result": record.get("result"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }


def _get_session(session_id: str) -> dict[str, Any]:
    try:
        return get_api_store().get_session(session_id)
    except KeyError:
        _raise(404, "not_found", f"Session {session_id!r} was not found.")


def _get_schema_session(session_id: str) -> dict[str, Any]:
    record = _get_session(session_id)
    if record["workflow"] != "schema_twin":
        _raise(404, "not_found", f"Schema session {session_id!r} was not found.")
    return record


def _get_job(job_id: str) -> dict[str, Any]:
    try:
        return get_api_store().get_job(job_id)
    except KeyError:
        _raise(404, "not_found", f"Job {job_id!r} was not found.")


def _get_download(download_id: str) -> dict[str, Any]:
    try:
        return get_api_store().get_download(download_id)
    except KeyError:
        _raise(404, "not_found", f"Download {download_id!r} was not found.")


def _get_schema_result(session_id: str) -> dict[str, Any]:
    try:
        return get_api_store().read_result(session_id)
    except KeyError:
        _raise(404, "not_found", f"No generated result exists for schema session {session_id!r}.")


def _raise(status_code: int, code: str, message: str) -> None:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


app = create_app()
