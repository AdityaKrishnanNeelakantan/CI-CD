"""FastAPI application for Synthetic Data Twin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
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
from synth_platform.engine.generation.schema.templates.library import list_templates, load_template
from synth_platform.errors import TransferBlockedError
from synth_platform.infrastructure.persistence.platform_db import PlatformDB, ProjectRecord, RunRecord, get_platform_db
from synth_platform.infrastructure.persistence.project_views import (
    WORKFLOW_LABELS,
    load_project_summaries,
    read_product_settings,
    run_result_id as recorded_result_id,
    write_product_settings,
)
from synth_platform.infrastructure.jobs.inline import LocalJobRunner
from synth_platform.interfaces.api.contracts import GenerationJob, ResultBundle, WorkflowSession
from synth_platform.interfaces.api.routes.database import router as database_router
from synth_platform.interfaces.api.routes.demo import router as demo_router
from synth_platform.interfaces.api.routes.document import router as document_router
from synth_platform.interfaces.api.routes.intent import router as intent_router
from synth_platform.interfaces.api.routes.interaction import router as interaction_router
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


class SchemaTemplateRequest(BaseModel):
    template_id: str


class SettingsRequest(BaseModel):
    generation_mode: str | None = None
    default_record_count: int | None = None
    privacy_level: str | None = None
    default_output_format: str | None = None


class ProjectUpdateRequest(BaseModel):
    name: str | None = None


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
    app.include_router(database_router)
    app.include_router(demo_router)
    app.include_router(document_router)
    app.include_router(intent_router)
    app.include_router(interaction_router)
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

    @app.get("/api/results/{result_id}")
    def get_result(result_id: str) -> dict[str, Any]:
        return _ok(_result_response(_get_result_bundle(result_id)))

    @app.get("/api/results/{result_id}/artifacts/{artifact_id}/download")
    def download_result_artifact(result_id: str, artifact_id: str) -> FileResponse:
        bundle = _get_result_bundle(result_id)
        artifact = _find_artifact(bundle, artifact_id)
        path = Path(str(artifact.get("path") or ""))
        if not path.is_file():
            _raise(410, "artifact_unavailable", f"Artifact {artifact_id!r} is no longer available.")
        filename = str(artifact.get("filename") or artifact.get("name") or path.name)
        content_type = str(artifact.get("content_type") or artifact.get("media_type") or "application/octet-stream")
        return FileResponse(path, media_type=content_type, filename=filename)

    @app.post("/api/results/{result_id}/save", status_code=201)
    def save_result_to_project(result_id: str) -> dict[str, Any]:
        store = get_api_store()
        bundle = _get_result_bundle(result_id)
        metadata = dict(bundle.get("metadata") or {})
        db = get_platform_db()
        saved_project_id = metadata.get("saved_project_id")
        saved_run_id = metadata.get("saved_run_id")
        if saved_project_id and saved_run_id:
            try:
                project = db.get_project(str(saved_project_id))
                run = db.get_run(str(saved_run_id))
                metadata.update({"project_id": project.id, "run_id": run.id})
                bundle["metadata"] = metadata
                store.save_result_bundle(bundle)
                return _ok({"project_id": project.id, "run_id": run.id, "project": project.__dict__, "saved": False})
            except KeyError:
                pass

        workflow_type = _project_workflow_type(str(bundle.get("workflow_type") or "schema_twin"))
        existing_run = _saved_run_for_result(db, workflow_type, result_id)
        if existing_run is not None:
            project = db.get_project(existing_run.project_id)
            metadata.update(
                {
                    "saved_project_id": project.id,
                    "saved_run_id": existing_run.id,
                    "project_id": project.id,
                    "run_id": existing_run.id,
                }
            )
            bundle["metadata"] = metadata
            store.save_result_bundle(bundle)
            return _ok({"project_id": project.id, "run_id": existing_run.id, "project": project.__dict__, "saved": False})

        name = _project_name_for_result(bundle)
        quality_report = bundle.get("quality_report") if isinstance(bundle.get("quality_report"), dict) else {}
        summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
        validation_passed = _validation_passed(quality_report)
        session_state = _session_state_for_bundle(bundle)
        run = db.record_run(
            workflow_type=workflow_type,
            project_name=name,
            status="completed" if str(bundle.get("status") or "available") == "available" else "in_progress",
            validation_status=str(quality_report.get("status") or quality_report.get("overall") or "") or None,
            validation_passed=validation_passed,
            output_id=result_id,
            metadata={
                "result_id": result_id,
                "session_id": bundle.get("session_id"),
                "job_id": metadata.get("job_id"),
                "row_counts": summary.get("row_counts"),
                "requested_row_count": summary.get("requested_row_count"),
                "row_count_mode": summary.get("row_count_mode"),
                "turn_count": summary.get("synthetic_turn_count"),
                "artifact_count": len(bundle.get("artifacts") or []),
                "input": _public_session_input(session_state),
                "config": _public_run_metadata(session_state.get("config") if isinstance(session_state.get("config"), dict) else {}),
            },
            run_id=f"result-{result_id}",
        )
        project = db.get_project(run.project_id)
        metadata.update({"saved_project_id": project.id, "saved_run_id": run.id, "project_id": project.id, "run_id": run.id})
        bundle["metadata"] = metadata
        store.save_result_bundle(bundle)
        return _ok({"project_id": project.id, "run_id": run.id, "project": project.__dict__, "saved": True})

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
        filename = file.filename or "schema.json"
        if Path(filename).suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            _raise(400, "wrong_workflow", "SQLite database files must be uploaded through Database Twin.")
        raw = await file.read()
        try:
            schema = load_schema_bytes(raw, filename)
        except Exception as exc:
            _raise(400, "schema_parse_error", str(exc))
        schema_blob = store.write_blob(filename, raw)
        summary = summarize_schema(schema)
        state = dict(record.get("state") or {})
        state.update(
            {
                "stage": "schema_review",
                "schema_file": {"filename": filename, "path": str(schema_blob), "size": len(raw)},
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

    @app.post("/api/schema/sessions/{session_id}/template")
    def attach_schema_template(session_id: str, body: SchemaTemplateRequest) -> dict[str, Any]:
        store = get_api_store()
        record = _get_schema_session(session_id)
        template = _get_template_metadata(body.template_id)
        if not template.get("can_generate"):
            _raise(409, "template_unavailable", f"Template {body.template_id!r} is not available for generation.")
        schema = load_template(body.template_id)
        summary = summarize_schema(schema)
        state = dict(record.get("state") or {})
        state.update(
            {
                "stage": "schema_review",
                "template": template,
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
        job = LocalJobRunner(get_api_store()).submit(
            kind="schema.generate",
            session_id=session_id,
            workflow_type="schema_twin",
            payload={"session_id": session_id, "request": body.model_dump(mode="json")},
            handler=_run_schema_generation_job,
        )
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
        db = get_platform_db()
        summaries = load_project_summaries(db, workflow_type=workflow_type)
        projects = []
        for summary in summaries:
            payload = dict(summary.__dict__)
            try:
                latest_run = db.get_run(summary.latest_run_id) if summary.latest_run_id else None
            except KeyError:
                latest_run = None
            payload["latest_result_id"] = _run_result_id(latest_run)
            projects.append(payload)
        return _ok({"projects": projects})

    @app.get("/api/projects/{project_id}")
    def get_project_detail(project_id: str) -> dict[str, Any]:
        db = get_platform_db()
        try:
            project = db.get_project(project_id)
        except KeyError:
            _raise(404, "not_found", f"Project {project_id!r} was not found.")
        return _ok({"project": _project_detail_response(db, project)})

    @app.get("/api/projects/{project_id}/runs")
    def get_project_runs(project_id: str) -> dict[str, Any]:
        db = get_platform_db()
        try:
            db.get_project(project_id)
        except KeyError:
            _raise(404, "not_found", f"Project {project_id!r} was not found.")
        return _ok({"runs": [_run_summary_response(run) for run in db.list_runs(project_id=project_id)]})

    @app.get("/api/projects/{project_id}/runs/{run_id}")
    def get_project_run_detail(project_id: str, run_id: str) -> dict[str, Any]:
        db = get_platform_db()
        try:
            run = db.get_run(run_id)
        except KeyError:
            _raise(404, "not_found", f"Run {run_id!r} was not found.")
        if run.project_id != project_id:
            _raise(404, "not_found", f"Run {run_id!r} was not found for project {project_id!r}.")
        return _ok({"run": _run_detail_response(run)})

    @app.get("/api/templates")
    def get_templates() -> dict[str, Any]:
        templates = [_get_template_metadata(template_id) for template_id in _all_template_ids()]
        return _ok({"templates": templates})

    @app.get("/api/templates/{template_id}")
    def get_template_detail(template_id: str) -> dict[str, Any]:
        return _ok({"template": _get_template_metadata(template_id, include_schema=True)})

    @app.patch("/api/projects/{project_id}")
    def update_project(project_id: str, body: ProjectUpdateRequest) -> dict[str, Any]:
        updates = body.model_dump(exclude_none=True)
        if not updates:
            _raise(422, "validation_error", "At least one project field must be provided.")
        try:
            project = get_platform_db().update_project(project_id, **updates)
        except KeyError:
            _raise(404, "not_found", f"Project {project_id!r} was not found.")
        return _ok(project.__dict__)


def _run_schema_generation_job(job_id: str) -> None:
    store = get_api_store()
    job = store.advance_job(job_id, status="running", stage="analyzing_input", percent=5, message="Analyzing input")
    session_id = str(job["payload"]["session_id"])
    request = dict(job["payload"].get("request") or {})
    try:
        session = store.get_session(session_id)
        schema_payload = (session.get("state") or {}).get("schema")
        if not isinstance(schema_payload, dict):
            raise ValueError("session does not contain an uploaded schema")
        store.advance_job(job_id, stage="learning_patterns", percent=20, message="Learning data patterns")
        schema = load_schema(schema_payload)
        output_dir = store.root / "schema_outputs" / session_id / job_id
        store.advance_job(job_id, stage="generating", percent=45, message="Generating synthetic data")
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
            history=None,
            product_settings=PlatformDB(),
        )
        store.advance_job(job_id, stage="validating", percent=75, message="Validating data quality")
        zip_path = store.write_blob(f"{session_id}.zip", package_download(result))
        store.advance_job(job_id, stage="preparing_files", percent=90, message="Preparing files")
        bundle = store.create_result_bundle(
            session_id=session_id,
            workflow_type="schema_twin",
            preview={
                "tables": {
                    name: frame.head(100).to_dict(orient="records") for name, frame in result.preview_tables.items()
                },
                "row_counts": dict(result.row_counts),
            },
            quality_report=result.validation_report,
            summary={
                "row_counts": dict(result.row_counts),
                "requested_row_count": request.get("row_count"),
                "row_count_mode": "fk_aware",
                "tables": list(result.preview_tables.keys()),
                "export_count": len(result.export_paths),
            },
            artifacts=[
                {
                    "id": f"{session_id}-download",
                    "name": f"{session_id}.zip",
                    "path": str(zip_path),
                    "media_type": "application/zip",
                    "size": zip_path.stat().st_size if zip_path.is_file() else None,
                    "role": "download",
                    "metadata": {"format": "zip"},
                },
                *[
                    {
                        "id": f"{session_id}-{name}",
                        "name": Path(path).name,
                        "path": str(path),
                        "media_type": "text/csv" if Path(path).suffix.lower() == ".csv" else "application/octet-stream",
                        "size": Path(path).stat().st_size if Path(path).is_file() else None,
                        "role": "table_export",
                        "metadata": {"table": name},
                    }
                    for name, path in result.export_paths.items()
                ],
            ],
            metadata={"job_id": job_id},
        )
        result_payload = _schema_result_payload(result, zip_path, result_id=str(bundle["id"]))
        store.write_result(session_id, result_payload)
        state = dict(session.get("state") or {})
        state.update({"stage": "validation", "result_job_id": job_id, "result_id": bundle["id"]})
        session["state"] = state
        store.save_session(session)
        store.update_job(
            job_id,
            status="succeeded",
            stage="complete",
            percent=100.0,
            message="Generation complete",
            result={"session_id": session_id, "result_id": bundle["id"]},
            result_id=bundle["id"],
            error=None,
        )
    except LlmPolicyError as exc:
        store.update_job(
            job_id,
            status="failed",
            stage="failed",
            percent=100.0,
            message="Generation failed",
            error={"code": "llm_policy_error", "message": str(exc)},
            result=None,
        )
    except Exception as exc:
        store.update_job(
            job_id,
            status="failed",
            stage="failed",
            percent=100.0,
            message="Generation failed",
            error={"code": "generation_failed", "message": str(exc)},
            result=None,
        )


def _schema_result_payload(result, zip_path: Path, *, result_id: str | None = None) -> dict[str, Any]:
    return {
        "result_id": result_id,
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
    return WorkflowSession(
        id=record["id"],
        workflow=record["workflow"],
        workflow_type=record["workflow"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        state=record.get("state") or {},
    ).model_dump(mode="json")


def _job_response(record: dict[str, Any]) -> dict[str, Any]:
    status = str(record.get("status"))
    if status not in JOB_STATUSES:
        status = "failed"
    progress = list(record.get("progress") or [])
    message = str(record.get("message") or (progress[-1] if progress else status))
    percent = float(record.get("percent") or (100.0 if status in {"succeeded", "failed"} else 0.0))
    return GenerationJob(
        id=record["id"],
        job_id=record["id"],
        session_id=record.get("session_id") or (record.get("payload") or {}).get("session_id"),
        workflow_type=record.get("workflow_type"),
        kind=record.get("kind"),
        status=status,
        stage=str(record.get("stage") or status),
        percent=max(0.0, min(100.0, percent)),
        message=message,
        progress=progress,
        error=record.get("error"),
        result_id=record.get("result_id"),
        result=record.get("result"),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
    ).model_dump(mode="json")


def _result_response(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    result_id = str(normalized.get("result_id") or normalized.get("id"))
    artifacts = [_artifact_response(result_id, artifact) for artifact in normalized.get("artifacts") or []]
    normalized["artifacts"] = artifacts
    return ResultBundle.model_validate(normalized).model_dump(mode="json")


def _project_detail_response(db: PlatformDB, project: ProjectRecord) -> dict[str, Any]:
    runs = db.list_runs(project_id=project.id)
    latest_run = runs[0] if runs else None
    latest_result_id = _run_result_id(latest_run) if latest_run else project.metadata.get("latest_result_id")
    latest_result = _optional_result_response(str(latest_result_id)) if latest_result_id else None
    artifacts = latest_result.get("artifacts") if isinstance(latest_result, dict) else None
    return {
        "project_id": project.id,
        "id": project.id,
        "name": project.name,
        "workflow_type": project.workflow_type,
        "workflow_label": WORKFLOW_LABELS.get(project.workflow_type, project.workflow_type.replace("_", " ").title()),
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "status": project.status,
        "records": _records_label(latest_run),
        "validation": _validation_label(latest_run),
        "transfer": _transfer_label(latest_run),
        "run_count": len(runs),
        "latest_run_id": latest_run.id if latest_run else None,
        "latest_result_id": latest_result_id,
        "metadata": project.metadata,
        "summary": latest_result.get("summary") if isinstance(latest_result, dict) else None,
        "quality_report": latest_result.get("quality_report") if isinstance(latest_result, dict) else None,
        "artifacts": artifacts,
        "runs": [_run_summary_response(run) for run in runs],
    }


def _run_summary_response(run: RunRecord) -> dict[str, Any]:
    result_id = _run_result_id(run)
    return {
        "run_id": run.id,
        "id": run.id,
        "project_id": run.project_id,
        "workflow_type": run.workflow_type,
        "workflow_label": WORKFLOW_LABELS.get(run.workflow_type, run.workflow_type.replace("_", " ").title()),
        "status": run.status,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "completed_at": run.metadata.get("completed_at"),
        "result_id": result_id,
        "job_id": run.metadata.get("job_id"),
        "validation_status": run.validation_status,
        "validation_passed": run.validation_passed,
        "validation": _validation_label(run),
        "transfer_status": run.transfer_status,
        "transfer": _transfer_label(run),
        "transfer_allowed": run.transfer_allowed,
        "transfer_attempted_at": run.transfer_attempted_at,
        "records": _records_label(run),
        "metadata": _public_run_metadata(run.metadata),
    }


def _run_detail_response(run: RunRecord) -> dict[str, Any]:
    result_id = _run_result_id(run)
    result = _optional_result_response(str(result_id)) if result_id else None
    return {
        **_run_summary_response(run),
        "summary": result.get("summary") if isinstance(result, dict) else run.metadata.get("summary"),
        "quality_report": result.get("quality_report") if isinstance(result, dict) else run.metadata.get("quality_report"),
        "artifacts": result.get("artifacts") if isinstance(result, dict) else [],
        "input": _public_run_metadata(run.metadata.get("input") if isinstance(run.metadata.get("input"), dict) else {}),
        "config": _public_run_metadata(run.metadata.get("config") if isinstance(run.metadata.get("config"), dict) else {}),
    }


def _run_result_id(run: RunRecord | None) -> str | None:
    if run is None:
        return None
    candidates = [recorded_result_id(run), run.output_id]
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        try:
            get_api_store().get_result_bundle(candidate)
        except KeyError:
            continue
        return candidate
    return None


def _public_run_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"path", "download_path"} or key.endswith("_path"):
            continue
        if isinstance(value, dict):
            safe[key] = _public_run_metadata(value)
        else:
            safe[key] = value
    return safe


def _records_label(run: RunRecord | None) -> str:
    if run is None:
        return "-"
    for key in ("row_counts", "row_counts_by_table"):
        counts = run.metadata.get(key)
        if isinstance(counts, dict):
            numeric = [int(value) for value in counts.values() if isinstance(value, int | float)]
            if numeric:
                total = sum(numeric)
                requested = run.metadata.get("requested_row_count")
                row_count_mode = run.metadata.get("row_count_mode")
                if isinstance(requested, int | float) and row_count_mode == "per_table":
                    return f"{total:,} total / {int(requested):,} per table"
                unique_counts = set(numeric)
                if len(unique_counts) == 1 and len(numeric) > 1:
                    return f"{total:,} total / {numeric[0]:,} per table"
                return f"{total:,} total"
    turn_count = run.metadata.get("turn_count")
    if isinstance(turn_count, int | float):
        return f"{int(turn_count):,}"
    return "-"


def _validation_label(run: RunRecord | None) -> str:
    if run is None:
        return "-"
    if run.validation_passed is True:
        return "Passed"
    if run.validation_passed is False:
        return "Failed"
    return run.validation_status.replace("_", " ").title() if run.validation_status else "-"


def _transfer_label(run: RunRecord | None) -> str:
    if run is None:
        return "-"
    if run.transfer_status == "success":
        return "Allowed"
    if run.transfer_status == "blocked":
        return "Blocked"
    return "Not attempted"


def _session_state_for_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    session_id = bundle.get("session_id")
    if not session_id:
        return {}
    try:
        session = get_api_store().get_session(str(session_id))
    except KeyError:
        return {}
    state = session.get("state")
    return state if isinstance(state, dict) else {}


def _public_session_input(state: dict[str, Any]) -> dict[str, Any]:
    for key in ("schema_file", "template", "source", "document", "transcript"):
        value = state.get(key)
        if isinstance(value, dict):
            return {key: _public_run_metadata(value)}
    return {}


def _optional_result_response(result_id: str) -> dict[str, Any] | None:
    try:
        return _result_response(get_api_store().get_result_bundle(result_id))
    except KeyError:
        return None


def _artifact_response(result_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
    item = dict(artifact)
    artifact_id = str(item.get("artifact_id") or item.get("id") or item.get("name"))
    filename = str(item.get("filename") or item.get("name") or artifact_id)
    content_type = str(item.get("content_type") or item.get("media_type") or "application/octet-stream")
    size = item.get("size_bytes", item.get("size"))
    path = Path(str(item.get("path") or ""))
    downloadable = bool(item.get("downloadable", path.is_file()))
    item.update(
        {
            "id": artifact_id,
            "artifact_id": artifact_id,
            "name": str(item.get("name") or filename),
            "filename": filename,
            "media_type": content_type,
            "content_type": content_type,
            "size": size,
            "size_bytes": size,
            "role": item.get("role") or item.get("kind"),
            "kind": item.get("kind") or item.get("role"),
            "downloadable": downloadable,
            "download_url": f"/api/results/{result_id}/artifacts/{artifact_id}/download" if downloadable else None,
            "metadata": item.get("metadata") or {},
        }
    )
    return item


def _find_artifact(bundle: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    for artifact in bundle.get("artifacts") or []:
        normalized_id = str(artifact.get("artifact_id") or artifact.get("id") or artifact.get("name"))
        if normalized_id == artifact_id:
            return artifact
    _raise(404, "not_found", f"Artifact {artifact_id!r} was not found for result {bundle.get('id')!r}.")


def _project_workflow_type(workflow_type: str) -> str:
    aliases = {
        "schema_twin": "schema",
        "database_twin": "database",
        "pdf_twin": "pdf",
        "interaction_twin": "interaction",
    }
    return aliases.get(workflow_type, workflow_type)


def _saved_run_for_result(db: PlatformDB, workflow_type: str, result_id: str):
    for run in db.list_runs(workflow_type=workflow_type):
        if run.output_id == result_id or run.metadata.get("result_id") == result_id:
            return run
    return None


def _project_name_for_result(bundle: dict[str, Any]) -> str:
    labels = {
        "schema_twin": "Schema Twin",
        "database_twin": "Database Twin",
        "pdf_twin": "Document Twin",
        "interaction_twin": "Customer Interaction Twin",
    }
    workflow_type = str(bundle.get("workflow_type") or "schema_twin")
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    explicit = summary.get("name") or summary.get("doc_id")
    return str(explicit or f"{labels.get(workflow_type, workflow_type)} Result")


def _all_template_ids() -> list[str]:
    return [*list_templates(), "sqlite-customer-360", "document-bank-statement", "support-transcript"]


def _get_template_metadata(template_id: str, *, include_schema: bool = False) -> dict[str, Any]:
    schema_ids = set(list_templates())
    static: dict[str, dict[str, Any]] = {
        "sqlite-customer-360": {
            "template_id": "sqlite-customer-360",
            "name": "SQLite Customer 360",
            "description": "Database Twin project shape for customer, account, and event tables.",
            "workflow_type": "database_twin",
            "category": "database",
            "supported_formats": ["sqlite"],
            "status": "coming_soon",
            "can_generate": False,
            "fields": [],
        },
        "document-bank-statement": {
            "template_id": "document-bank-statement",
            "name": "Bank Statement PDF",
            "description": "Document Twin template metadata for bank statement extraction and rendering.",
            "workflow_type": "pdf_twin",
            "category": "document",
            "supported_formats": ["pdf"],
            "status": "coming_soon",
            "can_generate": False,
            "fields": [],
        },
        "support-transcript": {
            "template_id": "support-transcript",
            "name": "Support Transcript",
            "description": "Interaction Twin transcript shape for support chats and logs.",
            "workflow_type": "interaction_twin",
            "category": "interaction",
            "supported_formats": ["txt", "log"],
            "status": "coming_soon",
            "can_generate": False,
            "fields": [],
        },
    }
    if template_id in static:
        return dict(static[template_id])
    if template_id not in schema_ids:
        _raise(404, "not_found", f"Template {template_id!r} was not found.")
    schema = load_template(template_id)
    summary = summarize_schema(schema)
    metadata = {
        "template_id": template_id,
        "name": schema.name,
        "description": schema.description or f"{schema.name} schema template",
        "workflow_type": "schema_twin",
        "category": schema.domain or template_id,
        "supported_formats": ["json", "csv", "parquet"],
        "status": "available",
        "can_generate": True,
        "fields": schema_column_details(schema)[:80],
        "schema_preview": summary.__dict__,
    }
    if include_schema:
        metadata["schema"] = schema.model_dump(mode="json")
    return metadata


def _validation_passed(report: dict[str, Any]) -> bool | None:
    if "passed" in report:
        return bool(report["passed"])
    if "hard_checks_passed" in report:
        return bool(report["hard_checks_passed"])
    status = str(report.get("status") or report.get("overall") or "").lower()
    if status:
        return status in {"pass", "passed", "success", "succeeded"}
    return None


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


def _get_result_bundle(result_id: str) -> dict[str, Any]:
    try:
        return get_api_store().get_result_bundle(result_id)
    except KeyError:
        _raise(404, "not_found", f"Result {result_id!r} was not found.")


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
