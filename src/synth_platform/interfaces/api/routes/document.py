"""Document Twin API routes backed by the existing PDF Twin internals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from synth_platform.application.workflows.pdf_twin import (
    DOCUMENT_PROFILE_FILENAME,
    PDFDocumentAdapter,
    RunManifest,
    load_document_binding_map,
    load_document_ground_truth,
    load_document_profile,
    load_document_synthetic_values,
    load_document_template,
    load_document_validation_report,
    run_document_profiling,
    run_document_rendering,
    run_document_validation,
    run_semantic_binding,
    run_template_compilation,
    run_value_generation,
)
from synth_platform.infrastructure.jobs.inline import LocalJobRunner
from synth_platform.interfaces.api.routes.shared import (
    get_workflow_session_or_404,
    job_response,
    ok,
    raise_api,
    session_response,
)
from synth_platform.interfaces.api.store import get_api_store

router = APIRouter(prefix="/api/document", tags=["document"])
WORKFLOW_TYPE = "pdf_twin"


class DocumentSessionRequest(BaseModel):
    intent: str = "Document testing"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentConfigureRequest(BaseModel):
    seed: int = 42
    extraction_method: str | None = None
    llm_text_enabled: bool = False


@router.post("/sessions", status_code=201)
def create_document_session(body: DocumentSessionRequest) -> dict[str, Any]:
    record = get_api_store().create_session(
        WORKFLOW_TYPE,
        {"intent": body.intent, "metadata": body.metadata, "stage": "created"},
    )
    return ok(session_response(record))


@router.get("/sessions/{session_id}")
def get_document_session(session_id: str) -> dict[str, Any]:
    return ok(session_response(get_workflow_session_or_404(session_id, WORKFLOW_TYPE)))


@router.post("/sessions/{session_id}/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    store = get_api_store()
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    raw = await file.read()
    if not raw:
        raise_api(400, "empty_upload", "Uploaded document is empty.")
    if Path(file.filename or "").suffix.lower() != ".pdf":
        raise_api(400, "unsupported_file_type", "Upload a PDF file.")
    if not raw.lstrip().startswith(b"%PDF"):
        raise_api(400, "document_validation_error", "Uploaded file does not look like a PDF.")

    source_path = store.write_blob(file.filename or "source.pdf", raw)
    state = dict(record.get("state") or {})
    state.update(
        {
            "stage": "document_uploaded",
            "document": {
                "filename": file.filename,
                "path": str(source_path),
                "size": len(raw),
                "content_type": file.content_type,
            },
        }
    )
    record["state"] = state
    return ok(session_response(store.save_session(record)))


@router.post("/sessions/{session_id}/configure")
def configure_document_session(session_id: str, body: DocumentConfigureRequest) -> dict[str, Any]:
    store = get_api_store()
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    state = dict(record.get("state") or {})
    state.update({"stage": "configured", "config": body.model_dump(mode="json", exclude_none=True)})
    record["state"] = state
    return ok(session_response(store.save_session(record)))


@router.post("/sessions/{session_id}/generate", status_code=202)
def generate_document(session_id: str) -> dict[str, Any]:
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    document = (record.get("state") or {}).get("document")
    if not isinstance(document, dict):
        raise_api(409, "document_required", "Upload a PDF before starting Document Twin generation.")
    job = LocalJobRunner(get_api_store()).submit(
        kind="document.generate",
        session_id=session_id,
        workflow_type=WORKFLOW_TYPE,
        payload={"session_id": session_id},
        handler=_run_document_job,
    )
    return ok({"job": job_response(job)})


def _run_document_job(job_id: str) -> None:
    store = get_api_store()
    job = store.advance_job(job_id, status="running", stage="analyzing_input", percent=5, message="Analyzing input")
    session_id = str(job["payload"]["session_id"])
    try:
        session = store.get_session(session_id)
        state = dict(session.get("state") or {})
        document = state.get("document") or {}
        source_path = Path(str(document.get("path") or ""))
        if not source_path.is_file():
            raise ValueError("Uploaded PDF is missing.")

        config = dict(state.get("config") or {})
        seed = int(config.get("seed") or 42)
        doc_id = "doc1"
        workdir = store.root / "document_outputs" / session_id / job_id
        manifest = RunManifest.create(runs_dir=workdir / "runs")

        store.advance_job(job_id, stage="learning_patterns", percent=20, message="Learning data patterns")
        preferred = config.get("extraction_method")
        profile_result = run_document_profiling(
            PDFDocumentAdapter(),
            source_path,
            doc_id,
            manifest,
            workdir / "metadata",
            preferred_extraction_method=str(preferred) if preferred else None,
        )
        _ensure_success(profile_result, "document profiling")
        profile = load_document_profile(manifest.output_path(f"documents/{doc_id}/{DOCUMENT_PROFILE_FILENAME}"))

        template_result = run_template_compilation(
            source_path,
            doc_id,
            manifest,
            extraction_method=profile["extraction_method"],
        )
        _ensure_success(template_result, "template compilation")
        template = load_document_template(manifest.output_path(f"documents/{doc_id}/document_template.json"))

        binding_result = run_semantic_binding(template, doc_id, manifest, template_reference=str(source_path))
        _ensure_success(binding_result, "semantic binding")
        binding_map = load_document_binding_map(manifest.output_path(f"documents/{doc_id}/document_binding_map.json"))

        store.advance_job(job_id, stage="generating_synthetic_data", percent=55, message="Generating synthetic data")
        values_result = run_value_generation(
            template,
            binding_map,
            doc_id,
            manifest,
            binding_map_reference=str(source_path),
            seed=seed,
            llm_text_enabled=bool(config.get("llm_text_enabled")),
        )
        _ensure_success(values_result, "value generation")
        values = load_document_synthetic_values(manifest.output_path(f"documents/{doc_id}/document_synthetic_values.json"))

        render_result = run_document_rendering(
            template,
            binding_map,
            values,
            doc_id,
            manifest,
            synthetic_values_reference=str(source_path),
        )
        _ensure_success(render_result, "document rendering")
        rendered_path = manifest.output_path(f"documents/{doc_id}/rendered.pdf")
        ground_truth = load_document_ground_truth(manifest.output_path(f"documents/{doc_id}/document_ground_truth.json"))

        store.advance_job(job_id, stage="validating_quality", percent=80, message="Validating data quality")
        validation_result = run_document_validation(
            rendered_path,
            ground_truth,
            doc_id,
            manifest,
            ground_truth_reference=str(source_path),
            history=None,
        )
        _ensure_success(validation_result, "document validation")
        validation = load_document_validation_report(
            manifest.output_path(f"documents/{doc_id}/document_validation_report.json")
        )

        store.advance_job(job_id, stage="preparing_files", percent=92, message="Preparing files")
        bundle = store.create_result_bundle(
            session_id=session_id,
            workflow_type=WORKFLOW_TYPE,
            preview={
                "document_type": profile.get("classification", {}).get("document_type"),
                "page_count": template.get("page_count"),
                "field_count": len(values.get("fields") or {}),
            },
            quality_report=validation,
            summary={
                "doc_id": doc_id,
                "rendered_pdf": str(rendered_path),
                "extraction_method": profile.get("extraction_method"),
                "run_id": manifest.run_id,
            },
            artifacts=_document_artifacts(manifest.run_dir, rendered_path),
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
            error={"code": "document_generation_failed", "message": str(exc)},
            result=None,
        )


def _document_artifacts(run_dir: Path, rendered_path: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if rendered_path.is_file():
        artifacts.append(_artifact(rendered_path, role="generated_pdf", media_type="application/pdf"))
    for path in sorted(run_dir.rglob("*.json")):
        artifacts.append(_artifact(path, role="report", media_type="application/json"))
    return artifacts


def _artifact(path: Path, *, role: str, media_type: str) -> dict[str, Any]:
    return {
        "id": path.stem,
        "name": path.name,
        "path": str(path),
        "media_type": media_type,
        "size": path.stat().st_size if path.is_file() else None,
        "role": role,
        "metadata": {},
    }


def _ensure_success(result: Any, stage: str) -> None:
    if hasattr(result, "is_success") and result.is_success():
        return
    errors = getattr(result, "errors", None)
    raise RuntimeError(f"{stage} failed: {errors or result}")
