"""Customer Interaction Twin API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from synth_platform.application.workflows.interaction_twin import parse_transcript, run_interaction_twin
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

router = APIRouter(prefix="/api/interaction", tags=["interaction"])
WORKFLOW_TYPE = "interaction_twin"


class InteractionSessionRequest(BaseModel):
    intent: str = "Customer support testing"
    metadata: dict[str, Any] = Field(default_factory=dict)


class InteractionConfigureRequest(BaseModel):
    interaction_type: str = "Customer Support"
    output_format: str = "Structured JSON + Synthetic Logs"
    remove_sensitive_information: bool = True
    seed: int = 42


@router.post("/sessions", status_code=201)
def create_interaction_session(body: InteractionSessionRequest) -> dict[str, Any]:
    record = get_api_store().create_session(
        WORKFLOW_TYPE,
        {"intent": body.intent, "metadata": body.metadata, "stage": "created"},
    )
    return ok(session_response(record))


@router.get("/sessions/{session_id}")
def get_interaction_session(session_id: str) -> dict[str, Any]:
    return ok(session_response(get_workflow_session_or_404(session_id, WORKFLOW_TYPE)))


@router.post("/sessions/{session_id}/upload")
async def upload_interaction_transcript(session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    store = get_api_store()
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    raw = await file.read()
    if not raw:
        raise_api(400, "empty_upload", "Uploaded transcript is empty.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".log"}:
        raise_api(400, "unsupported_file_type", "Upload a .txt or .log transcript.")
    try:
        text = raw.decode("utf-8-sig")
        parsed = parse_transcript(text)
    except Exception as exc:
        raise_api(400, "transcript_parse_error", str(exc))

    transcript_path = store.write_blob(file.filename or "transcript.txt", raw)
    state = dict(record.get("state") or {})
    state.update(
        {
            "stage": "transcript_uploaded",
            "transcript": {
                "filename": file.filename,
                "path": str(transcript_path),
                "size": len(raw),
                "turn_count": len(parsed.turns),
                "speaker_counts": parsed.speaker_counts,
            },
        }
    )
    record["state"] = state
    return ok(session_response(store.save_session(record)))


@router.post("/sessions/{session_id}/configure")
def configure_interaction_session(session_id: str, body: InteractionConfigureRequest) -> dict[str, Any]:
    store = get_api_store()
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    state = dict(record.get("state") or {})
    state.update({"stage": "configured", "config": body.model_dump(mode="json")})
    record["state"] = state
    return ok(session_response(store.save_session(record)))


@router.post("/sessions/{session_id}/generate", status_code=202)
def generate_interaction(session_id: str) -> dict[str, Any]:
    record = get_workflow_session_or_404(session_id, WORKFLOW_TYPE)
    transcript = (record.get("state") or {}).get("transcript")
    if not isinstance(transcript, dict):
        raise_api(409, "transcript_required", "Upload a transcript before starting Interaction Twin generation.")
    job = LocalJobRunner(get_api_store()).submit(
        kind="interaction.generate",
        session_id=session_id,
        workflow_type=WORKFLOW_TYPE,
        payload={"session_id": session_id},
        handler=_run_interaction_job,
    )
    return ok({"job": job_response(job)})


def _run_interaction_job(job_id: str) -> None:
    store = get_api_store()
    job = store.advance_job(job_id, status="running", stage="analyzing_input", percent=10, message="Analyzing input")
    session_id = str(job["payload"]["session_id"])
    try:
        session = store.get_session(session_id)
        state = dict(session.get("state") or {})
        transcript = state.get("transcript") or {}
        transcript_path = Path(str(transcript.get("path") or ""))
        if not transcript_path.is_file():
            raise ValueError("Uploaded transcript is missing.")
        config = dict(state.get("config") or {})

        store.advance_job(job_id, stage="applying_privacy_checks", percent=35, message="Applying privacy checks")
        text = transcript_path.read_text(encoding="utf-8-sig")
        output_dir = store.root / "interaction_outputs" / session_id / job_id
        store.advance_job(
            job_id,
            stage="generating_synthetic_data",
            percent=60,
            message="Generating synthetic data",
        )
        result = run_interaction_twin(
            text,
            interaction_type=str(config.get("interaction_type") or "Customer Support"),
            output_format=str(config.get("output_format") or "Structured JSON + Synthetic Logs"),
            remove_sensitive_information=bool(config.get("remove_sensitive_information", True)),
            seed=int(config.get("seed") or 42),
            output_dir=output_dir,
            project_name=Path(str(transcript.get("filename") or "Interaction Twin")).stem,
            history=None,
            product_settings=PlatformDB(),
        )
        store.advance_job(job_id, stage="validating_quality", percent=82, message="Validating data quality")
        if not result.hard_checks_passed:
            raise RuntimeError(f"interaction validation failed: {result.validation_report}")

        store.advance_job(job_id, stage="preparing_files", percent=92, message="Preparing files")
        package_path = store.write_blob("interaction_twin.zip", result.package_bytes or b"")
        artifacts = [_artifact(package_path, role="download", media_type="application/zip")]
        artifacts.extend(_artifact(path, role=role, media_type=_media_type(path)) for role, path in result.output_paths.items())
        bundle = store.create_result_bundle(
            session_id=session_id,
            workflow_type=WORKFLOW_TYPE,
            preview={"turns": [turn.to_dict() for turn in result.synthetic.turns[:25]]},
            quality_report=result.validation_report,
            summary={
                "synthetic_turn_count": len(result.synthetic.turns),
                "redaction_findings": result.redaction_report.get("finding_count", 0),
                "output_formats": sorted(result.output_paths.keys()),
            },
            artifacts=artifacts,
            metadata={"job_id": job_id},
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
            error={"code": "interaction_generation_failed", "message": str(exc)},
            result=None,
        )


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


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".log":
        return "text/plain"
    return "application/octet-stream"
