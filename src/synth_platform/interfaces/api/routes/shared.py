"""Shared helpers for API route modules."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from synth_platform.interfaces.api.contracts import GenerationJob, ResultBundle, WorkflowSession

JOB_STATUSES = {"queued", "running", "succeeded", "failed"}


def ok(data: Any) -> dict[str, Any]:
    return {"data": data}


def raise_api(status_code: int, code: str, message: str) -> None:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


def session_response(record: dict[str, Any]) -> dict[str, Any]:
    return WorkflowSession(
        id=record["id"],
        workflow=record["workflow"],
        workflow_type=record["workflow"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        state=record.get("state") or {},
    ).model_dump(mode="json")


def job_response(record: dict[str, Any]) -> dict[str, Any]:
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


def result_response(record: dict[str, Any]) -> dict[str, Any]:
    return ResultBundle.model_validate(record).model_dump(mode="json")


def get_session_or_404(session_id: str) -> dict[str, Any]:
    from synth_platform.interfaces.api.store import get_api_store

    try:
        return get_api_store().get_session(session_id)
    except KeyError:
        raise_api(404, "not_found", f"Session {session_id!r} was not found.")


def get_workflow_session_or_404(session_id: str, workflow_type: str) -> dict[str, Any]:
    record = get_session_or_404(session_id)
    if record["workflow"] != workflow_type:
        raise_api(404, "not_found", f"Session {session_id!r} was not found for workflow {workflow_type!r}.")
    return record

