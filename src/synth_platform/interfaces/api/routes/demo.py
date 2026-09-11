"""Demo-only placeholder runs for source-free chat requests."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from synth_platform.interfaces.api.routes.shared import job_response, ok
from synth_platform.interfaces.api.store import get_api_store

router = APIRouter(prefix="/api/demo", tags=["demo"])


class PlaceholderRunRequest(BaseModel):
    message: str = ""
    workflow_type: str = "database_twin"
    intent: dict[str, Any] = Field(default_factory=dict)


@router.post("/placeholder-run", status_code=201)
def create_placeholder_run(body: PlaceholderRunRequest) -> dict[str, Any]:
    workflow_type = _normalize_workflow(body.workflow_type)
    row_count = _requested_row_count(body.intent)
    customer_rows = max(1, min(row_count, 12))
    claim_rows = max(1, min(row_count, 12))
    store = get_api_store()
    session = store.create_session(
        workflow_type,
        {
            "stage": "completed",
            "intent": body.message or "Demo placeholder request",
            "metadata": {"source": "chat", "demo_placeholder": True, "intent": body.intent},
            "source": {
                "source_type": "demo_placeholder_db",
                "filename": "demo_placeholder.sqlite",
                "tables": ["customers", "claims"],
                "note": "Demo placeholder only. No external database was read.",
            },
            "config": {"row_count": row_count, "seed": 42},
        },
    )
    job = store.create_job(
        kind="demo.placeholder",
        payload={"session_id": session["id"], "workflow_type": workflow_type},
        session_id=session["id"],
        workflow_type=workflow_type,
        stage="preparing_files",
        message="Preparing demo placeholder output",
    )
    result = store.create_result_bundle(
        session_id=session["id"],
        workflow_type=workflow_type,
        preview={
            "tables": {
                "customers": [
                    {"customer_id": f"C-{1000 + index}", "segment": "commercial" if index % 2 else "individual", "status": "active"}
                    for index in range(1, customer_rows + 1)
                ],
                "claims": [
                    {"claim_id": f"CL-{9000 + index}", "customer_id": f"C-{1000 + index}", "amount": round(250.0 + index * 137.5, 2)}
                    for index in range(1, claim_rows + 1)
                ],
            },
            "source": "demo_placeholder_db",
        },
        quality_report={"status": "demo", "passed": True, "checks": ["placeholder_output_created"]},
        summary={
            "demo_placeholder": True,
            "message": body.message,
            "source": "demo_placeholder_db",
            "requested_row_count": row_count,
            "row_counts": {"customers": row_count, "claims": row_count},
            "limitation": "This demo output was generated from placeholder records, not a real uploaded source.",
        },
        artifacts=[],
        metadata={"job_id": job["id"], "demo_placeholder": True},
    )
    store.update_job(
        job["id"],
        status="succeeded",
        stage="completed",
        percent=100.0,
        message="Demo placeholder output ready",
        progress=["queued", "Preparing demo placeholder output", "Demo placeholder output ready"],
        result_id=result["id"],
        result={"session_id": session["id"], "result_id": result["id"]},
        error=None,
    )
    return ok({"job": job_response(store.get_job(job["id"])), "result_id": result["id"], "session_id": session["id"]})


def _requested_row_count(intent: dict[str, Any]) -> int:
    prefill = intent.get("prefill") if isinstance(intent, dict) else None
    value = prefill.get("record_count") if isinstance(prefill, dict) else None
    try:
        return max(1, min(int(value), 1_000_000))
    except (TypeError, ValueError):
        return 10


def _normalize_workflow(workflow_type: str) -> str:
    aliases = {
        "document_twin": "pdf_twin",
        "pdf_twin": "pdf_twin",
        "schema_twin": "schema_twin",
        "database_twin": "database_twin",
        "interaction_twin": "interaction_twin",
    }
    return aliases.get(workflow_type, "database_twin")
