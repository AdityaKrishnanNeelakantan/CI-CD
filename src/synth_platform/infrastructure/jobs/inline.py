"""Shared local job runner for the FastAPI migration surface."""

from __future__ import annotations

import threading
from typing import Any

from synth_platform.application.ports.job_queue import JobHandler
from synth_platform.interfaces.api.store import ApiStateStore, get_api_store


class LocalJobRunner:
    """Create API job records and execute handlers locally.

    The default path uses a daemon thread to preserve the existing schema API
    behavior. Tests and future CLI callers can set `threaded=False` for a
    deterministic inline run.
    """

    def __init__(self, store: ApiStateStore | None = None, *, threaded: bool = True) -> None:
        self.store = store or get_api_store()
        self.threaded = threaded

    def submit(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        handler: JobHandler,
        session_id: str | None = None,
        workflow_type: str | None = None,
        stage: str = "queued",
        message: str = "queued",
    ) -> dict[str, Any]:
        job = self.store.create_job(
            kind=kind,
            payload=payload,
            session_id=session_id,
            workflow_type=workflow_type,
            stage=stage,
            percent=0.0,
            message=message,
        )
        if self.threaded:
            threading.Thread(target=handler, args=(str(job["id"]),), daemon=True).start()
        else:
            handler(str(job["id"]))
            job = self.store.get_job(str(job["id"]))
        return job
