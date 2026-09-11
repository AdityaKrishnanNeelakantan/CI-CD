"""Application port for workflow job execution."""
from __future__ import annotations

from typing import Any, Callable, Protocol

JobHandler = Callable[[str], None]


class JobQueue(Protocol):
    def submit(self, *, kind: str, payload: dict[str, Any], handler: JobHandler) -> dict[str, Any]:
        """Create a job record and arrange for `handler(job_id)` to run."""
        ...
