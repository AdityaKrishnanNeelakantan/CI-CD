"""Small in-memory run metadata repository for the bounded local demo."""
from __future__ import annotations

from synth_platform.domain.runs.models import SourceRun


class InMemoryMetadataRepository:
    def __init__(self):
        self._runs: dict[str, SourceRun] = {}

    def save(self, run: SourceRun) -> None:
        self._runs[run.run_id] = run.model_copy(deep=True)

    def get(self, run_id: str) -> SourceRun | None:
        run = self._runs.get(run_id)
        return run.model_copy(deep=True) if run else None

    def list(self) -> list[SourceRun]:
        return [run.model_copy(deep=True) for run in self._runs.values()]
