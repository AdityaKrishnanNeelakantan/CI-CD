"""ReleaseGate port: the single publication door.

Generated tables are staged, then promoted to an approved destination ONLY on a
PASS verdict. The application depends on this Protocol; a concrete adapter (e.g.
the filesystem Quarantine) implements it. This keeps the pipeline use-case free
of any adapter import (ARCHITECTURE_AUDIT A-06 / RC-10, and RC-7: one door).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from synth_platform.domain.validation.models import Status, ValidationReport


@runtime_checkable
class StagedResult(Protocol):
    status: Status
    report: ValidationReport

    @property
    def promotable(self) -> bool: ...


@runtime_checkable
class ReleaseGate(Protocol):
    def submit(self, tables: dict[str, pd.DataFrame], report: ValidationReport,
               run_id: str = "run") -> StagedResult: ...

    def promote(self, result: StagedResult, destination) -> StagedResult: ...
