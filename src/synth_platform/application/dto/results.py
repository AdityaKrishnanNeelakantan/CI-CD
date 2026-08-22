"""Result models returned by use cases."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from synth_platform.domain.validation.models import ReleaseDecision, ValidationReport


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    report: ValidationReport
    decision: ReleaseDecision
    output_dir: str | None = None
    staging_dir: str = ""
