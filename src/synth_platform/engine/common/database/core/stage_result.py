"""Common result contract every pipeline stage returns.

Every stage (discovery, profiling, inference, training, ...) returns a
StageResult so downstream code and the run manifest can treat all stages
uniformly instead of each checkpoint inventing its own response shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


@dataclass
class StageResult:
    stage_name: str
    status: str
    input_references: list[str]
    output_references: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        return self.status == STATUS_SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
