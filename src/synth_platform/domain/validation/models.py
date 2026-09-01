"""Validation domain models with honest, capability-aware status semantics.

Pure (no pandas/IO). Two ideas make the release gate honest for *any* source:

1. NOT_APPLICABLE — a neutral verdict for a metric that does not apply to this
   source (e.g. a temporal-preservation metric on a static relational source).
   It never blocks a release and never fake-passes one. This is what lets
   "every critical metric must pass" be true AND satisfiable at once
   (ARCHITECTURE_AUDIT RC-9 / A-03 / A-13).

2. Capabilities — each check declares which capabilities it needs to be
   meaningful; the compiled IR declares which capabilities the source actually
   exercises. A check whose needs are not exercised is marked NOT_APPLICABLE by
   the gate, rather than being force-run into a permanent WARN or silently
   dropped (fake-green).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIPPED = "SKIPPED"
    NOT_RUN = "NOT_RUN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Capability(str, Enum):
    """A structural property a source may or may not exercise. Metrics that
    need a capability the source does not have are NOT_APPLICABLE."""
    RELATIONAL = "relational"      # >1 table with foreign keys
    CATEGORICAL = "categorical"    # at least one categorical attribute
    NUMERIC = "numeric"            # at least one numeric attribute
    TEMPORAL = "temporal"          # timestamped / dated attributes
    SEQUENCE = "sequence"          # ordered per-entity sequences
    BEHAVIORAL = "behavioral"      # state-machine / event-stream structure
    HOLDOUT = "holdout"            # a real train/holdout split is available


class CheckResult(_Base):
    name: str
    dimension: str  # structural|fidelity|privacy|utility|business|behavioral
    status: Status
    ran: bool = False
    critical: bool | None = None   # gate-driving; defaults to `required`
    required: bool = True          # retained: back-compat + human intent
    requires: list[Capability] = Field(default_factory=list)
    metric: float | None = None
    threshold: float | None = None
    detail: str = ""

    @model_validator(mode="after")
    def _default_critical(self) -> "CheckResult":
        if self.critical is None:
            object.__setattr__(self, "critical", self.required)
        return self

    @property
    def is_critical(self) -> bool:
        return bool(self.critical)


class ValidationReport(_Base):
    checks: list[CheckResult] = Field(default_factory=list)
    overall: Status = Status.NOT_RUN


class ReleaseDecision(_Base):
    verdict: Status
    blocking: list[str] = Field(default_factory=list)
    not_applicable: list[str] = Field(default_factory=list)
    detail: str = ""
