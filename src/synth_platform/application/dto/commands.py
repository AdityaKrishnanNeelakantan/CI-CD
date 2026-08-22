"""Command models for use cases (keeps parameter lists small & typed)."""
from __future__ import annotations

from synth_platform.domain.generation.models import GenerationRequest  # re-export

from pydantic import BaseModel, ConfigDict, Field


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str                      # e.g. sqlite:///banking.db  OR a file path
    sample_size: int = 50_000
    seed: int = 42
    rare_category_threshold: int = 10


class ValidationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum_fidelity_score: float = 0.80
    fail_on_fk_violation: bool = True
