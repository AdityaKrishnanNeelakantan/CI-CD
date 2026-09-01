"""Generation request value objects (pure, pandas-free).

Lives in the innermost layer so both the engine (which executes generation) and
the application (which orchestrates it) depend inward, never on each other
(ARCHITECTURE_AUDIT A-06 / RC-10).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_table_rows: dict[str, int] = Field(default_factory=dict)
    scale: float = 1.0
    seed: int = 42
    output_format: str = "csv"  # csv|sqlite|parquet|postgresql
