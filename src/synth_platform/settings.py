"""Process settings. Selects components; contains no business logic."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    max_rows_per_table: int = 50_000
    holdout_fraction: float = 0.30
    allowed_backends: tuple[str, ...] = ("statistical",)
    minimum_fidelity_score: float = 0.80
    approved_output_root: str = "output"
    staging_root: str = ".staging"

    @classmethod
    def from_env(cls) -> "Settings":
        import os

        def _f(k, d):
            v = os.environ.get(k)
            return type(d)(v) if v is not None else d

        return cls(
            seed=_f("SP_SEED", 42),
            max_rows_per_table=_f("SP_MAX_ROWS", 50_000),
            holdout_fraction=_f("SP_HOLDOUT", 0.30),
            approved_output_root=os.environ.get("SP_APPROVED_OUTPUT_ROOT", "output"),
            staging_root=os.environ.get("SP_STAGING_ROOT", ".staging"),
        )
