"""Process settings. Selects components; contains no business logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    max_rows_per_table: int = 50_000
    holdout_fraction: float = 0.30
    allowed_backends: tuple[str, ...] = ("statistical",)
    minimum_fidelity_score: float = 0.80
    approved_output_root: str = "output"
    staging_root: str = ".staging"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"

    @classmethod
    def from_env(cls) -> "Settings":
        import os

        defaults = cls()

        def _f(k, d):
            v = os.environ.get(k)
            return type(d)(v) if v is not None else d

        return cls(
            seed=_f("SP_SEED", defaults.seed),
            max_rows_per_table=_f("SP_MAX_ROWS", defaults.max_rows_per_table),
            holdout_fraction=_f("SP_HOLDOUT", defaults.holdout_fraction),
            approved_output_root=os.environ.get("SP_APPROVED_OUTPUT_ROOT", defaults.approved_output_root),
            staging_root=os.environ.get("SP_STAGING_ROOT", defaults.staging_root),
            ollama_host=os.environ.get("OLLAMA_HOST", defaults.ollama_host),
            ollama_model=(
                os.environ.get("OLLAMA_MODEL")
                or os.environ.get("OLLAMA_LLM_TEXT_MODEL")
                or os.environ.get("OLLAMA_SMART_VALUE_MODEL")
                or os.environ.get("MVP_LLM_TEXT_MODEL")
                or defaults.ollama_model
            ),
        )
