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
    guardrail_policy_version: str = "1.0"
    guardrail_max_input_characters: int = 20_000
    guardrail_max_output_characters: int = 20_000
    local_model_host: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> Settings:
        import os

        def _value(key: str, default):
            value = os.environ.get(key)
            return type(default)(value) if value is not None else default

        allowed_backends = tuple(
            item.strip()
            for item in os.environ.get("SP_ALLOWED_BACKENDS", "statistical").split(",")
            if item.strip()
        )
        return cls(
            seed=_value("SP_SEED", 42),
            max_rows_per_table=_value("SP_MAX_ROWS", 50_000),
            holdout_fraction=_value("SP_HOLDOUT", 0.30),
            allowed_backends=allowed_backends or ("statistical",),
            minimum_fidelity_score=_value("SP_MIN_FIDELITY", 0.80),
            approved_output_root=os.environ.get("SP_OUTPUT_ROOT", "output"),
            staging_root=os.environ.get("SP_STAGING_ROOT", ".staging"),
            guardrail_policy_version=os.environ.get(
                "SP_GUARDRAIL_POLICY_VERSION", "1.0"
            ),
            guardrail_max_input_characters=_value(
                "SP_GUARDRAIL_MAX_INPUT_CHARS", 20_000
            ),
            guardrail_max_output_characters=_value(
                "SP_GUARDRAIL_MAX_OUTPUT_CHARS", 20_000
            ),
            local_model_host=os.environ.get(
                "SP_LOCAL_MODEL_HOST", "http://localhost:11434"
            ),
        )
