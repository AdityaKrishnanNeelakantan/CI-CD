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
    ui_source_preview_rows: int = 5
    db_default_parent_rows: int = 500
    db_default_child_rows: int = 1_000
    db_min_recommended_total_rows: int = 500
    db_max_recommended_total_rows: int = 5_000
    db_sample_database_min_entities: int = 50
    db_sample_database_max_entities: int = 2_000
    db_sample_database_step_entities: int = 50
    db_profile_sample_limit: int = 3_000
    db_inference_sample_limit: int = 5_000
    db_training_sample_limit: int = 5_000
    db_training_preview_rows: int = 5
    db_generation_seed: int = 7
    db_drift_preview_rows: int = 500
    db_default_chunk_size: int = 10_000
    db_min_chunk_size: int = 100
    db_chunk_step: int = 100
    db_llm_max_rows: int = 50
    db_default_numeric_upper_bound: float = 1_000_000.0
    db_large_generation_warning_rows: int = 10_000
    transcript_default_turns: int = 6
    transcript_generation_seed: int = 17
    schema_default_rows_per_table: int = 100
    pdf_generation_backend: str = "nemo"
    transcript_generation_backend: str = "nemo"
    nvidia_nemo_enabled: bool = True
    nvidia_guardrails_config_path: str = ""
    nvidia_curator_base_url: str = ""
    nvidia_curator_api_key: str = ""
    nvidia_curator_model: str = "meta/llama-3.1-70b-instruct"

    @classmethod
    def from_env(cls) -> "Settings":
        import os

        def _f(k, d):
            v = os.environ.get(k)
            return type(d)(v) if v is not None else d

        def _bool(k: str, d: bool) -> bool:
            v = os.environ.get(k)
            if v is None:
                return d
            return v.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            seed=_f("SP_SEED", 42),
            max_rows_per_table=_f("SP_MAX_ROWS", 50_000),
            holdout_fraction=_f("SP_HOLDOUT", 0.30),
            ui_source_preview_rows=_f("SP_UI_SOURCE_PREVIEW_ROWS", 5),
            db_default_parent_rows=_f("SP_DB_DEFAULT_PARENT_ROWS", 500),
            db_default_child_rows=_f("SP_DB_DEFAULT_CHILD_ROWS", 1_000),
            db_min_recommended_total_rows=_f("SP_DB_MIN_RECOMMENDED_TOTAL_ROWS", 500),
            db_max_recommended_total_rows=_f("SP_DB_MAX_RECOMMENDED_TOTAL_ROWS", 5_000),
            db_sample_database_min_entities=_f("SP_DB_SAMPLE_MIN_ENTITIES", 50),
            db_sample_database_max_entities=_f("SP_DB_SAMPLE_MAX_ENTITIES", 2_000),
            db_sample_database_step_entities=_f("SP_DB_SAMPLE_STEP_ENTITIES", 50),
            db_profile_sample_limit=_f("SP_DB_PROFILE_SAMPLE_LIMIT", 3_000),
            db_inference_sample_limit=_f("SP_DB_INFERENCE_SAMPLE_LIMIT", 5_000),
            db_training_sample_limit=_f("SP_DB_TRAINING_SAMPLE_LIMIT", 5_000),
            db_training_preview_rows=_f("SP_DB_TRAINING_PREVIEW_ROWS", 5),
            db_generation_seed=_f("SP_DB_GENERATION_SEED", 7),
            db_drift_preview_rows=_f("SP_DB_DRIFT_PREVIEW_ROWS", 500),
            db_default_chunk_size=_f("SP_DB_DEFAULT_CHUNK_SIZE", 10_000),
            db_min_chunk_size=_f("SP_DB_MIN_CHUNK_SIZE", 100),
            db_chunk_step=_f("SP_DB_CHUNK_STEP", 100),
            db_llm_max_rows=_f("SP_DB_LLM_MAX_ROWS", 50),
            db_default_numeric_upper_bound=_f("SP_DB_DEFAULT_NUMERIC_UPPER_BOUND", 1_000_000.0),
            db_large_generation_warning_rows=_f("SP_DB_LARGE_GENERATION_WARNING_ROWS", 10_000),
            transcript_default_turns=_f("SP_TRANSCRIPT_DEFAULT_TURNS", 6),
            transcript_generation_seed=_f("SP_TRANSCRIPT_GENERATION_SEED", 17),
            schema_default_rows_per_table=_f("SP_SCHEMA_DEFAULT_ROWS_PER_TABLE", 100),
            pdf_generation_backend=_f("SP_PDF_GENERATION_BACKEND", "nemo"),
            transcript_generation_backend=_f("SP_TRANSCRIPT_GENERATION_BACKEND", "nemo"),
            nvidia_nemo_enabled=_bool("SP_NVIDIA_NEMO_ENABLED", True),
            nvidia_guardrails_config_path=_f("SP_NVIDIA_GUARDRAILS_CONFIG_PATH", ""),
            nvidia_curator_base_url=_f("SP_NVIDIA_CURATOR_BASE_URL", ""),
            nvidia_curator_api_key=_f("SP_NVIDIA_CURATOR_API_KEY", ""),
            nvidia_curator_model=_f("SP_NVIDIA_CURATOR_MODEL", "meta/llama-3.1-70b-instruct"),
        )
