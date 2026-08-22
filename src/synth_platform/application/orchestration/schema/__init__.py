"""Unified schema-first and source-driven generation pipelines."""

from synth_platform.application.orchestration.schema.config import PipelineConfig, PipelineProgress
from synth_platform.application.orchestration.schema.export import make_zip_from_paths
from synth_platform.application.orchestration.schema.result import PipelineResult
from synth_platform.application.orchestration.schema.schema_driven import run_schema_pipeline
from synth_platform.application.orchestration.schema.session import (
    clear_large_session_objects,
    get_artifact_for_export,
    render_preview_sample_only,
    store_artifact_reference,
    store_preview_sample_only,
)
from synth_platform.application.orchestration.schema.source_driven import fit_or_reuse_source_generator, run_source_pipeline

__all__ = [
    "PipelineConfig",
    "PipelineProgress",
    "PipelineResult",
    "run_schema_pipeline",
    "run_source_pipeline",
    "fit_or_reuse_source_generator",
    "clear_large_session_objects",
    "store_artifact_reference",
    "get_artifact_for_export",
    "store_preview_sample_only",
    "render_preview_sample_only",
    "make_zip_from_paths",
]
