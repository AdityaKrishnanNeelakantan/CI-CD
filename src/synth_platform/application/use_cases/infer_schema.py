"""Schema/inference use-case entry points.

This module gives callers a stable application-level import without exposing
workflow-specific engine locations.
"""

from synth_platform.engine.inference.database.service import run_inference
from synth_platform.application.workflows.schema_twin import load_schema, summarize_schema

__all__ = ["run_inference", "load_schema", "summarize_schema"]
