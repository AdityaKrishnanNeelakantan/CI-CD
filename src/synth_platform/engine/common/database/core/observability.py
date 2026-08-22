"""Structured logging for pipeline stage execution.

Wraps stdlib `logging` behind a single named logger. RunManifest.record_stage()
is the one choke-point every stage result passes through (discovery, profiling,
cleaning, inference, contract approval, synthesis, artifact export, relational
generation, QA, target write), so that is the only call site this needs to be
wired into - no per-service instrumentation required.
"""

from __future__ import annotations

import logging

from synth_platform.engine.common.database.core.stage_result import StageResult

LOGGER_NAME = "synthetic_data_generation"

logger = logging.getLogger(LOGGER_NAME)


def log_stage_result(run_id: str, result: StageResult) -> None:
    """Log a single pipeline stage's outcome at a level matching its severity.

    success, no warnings -> INFO; success with warnings -> WARNING;
    failed -> ERROR. Callers (application code, not tests) attach handlers
    via standard `logging` configuration - this module never configures
    handlers itself, so it stays silent unless the host app opts in.
    """
    context = {
        "run_id": run_id,
        "stage": result.stage_name,
        "status": result.status,
        "metrics": result.metrics,
        "duration_seconds": result.evidence.get("duration_seconds"),
    }
    if not result.is_success():
        logger.error("stage failed: %s", context | {"errors": result.errors})
    elif result.warnings:
        logger.warning("stage completed with warnings: %s", context | {"warnings": result.warnings})
    else:
        logger.info("stage completed: %s", context)
