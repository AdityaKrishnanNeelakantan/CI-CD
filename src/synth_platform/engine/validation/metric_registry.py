"""Metric registry — the single driver of validation (A-05).

Holds the ordered set of Metrics and runs them against a ValidationContext.
Adding/removing a metric happens here; the service just delegates. Privacy and
utility metrics are contributed by the holdout evaluator when a split exists, so
they are not in the default registry (they require real holdout data).
"""
from __future__ import annotations

from synth_platform.domain.validation.metric import Metric, ValidationContext
from synth_platform.domain.validation.models import CheckResult
from synth_platform.engine.validation.metrics import (
    CorrelationPreservationMetric, MarginalFidelityMetric, StructuralIntegrityMetric,
    ConstraintComplianceMetric,
)


def default_metrics() -> list[Metric]:
    return [
        StructuralIntegrityMetric(),
        MarginalFidelityMetric(),
        CorrelationPreservationMetric(),
        ConstraintComplianceMetric(),
    ]


def run_metrics(ctx: ValidationContext, metrics: list[Metric] | None = None) -> list[CheckResult]:
    checks: list[CheckResult] = []
    for metric in (metrics if metrics is not None else default_metrics()):
        checks.extend(metric.evaluate(ctx))
    return checks
