"""Validation service: delegates to the metric registry (A-05).

Structural + fidelity + correlation come from `run_metrics`. Privacy/utility are
capability-gated and contributed by the holdout evaluator when a split exists;
otherwise `privacy_utility_not_run` emits NOT_RUN checks that the gate flips to
NOT_APPLICABLE (RC-9). No per-check thresholding lives here anymore.
"""
from __future__ import annotations

from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.domain.validation.metric import ValidationContext
from synth_platform.domain.validation.models import Capability, CheckResult, Status
from synth_platform.engine.validation.metric_registry import run_metrics


def structural_and_fidelity_checks(artifact: SynthArtifact, tables) -> list[CheckResult]:
    """Run the default metric registry (structural, marginal fidelity,
    correlation preservation) against the generated tables."""
    return run_metrics(ValidationContext(artifact=artifact, tables=tables))


def privacy_utility_not_run() -> list[CheckResult]:
    """Privacy/utility require a real holdout split. When none is exercised they
    are NOT_APPLICABLE (neutral), resolved by the gate from the HOLDOUT
    capability (RC-9)."""
    return [
        CheckResult(name="membership_inference", dimension="privacy", critical=True,
                    requires=[Capability.HOLDOUT], ran=False, status=Status.NOT_RUN,
                    detail="requires a holdout split"),
        CheckResult(name="tstr_utility", dimension="utility", critical=True,
                    requires=[Capability.HOLDOUT], ran=False, status=Status.NOT_RUN,
                    detail="requires a holdout split"),
        CheckResult(name="formal_differential_privacy", dimension="privacy",
                    critical=False, required=False, ran=False, status=Status.NOT_APPLICABLE,
                    detail="not claimed"),
    ]
