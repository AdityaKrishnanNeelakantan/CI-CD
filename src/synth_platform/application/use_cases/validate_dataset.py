"""Validate generated tables against the compiled artifact.

Structural + fidelity always run. Privacy/utility run only when a holdout
evaluator is supplied. The set of capabilities the source *exercises* is derived
here and handed to the release gate, which flips inapplicable checks to
NOT_APPLICABLE rather than forcing a verdict (ARCHITECTURE_AUDIT RC-9).
"""
from __future__ import annotations

import pandas as pd

from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.domain.validation.models import (
    Capability, CheckResult, ValidationReport,
)
from synth_platform.domain.validation.release_gate import decide
from synth_platform.engine.validation.service import structural_and_fidelity_checks


def exercised_capabilities(
    artifact: SynthArtifact, has_holdout: bool
) -> set[Capability]:
    """Which structural properties this source actually exercises. Drives
    applicability; derived from the compiled IR, never hardcoded per source."""
    caps: set[Capability] = set()
    if artifact.schema_.foreign_keys:
        caps.add(Capability.RELATIONAL)
    for tprof in artifact.data_profile.tables.values():
        for cprof in tprof.columns.values():
            if cprof.physical_type == "categorical":
                caps.add(Capability.CATEGORICAL)
            elif cprof.physical_type in ("numeric", "datetime"):
                caps.add(Capability.NUMERIC)
            if cprof.physical_type == "datetime":
                caps.add(Capability.TEMPORAL)
    if has_holdout:
        caps.add(Capability.HOLDOUT)
    return caps


def validate_dataset(
    artifact: SynthArtifact,
    tables: dict[str, pd.DataFrame],
    privacy_utility_checks: list[CheckResult] | None = None,
) -> ValidationReport:
    checks = structural_and_fidelity_checks(artifact, tables)
    has_holdout = privacy_utility_checks is not None
    if privacy_utility_checks is None:
        from synth_platform.engine.validation.service import privacy_utility_not_run
        checks += privacy_utility_not_run()
    else:
        checks += privacy_utility_checks
    report = ValidationReport(checks=checks)
    decide(report, exercised=exercised_capabilities(artifact, has_holdout))
    return report
