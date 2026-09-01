"""The single, capability-aware release-verdict computation (pure).

Canonical gate (ARCHITECTURE_AUDIT RC-9 / A-03 / A-13). Nothing else may
re-implement this. Two pure steps:

  select_applicability(checks, exercised) -> checks
      flips a check to NOT_APPLICABLE when a capability it `requires` is not in
      the set the source actually exercises. Neutral: never blocks, never passes.

  compute_overall(checks) -> Status
      FAIL  if any check FAILs, or any *applicable critical* check did not run
            (NOT_RUN/SKIPPED) — an unmeasured critical dimension is a failure,
            never a silent pass.
      PASS  only if every applicable critical check ran and PASSed.
      WARN  otherwise (non-critical gaps / warnings only).

NOT_APPLICABLE and non-critical checks never turn a release red. This makes
"every critical metric must pass" honest for any source: a static relational
source's temporal metric is NOT_APPLICABLE, not a permanent WARN.
"""
from __future__ import annotations

from collections.abc import Iterable

from synth_platform.domain.validation.models import (
    Capability,
    CheckResult,
    ReleaseDecision,
    Status,
    ValidationReport,
)

_UNMEASURED = (Status.NOT_RUN, Status.SKIPPED)


def select_applicability(
    checks: list[CheckResult], exercised: Iterable[Capability]
) -> list[CheckResult]:
    """Return checks with inapplicable ones flipped to NOT_APPLICABLE."""
    have = set(exercised)
    out: list[CheckResult] = []
    for c in checks:
        if c.requires and not set(c.requires).issubset(have):
            missing = sorted(cap.value for cap in set(c.requires) - have)
            out.append(c.model_copy(update={
                "status": Status.NOT_APPLICABLE,
                "detail": (c.detail + " " if c.detail else "")
                          + f"[not applicable: source lacks {', '.join(missing)}]",
            }))
        else:
            out.append(c)
    return out


def _applicable(checks: list[CheckResult]) -> list[CheckResult]:
    return [c for c in checks if c.status != Status.NOT_APPLICABLE]


def compute_overall(checks: list[CheckResult]) -> Status:
    applicable = _applicable(checks)
    if any(c.status == Status.FAIL for c in applicable):
        return Status.FAIL
    critical = [c for c in applicable if c.is_critical]
    if any(c.status in _UNMEASURED for c in critical):
        return Status.FAIL  # an unmeasured critical dimension blocks the release
    if any(c.status in _UNMEASURED for c in applicable):
        return Status.WARN  # only non-critical gaps remain
    if any(c.status == Status.WARN for c in applicable):
        return Status.WARN
    if critical and all(c.status == Status.PASS for c in critical):
        return Status.PASS
    return Status.WARN


def decide(
    report: ValidationReport, exercised: Iterable[Capability] | None = None
) -> ReleaseDecision:
    """Compute the verdict in place. If `exercised` is given, applicability is
    resolved first (the normal path); otherwise checks are taken as-is."""
    checks = (select_applicability(report.checks, exercised)
              if exercised is not None else report.checks)
    report.checks = checks
    verdict = compute_overall(checks)
    report.overall = verdict
    blocking = [
        c.name for c in _applicable(checks)
        if c.status == Status.FAIL or (c.is_critical and c.status in _UNMEASURED)
    ]
    not_applicable = [c.name for c in checks if c.status == Status.NOT_APPLICABLE]
    return ReleaseDecision(
        verdict=verdict, blocking=blocking, not_applicable=not_applicable,
        detail=f"{len(_applicable(checks))} applicable of {len(checks)} checks")
