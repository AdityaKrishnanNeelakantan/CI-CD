"""Release-gate semantics (W0 / ARCHITECTURE_AUDIT RC-9, A-03, A-13).

The gate is capability-aware and honest:
  - an unmeasured *critical* applicable check FAILs the release (not WARN),
  - a NOT_APPLICABLE check is neutral (never blocks, never fake-passes),
  - PASS requires every applicable critical check to have run and passed.
"""
from synth_platform.domain.validation.models import (
    Capability, CheckResult, Status, ValidationReport,
)
from synth_platform.domain.validation.release_gate import (
    compute_overall, decide, select_applicability,
)


def _c(status, *, critical=True, ran=True, requires=None, name="c"):
    return CheckResult(name=name, dimension="structural", critical=critical,
                       required=critical, ran=ran, status=status,
                       requires=requires or [])


def test_any_fail_dominates():
    assert compute_overall([_c(Status.PASS), _c(Status.FAIL)]) == Status.FAIL


def test_all_critical_pass_is_pass():
    assert compute_overall([_c(Status.PASS), _c(Status.PASS)]) == Status.PASS


def test_unmeasured_critical_fails_not_warns():
    # stricter than the legacy gate: an unmeasured critical dimension blocks.
    assert compute_overall([_c(Status.PASS), _c(Status.NOT_RUN, ran=False)]) == Status.FAIL


def test_unmeasured_noncritical_only_warns():
    got = compute_overall([_c(Status.PASS),
                           _c(Status.NOT_RUN, critical=False, ran=False)])
    assert got == Status.WARN


def test_not_applicable_is_neutral():
    # a NOT_APPLICABLE critical check neither blocks nor is required to pass
    checks = [_c(Status.PASS), _c(Status.NOT_APPLICABLE)]
    assert compute_overall(checks) == Status.PASS


def test_select_applicability_flips_missing_capability():
    checks = [_c(Status.NOT_RUN, ran=False, requires=[Capability.TEMPORAL], name="temporal")]
    resolved = select_applicability(checks, exercised={Capability.RELATIONAL})
    assert resolved[0].status == Status.NOT_APPLICABLE
    # and therefore does not block a release
    assert compute_overall(resolved) != Status.FAIL


def test_select_applicability_keeps_exercised_capability():
    checks = [_c(Status.NOT_RUN, ran=False, requires=[Capability.HOLDOUT], name="priv")]
    resolved = select_applicability(checks, exercised={Capability.HOLDOUT})
    assert resolved[0].status == Status.NOT_RUN         # applicable, still unmeasured
    assert compute_overall(resolved) == Status.FAIL     # critical + unmeasured -> FAIL


def test_decide_reports_blocking_and_not_applicable():
    r = ValidationReport(checks=[
        _c(Status.PASS, name="ok"),
        _c(Status.NOT_RUN, ran=False, requires=[Capability.TEMPORAL], name="temporal"),
        _c(Status.FAIL, name="bad"),
    ])
    d = decide(r, exercised={Capability.RELATIONAL})
    assert d.verdict == Status.FAIL
    assert "bad" in d.blocking
    assert "temporal" in d.not_applicable
    assert "temporal" not in d.blocking
