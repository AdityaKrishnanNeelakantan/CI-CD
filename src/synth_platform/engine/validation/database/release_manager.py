"""Mode-gated release decisions for synthetic datasets.

Extracts the release-policy logic that was embedded in QA validation
into a standalone, reusable Release Manager.  Each release mode defines
how QA evidence maps to a release decision (PASS / FAIL / REVIEW /
BLOCKED) so the same policy can be applied outside the QA stage itself
(e.g. artifact export gates, API endpoints, CI pipelines).
"""

from __future__ import annotations

from typing import Any

RELEASE_PASS = "PASS"
RELEASE_FAIL = "FAIL"
RELEASE_REVIEW = "REVIEW"
RELEASE_BLOCKED = "BLOCKED"

MODE_LEARNED_RESTRICTED = "LEARNED_RESTRICTED"
MODE_MOCK_PRIVATE = "MOCK_PRIVATE"
MODE_DP_SHAREABLE = "DP_SHAREABLE"

VALID_RELEASE_MODES = frozenset({MODE_LEARNED_RESTRICTED, MODE_MOCK_PRIVATE, MODE_DP_SHAREABLE})


def evaluate_release(
    qa_report: dict[str, Any],
    release_mode: str = MODE_LEARNED_RESTRICTED,
    dp_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a synthetic dataset may be released under the given mode.

    Parameters
    ----------
    qa_report:
        A qa_report.json dict (must contain ``hard_checks_passed`` and
        optionally a nested ``report`` with fidelity/integrity evidence).
    release_mode:
        One of LEARNED_RESTRICTED, MOCK_PRIVATE, or DP_SHAREABLE.
    dp_report:
        Required for DP_SHAREABLE mode; must contain ``verified`` == True
        and an ``epsilon`` budget entry.
    """
    if release_mode not in VALID_RELEASE_MODES:
        raise ValueError(f"unknown release_mode {release_mode!r}; expected one of {sorted(VALID_RELEASE_MODES)}")

    hard_checks_passed = qa_report.get("hard_checks_passed", False)
    report = qa_report.get("report", {})
    fidelity = report.get("fidelity", {})
    has_fidelity_evidence = bool(fidelity)

    if release_mode == MODE_LEARNED_RESTRICTED:
        if hard_checks_passed:
            decision = RELEASE_PASS
            reason = "all hard integrity checks passed"
        else:
            decision = RELEASE_BLOCKED
            reason = "hard integrity checks failed; release blocked under LEARNED_RESTRICTED"

    elif release_mode == MODE_MOCK_PRIVATE:
        if hard_checks_passed:
            decision = RELEASE_PASS
            reason = "hard checks passed (informational mode; no fidelity gate)"
        else:
            decision = RELEASE_REVIEW
            reason = "hard checks failed; flagged for review under MOCK_PRIVATE (non-blocking)"

    else:  # MODE_DP_SHAREABLE
        dp_verified = dp_report is not None and dp_report.get("verified") is True
        if not dp_verified:
            decision = RELEASE_BLOCKED
            reason = "DP_SHAREABLE requires a verified differential-privacy report"
        elif not hard_checks_passed:
            decision = RELEASE_BLOCKED
            reason = "hard integrity checks failed despite verified DP report"
        elif not has_fidelity_evidence:
            decision = RELEASE_REVIEW
            reason = "no fidelity evidence available; manual review required before DP release"
        else:
            decision = RELEASE_PASS
            reason = "verified DP report and hard checks passed"

    return {
        "release_mode": release_mode,
        "decision": decision,
        "reason": reason,
        "hard_checks_passed": hard_checks_passed,
        "has_fidelity_evidence": has_fidelity_evidence,
        "dp_verified": dp_report.get("verified") if dp_report else None,
    }
