from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.pii import detect_pii
from synth_platform.engine.documents.pdf.pii_policy import (
    ACTION_FLAG_FOR_REVIEW,
    ACTION_REDACT,
    apply_pii_policy,
    summarize_policy_actions,
)

pytestmark = pytest.mark.unit


def test_high_confidence_email_is_auto_redacted():
    findings = detect_pii("Contact grace@example.com for details.")
    policed = apply_pii_policy(findings)
    assert policed[0]["type"] == "email"
    assert policed[0]["policy_action"] == ACTION_REDACT


def test_low_confidence_finding_below_threshold_is_flagged_for_review():
    findings = [{"type": "credit_card", "start": 0, "end": 4, "confidence": 0.1, "redaction_preview": "**"}]
    policed = apply_pii_policy(findings)
    assert policed[0]["policy_action"] == ACTION_FLAG_FOR_REVIEW


def test_unknown_pii_type_falls_back_to_the_strict_default_threshold():
    findings = [{"type": "some_new_type", "start": 0, "end": 4, "confidence": 0.85, "redaction_preview": "**"}]
    policed = apply_pii_policy(findings)
    assert policed[0]["policy_action"] == ACTION_FLAG_FOR_REVIEW


def test_apply_pii_policy_never_mutates_the_original_findings():
    original = [{"type": "email", "start": 0, "end": 4, "confidence": 0.95, "redaction_preview": "**"}]
    apply_pii_policy(original)
    assert "policy_action" not in original[0]


def test_summarize_policy_actions_counts_each_action():
    findings = [
        {"type": "email", "start": 0, "end": 1, "confidence": 0.95, "redaction_preview": "*"},
        {"type": "email", "start": 2, "end": 3, "confidence": 0.95, "redaction_preview": "*"},
        {"type": "credit_card", "start": 4, "end": 5, "confidence": 0.1, "redaction_preview": "*"},
    ]
    summary = summarize_policy_actions(apply_pii_policy(findings))
    assert summary == {ACTION_REDACT: 2, ACTION_FLAG_FOR_REVIEW: 1}


def test_custom_thresholds_override_the_default_policy():
    findings = [{"type": "ssn", "start": 0, "end": 4, "confidence": 0.65, "redaction_preview": "**"}]
    policed = apply_pii_policy(findings, min_confidence_by_type={"ssn": 0.9})
    assert policed[0]["policy_action"] == ACTION_FLAG_FOR_REVIEW
