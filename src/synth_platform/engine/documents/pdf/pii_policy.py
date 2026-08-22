"""PII policy engine (Step 6): decides an ACTION for each already-detected
finding (src/documents/pii.py's detect_pii), decoupled from detection
itself - the same separation Microsoft Presidio draws between its
AnalyzerEngine (RecognizerResult: entity_type/start/end/score) and its
AnonymizerEngine (Operator/OperatorConfig, configured per entity_type,
default operator "replace" if unconfigured)[1]. This project's default
action is REDACT (this pipeline's existing, already-tested masking - see
pii.py's redaction_preview and template_compiler.py's masked_preview),
applied only once a finding's confidence clears a per-type threshold;
anything below threshold is FLAG_FOR_REVIEW rather than silently redacted
or silently allowed through unmasked.

[1] https://presidio.dataprivacystack.org/anonymizer/ (Built-in operators;
    "when performing anonymization, if ... 'DEFAULT' key is not stated,
    the default anonymization operator is 'replace' for all entities").

Deliberately NOT wired into src/documents/service.py's document_profile.json:
that file's top-level key set and each pii_findings/entities entry's key
set are pinned exactly by tests/contract/test_document_profile_contract.py
(`assert set(profile) == REQUIRED_TOP_LEVEL_KEYS`, ditto per-finding) -
adding a policy_action key there would silently widen an already-locked
contract. This module is a standalone, additive utility instead: any
future consumer that wants graduated auto-redact-vs-review behavior (a
review-queue UI, a batch job) calls apply_pii_policy() on findings it
already has, without this project's core PII/document contracts changing
at all.
"""

from __future__ import annotations

from typing import Any

ACTION_REDACT = "redact"
ACTION_FLAG_FOR_REVIEW = "flag_for_review"

#: Per pii_type minimum confidence required to auto-redact without a
#: human-review flag - chosen to track src/documents/pii.py's own
#: confidence tiers: types with a required checksum gate (routing_number)
#: or a boosted-on-success checksum (credit_card, iban) already carry a
#: higher confidence floor when the checksum passes; types with no
#: checksum at all get a stricter default so a borderline match is
#: reviewed rather than auto-redacted (or worse, auto-allowed).
DEFAULT_MIN_CONFIDENCE_FOR_AUTO_REDACT: dict[str, float] = {
    "email": 0.9,
    "url": 0.9,
    "ssn": 0.6,
    "credit_card": 0.5,
    "ip_address": 0.85,
    "phone_number": 0.7,
    "street_address": 0.7,
    "iban": 0.6,
    "routing_number": 0.8,
}
_FALLBACK_MIN_CONFIDENCE = 0.9


def apply_pii_policy(
    findings: list[dict[str, Any]],
    min_confidence_by_type: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Return new finding dicts augmented with a "policy_action" key -
    REDACT for anything at/above its type's threshold, FLAG_FOR_REVIEW
    otherwise. Never mutates the input findings or drops any existing
    key - purely additive, so detect_pii()'s own contract is untouched.
    """
    thresholds = min_confidence_by_type or DEFAULT_MIN_CONFIDENCE_FOR_AUTO_REDACT
    policed = []
    for finding in findings:
        threshold = thresholds.get(finding["type"], _FALLBACK_MIN_CONFIDENCE)
        action = ACTION_REDACT if finding["confidence"] >= threshold else ACTION_FLAG_FOR_REVIEW
        policed.append({**finding, "policy_action": action})
    return policed


def summarize_policy_actions(policed_findings: list[dict[str, Any]]) -> dict[str, int]:
    """Counts per action, e.g. {"redact": 5, "flag_for_review": 1}."""
    counts: dict[str, int] = {}
    for finding in policed_findings:
        counts[finding["policy_action"]] = counts.get(finding["policy_action"], 0) + 1
    return counts
