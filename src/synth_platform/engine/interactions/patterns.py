"""Workflow-specific sensitive-value patterns for customer interactions."""

from __future__ import annotations

import re

_IDENTIFIER_LABELS = r"account|member|customer|case|ticket|order|reference|ref"
_IDENTIFIER_VALUE = r"([A-Z0-9](?:[A-Z0-9._/-]{2,62}[A-Z0-9]))"
_IDENTIFIER_WITH_DIGIT = (
    r"((?=[A-Z0-9._/-]{4,64}(?:\s|$))(?=[A-Z0-9._/-]*\d)"
    r"[A-Z0-9](?:[A-Z0-9._/-]{2,62}[A-Z0-9]))"
)

INTERACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "account_id",
        re.compile(
            rf"(?i)\b(?:{_IDENTIFIER_LABELS})\s*"
            rf"(?:id|number|no\.?|#)\s*(?:is|=|:|#|-)?\s*{_IDENTIFIER_VALUE}"
        ),
    ),
    (
        "account_id",
        re.compile(
            rf"(?i)\b(?:{_IDENTIFIER_LABELS})\s*(?:=|:|#)\s*"
            rf"{_IDENTIFIER_VALUE}"
        ),
    ),
    (
        "account_id",
        re.compile(rf"(?i)\b(?:{_IDENTIFIER_LABELS})\s+{_IDENTIFIER_WITH_DIGIT}"),
    ),
    (
        "secret",
        re.compile(
            r"(?i)\b(?:password|passcode|pin|otp|token|secret)"
            r"\s*(?:is|=|:)?\s*[\"']([^\"'\r\n]{1,128})[\"']"
        ),
    ),
    (
        "secret",
        re.compile(
            r"(?i)\b(?:password|passcode|pin|otp|token|secret)"
            r"\s*(?:is|=|:)\s*(?![\"'])([^\s,;]{4,128})"
        ),
    ),
    (
        "secret",
        re.compile(
            r"(?i)\b(?:password|passcode|pin|otp|token|secret)\s+"
            r"(?![\"'])(?=[^\s,;]{4,128}(?:\s|[,;]|$))"
            r"(?=[^\s,;]*[0-9@#$%^&*_=+:/\\-])([^\s,;]{4,128})"
        ),
    ),
)

# These labels are metadata rather than speakers. Generic participant labels
# such as Customer and Member are deliberately excluded.
SENSITIVE_FIELD_LABELS = {
    "account",
    "account id",
    "account number",
    "case",
    "case id",
    "customer id",
    "member id",
    "order",
    "order id",
    "otp",
    "passcode",
    "password",
    "pin",
    "ref",
    "reference",
    "reference id",
    "secret",
    "ticket",
    "ticket id",
    "token",
}

__all__ = ["INTERACTION_PATTERNS", "SENSITIVE_FIELD_LABELS"]
