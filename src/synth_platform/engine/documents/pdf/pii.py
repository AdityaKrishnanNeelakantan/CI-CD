"""Regex-based PII detection with exact character offsets.

Deliberately rule-based rather than model-based, for the same reason as
src/inference/engine.py: confidence must come from explicit, inspectable
patterns, not an opaque model. Findings never carry the raw matched text -
only offsets into the normalised document (a caller holding the original
text can look the span up itself) plus a masked redaction preview, so
persisted evidence never leaks the actual PII value.

Known limitation: these patterns are US/English-centric by construction
(e.g. SSN as \\d{3}-\\d{2}-\\d{4}, the +1 country-code phone prefix, and
English street suffixes such as "Street"/"Avenue" in _STREET_SUFFIXES).
Non-US identifiers and address formats will generally not be detected.
Documented here rather than silently accepted, consistent with this
project's overall approach of making rule-based gaps explicit.

Some patterns are additionally checksum-validated (credit_card via Luhn,
iban via ISO 7064 mod-97, routing_number via the ABA weighted checksum) -
see each validator function's own docstring for whether a failed check
rejects the match (routing_number, whose bare digit-run pattern would
otherwise be far too broad) or only leaves its confidence unboosted
(credit_card, iban - regex-shaped but checksum-invalid is still exactly
as sensitive to leave unmasked).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

PII_TYPE_EMAIL = "email"
PII_TYPE_URL = "url"
PII_TYPE_SSN = "ssn"
PII_TYPE_CREDIT_CARD = "credit_card"
PII_TYPE_IP_ADDRESS = "ip_address"
PII_TYPE_PHONE = "phone_number"
PII_TYPE_STREET_ADDRESS = "street_address"
PII_TYPE_IBAN = "iban"
PII_TYPE_ROUTING_NUMBER = "routing_number"

_STREET_SUFFIXES = (
    r"Street|St\.?|Avenue|Ave\.?|Drive|Dr\.?|Road|Rd\.?|Boulevard|Blvd\.?|"
    r"Lane|Ln\.?|Court|Ct\.?|Way|Place|Pl\.?|Circle|Cir\.?|Terrace|Ter\.?|"
    r"Highway|Hwy\.?"
)


def _luhn_is_valid(matched_text: str) -> bool:
    """Standard Luhn/mod-10 check digit, same algorithm real card networks
    use - lets a match's confidence reflect whether it's a structurally
    plausible card number, not just "looks like 4 groups of 4 digits".
    Never used to reject a match (a failed check is still redacted at the
    pattern's base confidence): a card-shaped number that fails Luhn is
    still exactly as sensitive to leave unmasked as one that passes.
    """
    digits = re.sub(r"\D", "", matched_text)
    total = 0
    for index, ch in enumerate(reversed(digits)):
        n = int(ch)
        if index % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _iban_is_valid(matched_text: str) -> bool:
    """ISO 7064 mod-97-10 check used by every real IBAN."""
    rearranged = matched_text[4:] + matched_text[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def _aba_routing_checksum_is_valid(matched_text: str) -> bool:
    """ABA routing-transit-number checksum (weights 3/7/1 repeating). A
    bare 9-digit run is otherwise indistinguishable from countless other
    9-digit values (a reference number, a zip+4, part of a longer ID), so
    unlike credit-card/IBAN this recognizer's checksum is a required gate,
    not just a confidence booster - without it this pattern would fire on
    almost any 9 consecutive digits.
    """
    weights = (3, 7, 1, 3, 7, 1, 3, 7, 1)
    total = sum(int(d) * w for d, w in zip(matched_text, weights))
    return total % 10 == 0


# Ordered by specificity: earlier patterns win when spans overlap (e.g. an
# email's domain must never be double-counted as part of a URL match).
# The optional 4th/5th elements are a checksum validator and whether that
# validator is a *required* gate (reject on failure) or just a confidence
# booster (accept regardless, raise confidence only on success) - see
# each validator's own docstring for which category it is and why.
_PATTERNS: list[tuple[str, re.Pattern[str], float, Callable[[str], bool] | None, bool]] = [
    (PII_TYPE_EMAIL, re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0.95, None, False),
    (PII_TYPE_URL, re.compile(r"\bhttps?://[^\s<>\"']+"), 0.95, None, False),
    (PII_TYPE_SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0.7, None, False),
    (
        # IBAN is claimed before credit_card/phone/street_address on purpose:
        # its digit tail (e.g. "60161331926819") would otherwise be partially
        # swallowed by the phone pattern first (that pattern has no leading
        # \b, by necessity, so it can match "(555) ..." - verified empirically
        # via test_iban_is_found_with_exact_offsets_and_boosted_confidence,
        # which failed with 0 findings until this pattern was moved here).
        PII_TYPE_IBAN,
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        0.7,
        _iban_is_valid,
        False,
    ),
    (PII_TYPE_CREDIT_CARD, re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b"), 0.6, _luhn_is_valid, False),
    (
        PII_TYPE_IP_ADDRESS,
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"),
        0.9,
        None,
        False,
    ),
    (PII_TYPE_PHONE, re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), 0.75, None, False),
    (
        # A house/building number followed by 1-4 capitalised words and a
        # street-type suffix ("1765 Sheridan Drive", "1765 SHERIDAN DRIVE")
        # - real gap found via smoke-testing against an actual scanned bank
        # statement, whose letterhead address leaked through unmasked.
        PII_TYPE_STREET_ADDRESS,
        re.compile(
            rf"\b\d{{1,6}}\s+[A-Za-z]+(?:\s+[A-Za-z]+){{0,3}}\s+(?:{_STREET_SUFFIXES})\b",
            re.IGNORECASE,
        ),
        0.75,
        None,
        False,
    ),
    (PII_TYPE_ROUTING_NUMBER, re.compile(r"\b\d{9}\b"), 0.85, _aba_routing_checksum_is_valid, True),
]


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def detect_pii(text: str) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    for pii_type, pattern, confidence, validator, validator_required in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < o_end and end > o_start for o_start, o_end in occupied):
                continue  # overlaps a higher-priority match already accepted

            matched_confidence = confidence
            if validator is not None:
                checksum_is_valid = validator(match.group(0))
                if validator_required and not checksum_is_valid:
                    continue  # e.g. a random 9-digit number that fails the ABA checksum is not a routing number
                if checksum_is_valid:
                    matched_confidence = min(0.99, confidence + 0.2)

            occupied.append((start, end))
            accepted.append(
                {
                    "type": pii_type,
                    "start": start,
                    "end": end,
                    "confidence": matched_confidence,
                    "redaction_preview": _mask(match.group(0)),
                }
            )

    accepted.sort(key=lambda finding: finding["start"])
    return accepted
