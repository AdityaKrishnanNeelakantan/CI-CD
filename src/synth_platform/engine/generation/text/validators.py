"""Privacy and safety validation for LLM-generated text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Set

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
PHONEISH_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
IDISH_RE = re.compile(r"\b(?:id|ref|acct|account|routing|iban|ssn|aadhaar|card)\s*[:#]?\s*\S+", re.I)
FORBIDDEN_PHRASES = (
    "ignore previous",
    "system prompt",
    "as an ai",
    "openai",
    "password is",
    "api key",
)

SENSITIVE_TOKEN_RE = re.compile(
    r"\b("
    r"ssn|social security|aadhaar|aadhar|passport|account number|routing number|"
    r"card number|cvv|password|secret|private key|email|phone|address|"
    r"tax id|national id|salary|income"
    r")\b",
    re.IGNORECASE,
)

CONTRADICTION_RULES = (
    (("delayed", "late", "pending"), ("resolved immediately", "no delay", "on time")),
    (("high", "urgent", "critical"), ("low priority", "not urgent")),
    (("closed", "completed"), ("still open", "in progress")),
)


@dataclass
class ValidationResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "reasons": list(self.reasons)}


def validate_generated_text(
    text: Any,
    *,
    source_values: Optional[Set[str]] = None,
    max_words: int = 80,
    min_words: int = 3,
    allow_digits: bool = False,
    safe_context: Optional[dict] = None,
) -> ValidationResult:
    """Validate one generated text value."""
    reasons: List[str] = []
    if not isinstance(text, str):
        return ValidationResult(passed=False, reasons=["not a string"])
    candidate = text.strip().strip('"').strip("'")
    if not candidate:
        return ValidationResult(passed=False, reasons=["empty output"])

    words = candidate.split()
    if len(words) < min_words:
        reasons.append("too short")
    if len(words) > max_words:
        reasons.append("excessive length")

    if EMAIL_RE.search(candidate) or URL_RE.search(candidate):
        reasons.append("email or url")
    if PHONEISH_RE.search(candidate):
        reasons.append("phone-like")
    if not allow_digits and any(ch.isdigit() for ch in candidate):
        reasons.append("contains digits")
    if SENSITIVE_TOKEN_RE.search(candidate):
        reasons.append("sensitive token")
    if IDISH_RE.search(candidate):
        reasons.append("id-like string")

    lowered = candidate.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            reasons.append("forbidden phrase")

    if source_values and candidate in source_values:
        reasons.append("source replay")

    if safe_context:
        ctx_text = " ".join(str(v).lower() for v in safe_context.values())
        for positive, negatives in CONTRADICTION_RULES:
            if any(p in ctx_text for p in positive):
                if any(n in lowered for n in negatives):
                    reasons.append("context contradiction")

    return ValidationResult(passed=not reasons, reasons=reasons)


def sanitize_text(text: str, *, max_chars: int = 500) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) > max_chars:
        cut = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
        cleaned = cut if cut.endswith(".") else f"{cut}."
    return cleaned
