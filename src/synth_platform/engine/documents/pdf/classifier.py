"""Rule-based document type classification.

Uses the same weighted-evidence, confidence, and REVIEW_REQUIRED pattern
as src/inference/engine.py's SemanticInferenceEngine, deliberately: one
consistent, explainable classification approach across the whole project
rather than a different opaque method per checkpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DOCUMENT_TYPES = frozenset({"invoice", "contract", "letter", "report", "resume"})
STATUS_PROPOSED = "proposed"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class ClassifierConfig:
    min_confidence: float = 0.5
    ambiguity_margin: float = 0.15
    keyword_weight: float = 2.0


_KEYWORDS: dict[str, list[str]] = {
    "invoice": [
        "invoice", "invoice number", "total due", "amount due", "bill to",
        "subtotal", "payment terms",
    ],
    "contract": [
        "agreement", "whereas", "hereby", "the parties", "terms and conditions",
        "governing law", "witness whereof",
    ],
    "letter": ["dear ", "sincerely", "yours truly", "best regards", "kind regards"],
    "report": ["executive summary", "introduction", "conclusion", "findings", "methodology", "abstract"],
    "resume": [
        "curriculum vitae", "work experience", "professional experience",
        "education", "skills", "references available",
    ],
}

_CURRENCY_PATTERN = re.compile(r"[$€£]\s?\d[\d,]*\.?\d*")


def classify_document(text: str, config: ClassifierConfig | None = None) -> dict[str, Any]:
    config = config or ClassifierConfig()
    lowered = text.lower()

    scores: dict[str, float] = {doc_type: 0.0 for doc_type in DOCUMENT_TYPES}
    evidence: list[str] = []

    for doc_type, keywords in _KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[doc_type] += config.keyword_weight
                evidence.append(f"keyword({doc_type})={keyword.strip()}")

    if _CURRENCY_PATTERN.search(text):
        scores["invoice"] += config.keyword_weight
        evidence.append("currency_amount_pattern")

    total = sum(scores.values())
    if total <= 0:
        return {
            "document_type": "other",
            "status": STATUS_REVIEW_REQUIRED,
            "confidence": 0.0,
            "evidence": ["no_signal"],
            "alternatives": [],
        }

    ranked = sorted(((t, s) for t, s in scores.items() if s > 0), key=lambda item: item[1], reverse=True)
    top_type, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    confidence = round(top_score / total, 6)
    second_confidence = round(second_score / total, 6)
    is_ambiguous = confidence < config.min_confidence or (
        confidence - second_confidence
    ) < config.ambiguity_margin

    alternatives = [
        {"document_type": t, "confidence": round(s / total, 6)} for t, s in ranked[1:4]
    ]

    return {
        "document_type": top_type,
        "status": STATUS_REVIEW_REQUIRED if is_ambiguous else STATUS_PROPOSED,
        "confidence": confidence,
        "evidence": evidence,
        "alternatives": alternatives,
    }
