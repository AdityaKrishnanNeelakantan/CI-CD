"""Language detection: a single cheap signal, not a certainty claim.

Uses langdetect (seeded for determinism - its underlying algorithm samples
n-grams and is otherwise non-deterministic run to run). Short or ambiguous
text is explicitly flagged REVIEW_REQUIRED rather than trusted at face
value, matching the same proposal/confidence/evidence/status shape used
throughout this project (see src/inference/engine.py).
"""

from __future__ import annotations

from typing import Any

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

MIN_TEXT_LENGTH_FOR_DETECTION = 20
MIN_CONFIDENCE = 0.8

STATUS_PROPOSED = "proposed"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"


def detect_language(text: str) -> dict[str, Any]:
    stripped = text.strip()

    if len(stripped) < MIN_TEXT_LENGTH_FOR_DETECTION:
        return {
            "language": None,
            "confidence": 0.0,
            "status": STATUS_REVIEW_REQUIRED,
            "evidence": [f"text_length={len(stripped)}_below_minimum={MIN_TEXT_LENGTH_FOR_DETECTION}"],
        }

    try:
        candidates = detect_langs(stripped)
    except LangDetectException:
        return {
            "language": None,
            "confidence": 0.0,
            "status": STATUS_REVIEW_REQUIRED,
            "evidence": ["language_detection_failed"],
        }

    top = candidates[0]
    confidence = round(float(top.prob), 6)
    status = STATUS_PROPOSED if confidence >= MIN_CONFIDENCE else STATUS_REVIEW_REQUIRED

    return {
        "language": top.lang,
        "confidence": confidence,
        "status": status,
        "evidence": [f"detector_probability={confidence}"],
    }
