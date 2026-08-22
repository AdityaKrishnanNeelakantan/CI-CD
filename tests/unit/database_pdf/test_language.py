from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.language import STATUS_PROPOSED, STATUS_REVIEW_REQUIRED, detect_language

pytestmark = pytest.mark.unit


def test_english_text_detected_confidently():
    result = detect_language(
        "This is a reasonably long piece of English text used to test language detection."
    )
    assert result["language"] == "en"
    assert result["status"] == STATUS_PROPOSED
    assert result["confidence"] >= 0.8


def test_french_text_detected_confidently():
    result = detect_language(
        "Ceci est un texte en français pour tester la detection de la langue automatique."
    )
    assert result["language"] == "fr"
    assert result["status"] == STATUS_PROPOSED


def test_short_text_requires_review():
    result = detect_language("Hi.")
    assert result["status"] == STATUS_REVIEW_REQUIRED
    assert result["language"] is None
    assert result["confidence"] == 0.0


def test_empty_text_requires_review():
    result = detect_language("")
    assert result["status"] == STATUS_REVIEW_REQUIRED


def test_detection_is_deterministic_across_calls():
    text = "This is a reasonably long piece of English text used to test language detection."
    first = detect_language(text)
    second = detect_language(text)
    assert first == second
