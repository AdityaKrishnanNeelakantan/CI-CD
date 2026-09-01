from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.pdf_classifier import classify_pdf_type

pytestmark = pytest.mark.unit


def _preflight(native_char_count=0, has_images=False, has_form_fields=False):
    return {
        "page_count": 1,
        "native_char_count": native_char_count,
        "has_images": has_images,
        "is_encrypted": False,
        "has_form_fields": has_form_fields,
    }


def test_form_fields_take_priority():
    result = classify_pdf_type(_preflight(native_char_count=500, has_images=True, has_form_fields=True))
    assert result["pdf_type"] == "form"


def test_zero_native_text_is_scanned():
    result = classify_pdf_type(_preflight(native_char_count=0, has_images=True))
    assert result["pdf_type"] == "scanned"


def test_zero_native_text_without_images_is_still_scanned():
    result = classify_pdf_type(_preflight(native_char_count=0, has_images=False))
    assert result["pdf_type"] == "scanned"


def test_native_text_without_images_is_text():
    result = classify_pdf_type(_preflight(native_char_count=200, has_images=False))
    assert result["pdf_type"] == "text"


def test_native_text_with_images_is_mixed():
    result = classify_pdf_type(_preflight(native_char_count=200, has_images=True))
    assert result["pdf_type"] == "mixed"


def test_result_always_has_evidence_and_confidence():
    result = classify_pdf_type(_preflight())
    assert result["evidence"]
    assert 0.0 < result["confidence"] <= 1.0
    assert result["status"] == "proposed"
