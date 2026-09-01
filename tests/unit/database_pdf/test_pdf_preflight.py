from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.errors import ExtractionError
from synth_platform.engine.documents.pdf.preflight import run_pdf_preflight
from tests.fixtures.pdf_factory import make_acroform_pdf, make_pdf, make_scanned_pdf

pytestmark = pytest.mark.unit


def test_text_pdf_has_native_chars_and_no_images(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Some real embedded text content."])
    preflight = run_pdf_preflight(pdf_path)
    assert preflight["page_count"] == 1
    assert preflight["native_char_count"] > 0
    assert preflight["has_images"] is False
    assert preflight["is_encrypted"] is False
    assert preflight["has_form_fields"] is False


def test_scanned_pdf_has_zero_native_chars_and_images(tmp_path: Path):
    pdf_path = make_scanned_pdf(tmp_path / "scan.pdf", [["Some rendered text."]])
    preflight = run_pdf_preflight(pdf_path)
    assert preflight["native_char_count"] == 0
    assert preflight["has_images"] is True


def test_multi_page_counts_all_pages(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["First page.", "Second page.", "Third page."])
    preflight = run_pdf_preflight(pdf_path)
    assert preflight["page_count"] == 3


def test_corrupt_pdf_raises_extraction_error(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4 not a real structure" + b"\x00" * 50)
    with pytest.raises(ExtractionError):
        run_pdf_preflight(corrupt_path)


def test_acroform_field_values_count_toward_native_char_count(tmp_path: Path):
    """Regression: page.extract_text() cannot see AcroForm field values at
    all (verified empirically - a real filled field extracts as
    completely empty text), so a genuinely data-bearing fillable form
    used to report native_char_count=0 and misroute to a useless OCR
    fallback despite having real, readable field data.
    """
    pdf_path = make_acroform_pdf(
        tmp_path / "acroform.pdf",
        [("customer_name", "Ada Lovelace", [100, 700, 300, 720])],
    )
    preflight = run_pdf_preflight(pdf_path)
    assert preflight["has_form_fields"] is True
    assert preflight["native_char_count"] >= len("Ada Lovelace")
