from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.errors import ExtractionError
from synth_platform.engine.documents.pdf.native_layout import (
    extract_acroform_field_spans,
    extract_acroform_field_values,
    extract_pdf_info_metadata,
    extract_positioned_spans,
)
from tests.fixtures.pdf_factory import make_acroform_pdf, make_pdf, make_pdf_with_metadata

pytestmark = pytest.mark.unit


def test_extracts_at_least_one_span_per_populated_page(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["First page text.", "Second page text."])
    spans = extract_positioned_spans(pdf_path)
    pages_seen = {s["page_number"] for s in spans}
    assert pages_seen == {1, 2}


def test_span_coordinates_are_fractional_and_top_left_origin(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Some text near the top of the page."])
    spans = extract_positioned_spans(pdf_path)
    assert spans
    for span in spans:
        assert 0.0 <= span["x0"] <= 1.0
        assert 0.0 <= span["y0"] <= 1.0
        assert 0.0 <= span["x1"] <= 1.0
        assert 0.0 <= span["y1"] <= 1.0
        assert span["y1"] >= span["y0"]


def test_blank_page_produces_no_spans(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "blank.pdf", [""])
    spans = extract_positioned_spans(pdf_path)
    assert spans == []


def test_corrupt_pdf_raises_extraction_error(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4 not a real structure" + b"\x00" * 50)
    with pytest.raises(ExtractionError):
        extract_positioned_spans(corrupt_path)


def test_extract_positioned_spans_never_sees_acroform_field_values(tmp_path: Path):
    """Documents the actual limit of extract_positioned_spans, the same
    way the module docstring's claim was verified empirically: field
    values are outside the content stream entirely, so the ordinary text
    extractor returns nothing for a page whose only content is a filled
    form field - this is exactly why extract_acroform_field_spans exists
    as a separate, dedicated path."""
    pdf_path = make_acroform_pdf(
        tmp_path / "acroform.pdf", [("customer_name", "Ada Lovelace", [100, 700, 300, 720])]
    )
    assert extract_positioned_spans(pdf_path) == []


def test_extract_acroform_field_values_reads_name_and_value(tmp_path: Path):
    pdf_path = make_acroform_pdf(
        tmp_path / "acroform.pdf",
        [
            ("customer_name", "Ada Lovelace", [100, 700, 300, 720]),
            ("account_number", "8823471", [100, 650, 300, 670]),
        ],
    )
    fields = extract_acroform_field_values(pdf_path)
    by_name = {f["name"]: f["value"] for f in fields}
    assert by_name == {"customer_name": "Ada Lovelace", "account_number": "8823471"}
    assert all(f["page_number"] == 1 for f in fields)


def test_extract_acroform_field_spans_are_fractional_and_positioned(tmp_path: Path):
    pdf_path = make_acroform_pdf(
        tmp_path / "acroform.pdf", [("customer_name", "Ada Lovelace", [100, 700, 300, 720])]
    )
    spans = extract_acroform_field_spans(pdf_path)
    assert len(spans) == 1
    span = spans[0]
    assert span["text"] == "Customer Name: Ada Lovelace"
    assert span["page_number"] == 1
    assert 0.0 <= span["x0"] < span["x1"] <= 1.0
    assert 0.0 <= span["y0"] < span["y1"] <= 1.0


def test_acroform_field_with_empty_value_is_skipped(tmp_path: Path):
    pdf_path = make_acroform_pdf(
        tmp_path / "acroform.pdf",
        [("blank_field", "", [100, 700, 300, 720]), ("filled_field", "value", [100, 650, 300, 670])],
    )
    fields = extract_acroform_field_values(pdf_path)
    assert {f["name"] for f in fields} == {"filled_field"}


def test_extract_pdf_info_metadata_reads_populated_fields(tmp_path: Path):
    pdf_path = make_pdf_with_metadata(
        tmp_path / "meta.pdf",
        ["Body text."],
        author="Ada Lovelace",
        title="Confidential Report",
    )
    entries = extract_pdf_info_metadata(pdf_path)
    by_name = {e["name"]: e["value"] for e in entries}
    assert by_name == {"Author": "Ada Lovelace", "Title": "Confidential Report"}


def test_extract_pdf_info_metadata_with_no_fields_set_returns_empty(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "plain.pdf", ["Body text."])
    assert extract_pdf_info_metadata(pdf_path) == []
