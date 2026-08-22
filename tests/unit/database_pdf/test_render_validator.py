from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.pdf_renderer import render_document_pdf
from synth_platform.engine.documents.pdf.render_validator import validate_rendered_document

pytestmark = pytest.mark.unit


def test_valid_render_passes_all_checks(tmp_path: Path):
    template = {
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "regions": [
                    {"region_id": "p1_r0", "region_type": "field", "reading_order_index": 0, "label": "Account Number"}
                ],
            }
        ],
    }
    values = {"fields": {"p1_r0": {"label": "Account Number", "value": "5551234"}}, "tables": {}, "inline_spans": {}}
    pdf_path = tmp_path / "twin.pdf"
    ground_truth = render_document_pdf(template, {"bindings": []}, values, str(pdf_path))

    report = validate_rendered_document(pdf_path, ground_truth)
    assert report["hard_checks_passed"] is True
    assert report["field_accuracy"] == 1.0
    assert report["unmatched_fields"] == []
    assert report["overflow_failure_count"] == 0


def test_tampered_pdf_produces_a_lower_field_accuracy(tmp_path: Path):
    """If the ground truth claims a value that was never actually
    rendered, the round-trip check must catch it rather than trivially
    pass.
    """
    template = {
        "page_count": 1,
        "pages": [
            {"page_number": 1, "regions": [{"region_id": "p1_r0", "region_type": "field", "reading_order_index": 0}]}
        ],
    }
    values = {"fields": {"p1_r0": {"label": "Account Number", "value": "5551234"}}, "tables": {}, "inline_spans": {}}
    pdf_path = tmp_path / "twin.pdf"
    ground_truth = render_document_pdf(template, {"bindings": []}, values, str(pdf_path))

    # Tamper: claim a value that was never rendered.
    ground_truth["regions"][0]["rendered_text"] = "Account Number: this-was-never-rendered"

    report = validate_rendered_document(pdf_path, ground_truth)
    assert report["hard_checks_passed"] is False
    assert report["field_accuracy"] < 1.0
    assert "p1_r0" in report["unmatched_fields"]


def test_table_cell_wider_than_recorded_column_is_flagged_as_overflow(tmp_path: Path):
    template = {
        "page_count": 1,
        "pages": [
            {"page_number": 1, "regions": [{"region_id": "p1_r0", "region_type": "table", "reading_order_index": 0, "column_count": 1}]}
        ],
    }
    values = {"fields": {}, "tables": {"p1_r0": [[{"value": "short"}]]}, "inline_spans": {}}
    pdf_path = tmp_path / "twin.pdf"
    ground_truth = render_document_pdf(template, {"bindings": []}, values, str(pdf_path))

    # Tamper: shrink the recorded column width below what the cell text needs.
    ground_truth["regions"][0]["column_widths"] = [0.1]

    report = validate_rendered_document(pdf_path, ground_truth)
    assert report["hard_checks_passed"] is False
    assert report["overflow_failure_count"] == 1


def test_empty_ground_truth_trivially_passes(tmp_path: Path):
    template = {"page_count": 1, "pages": [{"page_number": 1, "regions": []}]}
    values = {"fields": {}, "tables": {}, "inline_spans": {}}
    pdf_path = tmp_path / "twin.pdf"
    ground_truth = render_document_pdf(template, {"bindings": []}, values, str(pdf_path))

    report = validate_rendered_document(pdf_path, ground_truth)
    assert report["hard_checks_passed"] is True
    assert report["total_fields"] == 0
    assert report["field_accuracy"] == 1.0


def test_heading_and_paragraph_round_trip_are_checked(tmp_path: Path):
    template = {
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "regions": [
                    {"region_id": "p1_r0", "region_type": "heading", "reading_order_index": 0, "text": "Statement Summary"},
                    {"region_id": "p1_r1", "region_type": "paragraph", "reading_order_index": 1, "text": "Thank you for banking with us."},
                ],
            }
        ],
    }
    values = {"fields": {}, "tables": {}, "inline_spans": {}}
    pdf_path = tmp_path / "twin.pdf"
    ground_truth = render_document_pdf(template, {"bindings": []}, values, str(pdf_path))

    report = validate_rendered_document(pdf_path, ground_truth)
    assert report["hard_checks_passed"] is True
    assert report["total_fields"] == 2
    assert report["unmatched_fields"] == []


def test_tampered_heading_text_is_caught_by_round_trip_check(tmp_path: Path):
    template = {
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "regions": [
                    {"region_id": "p1_r0", "region_type": "heading", "reading_order_index": 0, "text": "Statement Summary"}
                ],
            }
        ],
    }
    values = {"fields": {}, "tables": {}, "inline_spans": {}}
    pdf_path = tmp_path / "twin.pdf"
    ground_truth = render_document_pdf(template, {"bindings": []}, values, str(pdf_path))

    # Tamper: claim heading text that was never actually rendered.
    ground_truth["regions"][0]["rendered_text"] = "this-heading-was-never-rendered"

    report = validate_rendered_document(pdf_path, ground_truth)
    assert report["hard_checks_passed"] is False
    assert "p1_r0" in report["unmatched_fields"]


def test_line_wrap_newlines_do_not_fail_round_trip_presence(tmp_path: Path, monkeypatch):
    """PDF extractors often insert newlines where the renderer wrapped text.
    Validation must compare whitespace-normalized content, not raw bytes.
    """
    from synth_platform.engine.documents.pdf import render_validator as rv

    ground_truth = {
        "regions": [
            {
                "region_id": "p1_r0",
                "region_type": "paragraph",
                "rendered_text": "Kimberly Lawrence, born on 06/17/2026, is a 46-year-old Female.",
            }
        ]
    }

    def _fake_extract(_path):
        return {
            "text": "Kimberly Lawrence, born on 06/17/2026,\nis a 46-year-old Female.",
            "extraction_warnings": [],
        }

    monkeypatch.setattr(rv.PDFDocumentAdapter, "extract", lambda self, path: _fake_extract(path))
    report = validate_rendered_document(tmp_path / "unused.pdf", ground_truth)
    assert report["hard_checks_passed"] is True
    assert report["field_accuracy"] == 1.0
    assert report["unmatched_fields"] == []
