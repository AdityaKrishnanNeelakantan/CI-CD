from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.pdf_renderer import render_document_pdf

pytestmark = pytest.mark.unit


def _template(regions):
    return {"page_count": 1, "pages": [{"page_number": 1, "regions": regions}]}


def _field_region(region_id, label, index):
    return {
        "region_id": region_id,
        "region_type": "field",
        "reading_order_index": index,
        "label": label,
        "value_type": "text",
    }


def _binding_map(bindings):
    return {"bindings": bindings}


def test_render_produces_a_readable_pdf_file(tmp_path: Path):
    template = _template([_field_region("p1_r0", "Account Number", 0)])
    binding_map = _binding_map([{"region_id": "p1_r0", "binding_type": "field", "label": "Account Number"}])
    values = {"fields": {"p1_r0": {"label": "Account Number", "value": "5551234"}}, "tables": {}, "inline_spans": {}}

    output_path = tmp_path / "twin.pdf"
    ground_truth = render_document_pdf(template, binding_map, values, str(output_path))

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert ground_truth["page_count"] == 1


def test_field_ground_truth_records_label_and_rendered_value(tmp_path: Path):
    template = _template([_field_region("p1_r0", "Total Due", 0)])
    binding_map = _binding_map([{"region_id": "p1_r0", "binding_type": "field", "label": "Total Due"}])
    values = {
        "fields": {"p1_r0": {"label": "Total Due", "semantic_role": "currency_amount", "formatted": "$401.50"}},
        "tables": {},
        "inline_spans": {},
    }

    ground_truth = render_document_pdf(template, binding_map, values, str(tmp_path / "twin.pdf"))
    region = ground_truth["regions"][0]
    assert region["label"] == "Total Due"
    assert region["rendered_text"] == "Total Due: $401.50"
    assert region["rendered_value"] == "$401.50"


def test_heading_and_paragraph_are_rendered_verbatim(tmp_path: Path):
    template = _template(
        [
            {"region_id": "p1_r0", "region_type": "heading", "reading_order_index": 0, "text": "SUMMARY"},
            {"region_id": "p1_r1", "region_type": "paragraph", "reading_order_index": 1, "text": "Some static wording."},
        ]
    )
    values = {"fields": {}, "tables": {}, "inline_spans": {}}
    ground_truth = render_document_pdf(template, _binding_map([]), values, str(tmp_path / "twin.pdf"))
    texts = [r["rendered_text"] for r in ground_truth["regions"]]
    assert "SUMMARY" in texts
    assert "Some static wording." in texts


def test_inline_variable_spans_are_spliced_into_paragraph_text(tmp_path: Path):
    template = _template(
        [
            {
                "region_id": "p1_r0",
                "region_type": "paragraph",
                "reading_order_index": 0,
                "text": "Beginning balance 69*96",
                "inline_variable_spans": [{"start": 18, "end": 23}],
            }
        ]
    )
    values = {
        "fields": {},
        "tables": {},
        "inline_spans": {"p1_r0": [{"start": 18, "end": 23, "value": "12.34"}]},
    }
    ground_truth = render_document_pdf(template, _binding_map([]), values, str(tmp_path / "twin.pdf"))
    assert ground_truth["regions"][0]["rendered_text"] == "Beginning balance 12.34"


def test_table_rows_render_each_cell_and_report_column_widths(tmp_path: Path):
    template = _template(
        [{"region_id": "p1_r0", "region_type": "table", "reading_order_index": 0, "column_count": 2}]
    )
    values = {
        "fields": {},
        "tables": {
            "p1_r0": [
                [{"value": "Fee A"}, {"formatted": "$10.00"}],
                [{"value": "Fee B"}, {"formatted": "$2,000.00"}],
            ]
        },
        "inline_spans": {},
    }
    ground_truth = render_document_pdf(template, _binding_map([]), values, str(tmp_path / "twin.pdf"))
    table_region = ground_truth["regions"][0]
    assert table_region["region_type"] == "table"
    assert table_region["column_count"] == 2
    assert len(table_region["rows"]) == 2
    assert table_region["rows"][0]["rendered_cells"] == ["Fee A", "$10.00"]
    assert len(table_region["column_widths"]) == 2


def test_regions_are_rendered_in_reading_order_across_pages(tmp_path: Path):
    template = {
        "page_count": 2,
        "pages": [
            {"page_number": 1, "regions": [_field_region("p1_r0", "First", 0)]},
            {"page_number": 2, "regions": [_field_region("p2_r0", "Second", 0)]},
        ],
    }
    values = {
        "fields": {
            "p1_r0": {"label": "First", "value": "1"},
            "p2_r0": {"label": "Second", "value": "2"},
        },
        "tables": {},
        "inline_spans": {},
    }
    ground_truth = render_document_pdf(template, _binding_map([]), values, str(tmp_path / "twin.pdf"))
    ordered_labels = [r["label"] for r in ground_truth["regions"]]
    assert ordered_labels == ["First", "Second"]


def test_null_field_value_renders_as_empty_string(tmp_path: Path):
    template = _template([_field_region("p1_r0", "Middle Name", 0)])
    values = {"fields": {"p1_r0": {"label": "Middle Name", "value": None}}, "tables": {}, "inline_spans": {}}
    ground_truth = render_document_pdf(template, _binding_map([]), values, str(tmp_path / "twin.pdf"))
    assert ground_truth["regions"][0]["rendered_text"] == "Middle Name: "


def test_smart_unicode_punctuation_never_crashes_rendering(tmp_path: Path):
    """Regression test: real-world PDFs (e.g. Word/PDF exporters) routinely
    contain curly quotes, em-dashes, and bullets. fpdf2's core Helvetica
    font is Latin-1 only and previously raised FPDFUnicodeEncodingException
    on these - found by running the pipeline against real sample PDFs
    (2 of 6 crashed at this stage). Common cases must be transliterated to
    a plausible ASCII equivalent (never crash, never dropped verbatim).
    """
    template = _template(
        [
            {
                "region_id": "p1_r0",
                "region_type": "heading",
                "reading_order_index": 0,
                "text": "It\u2019s the client\u2019s \u201cfinal\u201d offer \u2014 see \u2022 bullet below\u2026",
            }
        ]
    )
    values = {"fields": {}, "tables": {}, "inline_spans": {}}

    output_path = tmp_path / "twin.pdf"
    ground_truth = render_document_pdf(template, _binding_map([]), values, str(output_path))

    assert output_path.is_file()
    rendered_text = ground_truth["regions"][0]["rendered_text"]
    assert rendered_text == "It's the client's \"final\" offer - see - bullet below..."


def test_unsupported_unicode_characters_are_dropped_not_crashed(tmp_path: Path):
    template = _template(
        [
            {
                "region_id": "p1_r0",
                "region_type": "heading",
                "reading_order_index": 0,
                "text": "Priority \u25cf high",
            }
        ]
    )
    values = {"fields": {}, "tables": {}, "inline_spans": {}}

    output_path = tmp_path / "twin.pdf"
    ground_truth = render_document_pdf(template, _binding_map([]), values, str(output_path))

    assert output_path.is_file()
    assert "\u25cf" not in ground_truth["regions"][0]["rendered_text"]
