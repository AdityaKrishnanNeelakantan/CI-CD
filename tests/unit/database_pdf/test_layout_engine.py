from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.layout_engine import analyze_layout

pytestmark = pytest.mark.unit


def _span(text, x0, y0, x1, y1, page_number=1, confidence=None):
    return {"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "page_number": page_number, "confidence": confidence}


def test_empty_spans_produce_no_regions():
    layout = analyze_layout([], page_count=1)
    assert layout["pages"] == [{"page_number": 1, "regions": []}]


def test_two_spans_on_same_line_are_grouped_and_ordered_left_to_right():
    spans = [
        _span("World", 0.3, 0.1, 0.4, 0.12),
        _span("Hello", 0.1, 0.1005, 0.2, 0.12),
    ]
    layout = analyze_layout(spans, page_count=1)
    regions = layout["pages"][0]["regions"]
    assert len(regions) == 1
    assert regions[0]["lines"][0]["text"] == "Hello World"


def test_field_line_with_colon_gap_produces_two_cells():
    spans = [
        _span("Account", 0.05, 0.1, 0.12, 0.12),
        _span("Number:", 0.13, 0.1, 0.2, 0.12),
        _span("8823471", 0.4, 0.1, 0.48, 0.12),  # big x-gap -> new cell
    ]
    layout = analyze_layout(spans, page_count=1)
    line = layout["pages"][0]["regions"][0]["lines"][0]
    assert len(line["cells"]) == 2
    assert line["cells"][0] == "Account Number:"
    assert line["cells"][1] == "8823471"
    assert layout["pages"][0]["regions"][0]["region_type"] == "field"


def test_vertically_separated_lines_become_separate_regions():
    spans = [
        _span("First region.", 0.1, 0.1, 0.3, 0.12),
        _span("Second region.", 0.1, 0.5, 0.3, 0.52),  # far below -> new region
    ]
    layout = analyze_layout(spans, page_count=1)
    assert len(layout["pages"][0]["regions"]) == 2


def test_three_plus_multi_cell_lines_are_classified_as_table():
    spans = []
    for i, y in enumerate([0.1, 0.12, 0.14]):
        spans.append(_span(f"Row{i}", 0.05, y, 0.1, y + 0.015))
        spans.append(_span("100.00", 0.4, y, 0.48, y + 0.015))
    layout = analyze_layout(spans, page_count=1)
    assert layout["pages"][0]["regions"][0]["region_type"] == "table"


def test_reading_order_follows_top_to_bottom():
    spans = [
        _span("Bottom", 0.1, 0.5, 0.3, 0.52),
        _span("Top", 0.1, 0.1, 0.3, 0.12),
    ]
    layout = analyze_layout(spans, page_count=1)
    regions = layout["pages"][0]["regions"]
    assert [r["lines"][0]["text"] for r in regions] == ["Top", "Bottom"]


def test_single_title_case_word_is_not_a_heading():
    """Logo/name fragments like 'Trauma' / 'Center' must not become banners."""
    spans = [_span("Trauma", 0.1, 0.1, 0.2, 0.12)]
    layout = analyze_layout(spans, page_count=1)
    assert layout["pages"][0]["regions"][0]["region_type"] == "paragraph"


def test_multiword_title_case_line_is_still_a_heading():
    spans = [
        _span("Patient", 0.1, 0.1, 0.18, 0.12),
        _span("Demographics", 0.19, 0.1, 0.35, 0.12),
    ]
    layout = analyze_layout(spans, page_count=1)
    assert layout["pages"][0]["regions"][0]["region_type"] == "heading"


def test_spans_are_partitioned_by_page():
    spans = [
        _span("Page one text", 0.1, 0.1, 0.3, 0.12, page_number=1),
        _span("Page two text", 0.1, 0.1, 0.3, 0.12, page_number=2),
    ]
    layout = analyze_layout(spans, page_count=2)
    assert layout["pages"][0]["regions"][0]["lines"][0]["text"] == "Page one text"
    assert layout["pages"][1]["regions"][0]["lines"][0]["text"] == "Page two text"
