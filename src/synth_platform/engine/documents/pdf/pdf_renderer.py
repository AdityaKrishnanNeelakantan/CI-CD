"""Renders a synthetic PDF twin from a compiled DocumentTemplateSpec
(src/documents/template_compiler.py) plus generated synthetic values
(src/documents/value_generator.py) - the Render stage of the PDF track.

Deliberately a flow-based renderer, not a pixel-exact one: the source
geometry this project's layout/native extraction produces is itself only
approximate (see src/documents/native_layout.py's own docstring on
estimated span widths, and OCR's inherent word-box noise), so redrawing
at "the same" bbox would be false precision, not fidelity. Every region
is instead drawn fresh, in the template's own reading order, onto a
brand-new standard page sequence, using fpdf2's own auto-page-break
rather than reimplementing pagination height math - producing a
structurally equivalent twin (same regions, same order, same field
semantics, brand new synthetic content) rather than a pixel-identical one.

Never touches the source PDF or any raw value - only document_template.json
(structure + labels + static text, already redacted) and
document_synthetic_values.json (brand-new generated values, safe to
render and record verbatim since they were never real to begin with).
"""

from __future__ import annotations

from typing import Any

from fpdf import FPDF

FONT_FAMILY = "Helvetica"
_BODY_FONT_SIZE = 11
_HEADING_FONT_SIZE = 14
_PT_TO_MM = 25.4 / 72.0

# fpdf2's core fonts (Helvetica/Times/Courier) only support Latin-1 - common
# "smart" punctuation from real-world Word/PDF exporters (curly quotes,
# em-dashes, bullets) falls outside that range and previously crashed
# rendering outright (verified: 2 of 6 real-world sample PDFs raised
# FPDFUnicodeEncodingException here). Transliterate the common cases, then
# drop anything still unsupported rather than crash - matching this
# pipeline's established never-crash-on-a-real-document policy.
_UNICODE_TO_LATIN1 = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
    "\u2026": "...",
    "\u2022": "-", "\u25cf": "-", "\u25aa": "-",
    "\u00a0": " ",
}


def _sanitize_for_core_font(text: str) -> str:
    for unicode_char, replacement in _UNICODE_TO_LATIN1.items():
        text = text.replace(unicode_char, replacement)
    return text.encode("latin-1", errors="ignore").decode("latin-1")
# US Letter - only used when the template carries no page dimensions at
# all (e.g. a hand-built template in an older artifact or a unit test),
# never as a substitute for a real, known source page size.
_FALLBACK_PAGE_FORMAT = "Letter"


def _page_format_mm(template: dict[str, Any]) -> str | tuple[float, float]:
    pages = template.get("pages") or []
    if not pages or "width_pt" not in pages[0] or "height_pt" not in pages[0]:
        return _FALLBACK_PAGE_FORMAT
    return (pages[0]["width_pt"] * _PT_TO_MM, pages[0]["height_pt"] * _PT_TO_MM)
TABLE_FONT_SIZE = 10
_LINE_HEIGHT = 7.0
_CELL_PADDING = 4.0
_MIN_COLUMN_WIDTH = 20.0
_MIN_TABLE_FONT_SIZE = 6.0
_FONT_SHRINK_STEP = 0.9

# Generic stylistic colors - a deliberate approximation of common real-world
# document styling (colored section-header bars/text, shaded table header
# row), not values sampled from the source PDF's actual pixels or graphics.
# This pipeline never reads source pixel/image data (see module docstring),
# so these are fixed, made-up brand-neutral colors rather than an attempt at
# literal reproduction of the original document's exact palette.
_HEADING_TEXT_COLOR = (0, 90, 156)
_HEADING_FILL_COLOR = (225, 232, 245)
_TABLE_HEADER_FILL_COLOR = (225, 232, 245)


def _apply_inline_spans(text: str, generated_spans: list[dict[str, Any]]) -> str:
    """Splice freshly generated values into a heading/paragraph's already-
    redacted text, at the exact positions src/documents/template_compiler.py
    recorded - so the twin shows a plausible new number/date/etc. instead
    of a literal masked placeholder ("69*96") sitting in otherwise-fresh
    synthetic content.
    """
    result = text
    for span in sorted(generated_spans, key=lambda s: s["start"], reverse=True):
        value = span.get("value", span.get("formatted"))
        replacement = "" if value is None else str(value)
        result = result[: span["start"]] + replacement + result[span["end"] :]
    return result


def _field_display_text(binding: dict[str, Any] | None, generated_field: dict[str, Any]) -> str:
    value = generated_field.get("value", generated_field.get("formatted"))
    value_text = "" if value is None else str(value)
    label = generated_field.get("label")
    return f"{label}: {value_text}" if label else value_text


def _cell_display_text(generated_cell: dict[str, Any]) -> str:
    value = generated_cell.get("value", generated_cell.get("formatted"))
    return "" if value is None else str(value)


def _draw_heading(pdf: FPDF, text: str, ground_truth: list[dict[str, Any]], region_id: str) -> None:
    text = _sanitize_for_core_font(text)
    pdf.set_font(FONT_FAMILY, style="B", size=_HEADING_FONT_SIZE)
    x0, y0 = pdf.get_x(), pdf.get_y()
    pdf.set_fill_color(*_HEADING_FILL_COLOR)
    pdf.set_text_color(*_HEADING_TEXT_COLOR)
    pdf.multi_cell(0, _LINE_HEIGHT, text, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_text_color(0, 0, 0)
    ground_truth.append(
        {
            "region_id": region_id,
            "region_type": "heading",
            "page": pdf.page_no(),
            "bbox": {"x0": x0, "y0": y0, "x1": pdf.w - pdf.r_margin, "y1": pdf.get_y()},
            "rendered_text": text,
        }
    )
    pdf.set_font(FONT_FAMILY, style="", size=_BODY_FONT_SIZE)
    pdf.ln(2)


def _draw_paragraph(pdf: FPDF, text: str, ground_truth: list[dict[str, Any]], region_id: str) -> None:
    text = _sanitize_for_core_font(text)
    pdf.set_font(FONT_FAMILY, style="", size=_BODY_FONT_SIZE)
    x0, y0 = pdf.get_x(), pdf.get_y()
    pdf.multi_cell(0, _LINE_HEIGHT, text, new_x="LMARGIN", new_y="NEXT")
    ground_truth.append(
        {
            "region_id": region_id,
            "region_type": "paragraph",
            "page": pdf.page_no(),
            "bbox": {"x0": x0, "y0": y0, "x1": pdf.w - pdf.r_margin, "y1": pdf.get_y()},
            "rendered_text": text,
        }
    )
    pdf.ln(1)


def _draw_field(
    pdf: FPDF, binding: dict[str, Any], generated_field: dict[str, Any], ground_truth: list[dict[str, Any]]
) -> None:
    text = _sanitize_for_core_font(_field_display_text(binding, generated_field))
    pdf.set_font(FONT_FAMILY, style="", size=_BODY_FONT_SIZE)
    x0, y0 = pdf.get_x(), pdf.get_y()
    pdf.multi_cell(0, _LINE_HEIGHT, text, new_x="LMARGIN", new_y="NEXT")
    ground_truth.append(
        {
            "region_id": binding["region_id"],
            "region_type": "field",
            "page": pdf.page_no(),
            "bbox": {"x0": x0, "y0": y0, "x1": pdf.w - pdf.r_margin, "y1": pdf.get_y()},
            "label": generated_field.get("label"),
            "semantic_role": generated_field.get("semantic_role"),
            "rendered_text": text,
            "rendered_value": generated_field.get("value", generated_field.get("formatted")),
        }
    )
    pdf.ln(1)


def _compute_column_widths(
    pdf: FPDF, rows: list[list[str]], column_count: int, available_width: float, min_column_width: float = _MIN_COLUMN_WIDTH
) -> list[float]:
    widths = [min_column_width] * column_count
    for row in rows:
        for i, cell_text in enumerate(row):
            if i >= column_count:
                continue
            needed = pdf.get_string_width(cell_text) + _CELL_PADDING
            widths[i] = max(widths[i], needed)

    total = sum(widths)
    if total > available_width and total > 0:
        scale = available_width / total
        widths = [max(w * scale, min_column_width / 2) for w in widths]
    return widths


def _fit_table_font_size(
    pdf: FPDF, rows: list[list[str]], column_count: int, available_width: float
) -> tuple[float, list[float]]:
    """Picks the largest font size (down to _MIN_TABLE_FONT_SIZE) at which every
    column's natural (padded) width still fits available_width, so cell text
    never has to be drawn wider than the column fpdf2 actually reserves for it.

    A plain single-size _compute_column_widths pass with a fixed column-width
    floor breaks down for wide tables (many columns, e.g. a lab coverage grid):
    shrinking widths alone can't help once column_count * _MIN_COLUMN_WIDTH
    already exceeds the page, and shrinking widths without shrinking the font
    just makes the already-measured-at-full-size text overflow its new,
    narrower cell. Reducing the min-width floor in proportion to column_count
    plus iteratively shrinking the font (re-measuring at each candidate size)
    keeps the two in sync so the final widths always accommodate the text
    actually drawn at the chosen font size.
    """
    min_column_width = min(_MIN_COLUMN_WIDTH, available_width / max(column_count, 1))
    font_size = TABLE_FONT_SIZE
    widths = [min_column_width] * column_count
    while True:
        pdf.set_font(FONT_FAMILY, style="", size=font_size)
        widths = [min_column_width] * column_count
        for row in rows:
            for i, cell_text in enumerate(row):
                if i >= column_count:
                    continue
                needed = pdf.get_string_width(cell_text) + _CELL_PADDING
                widths[i] = max(widths[i], needed)
        total = sum(widths)
        if total <= available_width or font_size <= _MIN_TABLE_FONT_SIZE:
            break
        font_size = max(font_size * _FONT_SHRINK_STEP, _MIN_TABLE_FONT_SIZE)

    total = sum(widths)
    if total > available_width and total > 0:
        scale = available_width / total
        widths = [max(w * scale, min_column_width / 4) for w in widths]
    return font_size, widths


def _draw_table(
    pdf: FPDF, region_id: str, column_count: int, generated_rows: list[list[dict[str, Any]]], ground_truth: list[dict[str, Any]]
) -> None:
    pdf.set_font(FONT_FAMILY, style="", size=TABLE_FONT_SIZE)
    display_rows = [
        [_sanitize_for_core_font(_cell_display_text(cell)) for cell in row] for row in generated_rows
    ]
    font_size, column_widths = _fit_table_font_size(pdf, display_rows, column_count, pdf.epw)
    pdf.set_font(FONT_FAMILY, style="", size=font_size)

    row_ground_truth = []
    pdf.set_fill_color(*_TABLE_HEADER_FILL_COLOR)
    for row_index, row_texts in enumerate(display_rows):
        row_x0, row_y0 = pdf.get_x(), pdf.get_y()
        is_header_row = row_index == 0
        if is_header_row:
            pdf.set_font(FONT_FAMILY, style="B", size=font_size)
        for col_index in range(column_count):
            text = row_texts[col_index] if col_index < len(row_texts) else ""
            pdf.cell(
                column_widths[col_index], _LINE_HEIGHT, text, border=1, new_x="RIGHT", new_y="TOP", fill=is_header_row
            )
        if is_header_row:
            pdf.set_font(FONT_FAMILY, style="", size=font_size)
        pdf.ln(_LINE_HEIGHT)
        row_ground_truth.append(
            {
                "row_index": row_index,
                "page": pdf.page_no(),
                "bbox": {"x0": row_x0, "y0": row_y0, "x1": row_x0 + sum(column_widths), "y1": pdf.get_y()},
                "rendered_cells": row_texts,
            }
        )

    ground_truth.append(
        {
            "region_id": region_id,
            "region_type": "table",
            "column_count": column_count,
            "column_widths": column_widths,
            "font_size": font_size,
            "rows": row_ground_truth,
        }
    )
    pdf.ln(2)


def render_document_pdf(
    template: dict[str, Any],
    binding_map: dict[str, Any],
    synthetic_values: dict[str, Any],
    output_path: str,
) -> dict[str, Any]:
    """Render a synthetic PDF twin, writing it to output_path.

    Returns the ground-truth record: exactly what was drawn, where, and
    on which page - the authoritative expected output for
    src/documents/render_validator.py's round-trip check.
    """
    bindings_by_region = {b["region_id"]: b for b in binding_map["bindings"]}

    pdf = FPDF(format=_page_format_mm(template))
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=_BODY_FONT_SIZE)

    ground_truth: list[dict[str, Any]] = []

    regions = sorted(
        (
            (page["page_number"], region["reading_order_index"], region)
            for page in template["pages"]
            for region in page["regions"]
        ),
        key=lambda entry: (entry[0], entry[1]),
    )

    for _page_number, _order, region in regions:
        region_id = region["region_id"]
        region_type = region["region_type"]

        if region_type == "heading":
            text = _apply_inline_spans(region.get("text", ""), synthetic_values["inline_spans"].get(region_id, []))
            _draw_heading(pdf, text, ground_truth, region_id)
        elif region_type == "paragraph":
            text = _apply_inline_spans(region.get("text", ""), synthetic_values["inline_spans"].get(region_id, []))
            _draw_paragraph(pdf, text, ground_truth, region_id)
        elif region_type == "field":
            binding = bindings_by_region.get(region_id)
            generated_field = synthetic_values["fields"].get(region_id, {})
            _draw_field(pdf, binding or {"region_id": region_id}, generated_field, ground_truth)
        elif region_type == "table":
            generated_rows = synthetic_values["tables"].get(region_id, [])
            column_count = region.get("column_count", max((len(r) for r in generated_rows), default=0))
            _draw_table(pdf, region_id, column_count, generated_rows, ground_truth)

    pdf.output(output_path)

    return {"page_count": pdf.page_no(), "regions": ground_truth}
