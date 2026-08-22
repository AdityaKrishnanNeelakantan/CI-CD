"""Validates a rendered synthetic PDF twin against its own ground truth -
the Validate stage of the PDF track.

Deliberately validates structural/reproducibility properties, not a
source comparison: there is no source fingerprint anywhere in this
pipeline to compare against (see src/privacy/profile_sanitizer.py and the
privacy review that shaped it - packaging source fingerprints for later
comparison is exactly the risky pattern that review flagged). The only
meaningful question left to ask here is whether the rendered PDF
faithfully reproduces its own already-synthetic ground truth - a
round-trip extraction (src/documents/pdf_adapter.py) plus a table-cell
width-fit check, the same "W_text <= W_field" no-clipping requirement
this project's own reference design calls for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fpdf import FPDF

from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.pdf_renderer import FONT_FAMILY, TABLE_FONT_SIZE

FIELD_ACCURACY_PASS_THRESHOLD = 0.99
_WIDTH_TOLERANCE = 0.5  # floating-point slack between render-time and validation-time width measurement


def _normalize_for_match(text: str) -> str:
    """Collapse whitespace so line-wrap newlines do not fail round-trip presence checks."""
    return " ".join(str(text).split())


def _measure_string_width(text: str, font_size: float) -> float:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font(FONT_FAMILY, size=font_size)
    return pdf.get_string_width(text)


def validate_rendered_document(pdf_path: str | Path, ground_truth: dict[str, Any]) -> dict[str, Any]:
    extracted = PDFDocumentAdapter().extract(pdf_path)
    extracted_text = extracted["text"]
    extracted_norm = _normalize_for_match(extracted_text)

    field_checks: list[dict[str, Any]] = []
    overflow_checks: list[dict[str, Any]] = []

    for region in ground_truth["regions"]:
        if region["region_type"] in ("field", "heading", "paragraph"):
            # Headings/paragraphs carry synthetic inline-span substitutions (see
            # pdf_renderer._draw_heading/_draw_paragraph) just like fields do, so
            # they need the same round-trip presence check - a silent substitution
            # bug in narrative text would otherwise never be caught here. They are
            # rendered via multi_cell (auto-wrapping, never clipped), so unlike
            # table cells they need no width-overflow check.
            expected_text = region["rendered_text"]
            field_checks.append(
                {
                    "region_id": region["region_id"],
                    "matched": _normalize_for_match(expected_text) in extracted_norm,
                }
            )

        elif region["region_type"] == "table":
            column_widths = region["column_widths"]
            # Wide tables (many columns) get rendered with a shrunk font
            # (see pdf_renderer._fit_table_font_size) so their text still
            # fits within the narrower columns that many columns forces -
            # the overflow check must measure at that same actual font
            # size, not the nominal TABLE_FONT_SIZE, or it would flag
            # correctly-fitting shrunk text as a false overflow.
            font_size = region.get("font_size", TABLE_FONT_SIZE)
            for row in region["rows"]:
                for column_index, cell_text in enumerate(row["rendered_cells"]):
                    if not cell_text:
                        continue
                    field_checks.append(
                        {
                            "region_id": f"{region['region_id']}[{row['row_index']}][{column_index}]",
                            "matched": _normalize_for_match(cell_text) in extracted_norm,
                        }
                    )
                    measured_width = _measure_string_width(cell_text, font_size)
                    overflow_checks.append(
                        {
                            "region_id": region["region_id"],
                            "row_index": row["row_index"],
                            "column_index": column_index,
                            "fits": measured_width <= column_widths[column_index] + _WIDTH_TOLERANCE,
                        }
                    )

    total_fields = len(field_checks)
    matched_fields = sum(1 for f in field_checks if f["matched"])
    field_accuracy = round(matched_fields / total_fields, 6) if total_fields else 1.0
    overflow_failures = [c for c in overflow_checks if not c["fits"]]

    # "empty_extracted_text" only means something went wrong if the
    # ground truth actually expected content - a genuinely empty document
    # (zero regions, e.g. a blank compiled template) legitimately extracts
    # no text and that is not a rendering failure.
    has_any_content = bool(ground_truth["regions"])
    hard_checks_passed = (
        field_accuracy >= FIELD_ACCURACY_PASS_THRESHOLD
        and not overflow_failures
        and not (has_any_content and extracted["extraction_warnings"])
    )

    return {
        "field_accuracy": field_accuracy,
        "total_fields": total_fields,
        "matched_fields": matched_fields,
        "unmatched_fields": [f["region_id"] for f in field_checks if not f["matched"]],
        "overflow_failure_count": len(overflow_failures),
        "overflow_failures": overflow_failures,
        "extraction_warnings": extracted["extraction_warnings"],
        "hard_checks_passed": hard_checks_passed,
    }
