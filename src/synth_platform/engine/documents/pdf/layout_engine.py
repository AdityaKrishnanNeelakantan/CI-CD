"""Groups positioned text spans into lines, then regions (paragraph /
heading / field / table), with a top-to-bottom reading order per page.

Deliberately geometry-and-shape-only: this module decides *where* things
are and how they group, using position and count signals (line count,
cell count, case), never the field semantics (label meaning, value data
type) - that one layer up, in src/documents/template_compiler.py. Same
split as src/documents/preflight.py (signal) vs pdf_classifier.py
(decision) upstream of extraction.

Works identically on native (pypdf visitor) or OCR (pytesseract) spans -
both share the same {text, x0, y0, x1, y1, page_number, confidence} shape,
fractional 0-1 coordinates with a top-left origin.
"""

from __future__ import annotations

import re
from typing import Any

_LINE_Y_TOLERANCE = 0.012
_REGION_GAP_MULTIPLIER = 1.8
_DEFAULT_LINE_HEIGHT = 0.02
_COLUMN_GAP_THRESHOLD = 0.035
_TABLE_MIN_LINES = 3
_TABLE_MIN_MULTI_CELL_RATIO = 0.7
_HEADING_MAX_WORDS = 8
_FIELD_LINE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 /&.'-]{1,60}:\s*.+$")


def analyze_layout(
    spans: list[dict[str, Any]], page_count: int, page_dimensions: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    dimensions_by_page = {entry["page_number"]: entry for entry in (page_dimensions or [])}
    pages_out = []
    for page_number in range(1, page_count + 1):
        page_spans = [s for s in spans if s["page_number"] == page_number]
        lines = _group_into_lines(page_spans)
        lines.sort(key=lambda ln: ln["y0"])
        regions = _explode_field_line_stacks(_group_into_regions(lines))

        region_entries = []
        for index, region_lines in enumerate(regions):
            region_entries.append(
                {
                    "region_id": f"p{page_number}_r{index}",
                    "region_type": _classify_region(region_lines),
                    "bbox": {
                        "x0": min(ln["x0"] for ln in region_lines),
                        "y0": min(ln["y0"] for ln in region_lines),
                        "x1": max(ln["x1"] for ln in region_lines),
                        "y1": max(ln["y1"] for ln in region_lines),
                    },
                    "reading_order_index": index,
                    "lines": [{"text": ln["text"], "cells": ln["cells"]} for ln in region_lines],
                }
            )

        page_entry: dict[str, Any] = {"page_number": page_number, "regions": region_entries}
        dimensions = dimensions_by_page.get(page_number)
        if dimensions is not None:
            page_entry["width_pt"] = dimensions["width_pt"]
            page_entry["height_pt"] = dimensions["height_pt"]
        pages_out.append(page_entry)

    return {"page_count": page_count, "pages": pages_out}


def _group_into_lines(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not spans:
        return []

    ordered = sorted(spans, key=lambda s: s["y0"])
    clusters: list[list[dict[str, Any]]] = []
    for span in ordered:
        placed = False
        for cluster in clusters:
            if abs(cluster[0]["y0"] - span["y0"]) <= _LINE_Y_TOLERANCE:
                cluster.append(span)
                placed = True
                break
        if not placed:
            clusters.append([span])

    lines = []
    for cluster in clusters:
        cluster.sort(key=lambda s: s["x0"])
        lines.append(
            {
                "text": " ".join(s["text"] for s in cluster),
                "cells": _split_into_cells(cluster),
                "x0": min(s["x0"] for s in cluster),
                "y0": min(s["y0"] for s in cluster),
                "x1": max(s["x1"] for s in cluster),
                "y1": max(s["y1"] for s in cluster),
            }
        )
    return lines


def _split_into_cells(line_spans: list[dict[str, Any]]) -> list[str]:
    cells: list[list[str]] = [[line_spans[0]["text"]]]
    for prev, curr in zip(line_spans, line_spans[1:]):
        if curr["x0"] - prev["x1"] > _COLUMN_GAP_THRESHOLD:
            cells.append([curr["text"]])
        else:
            cells[-1].append(curr["text"])
    return [" ".join(cell_tokens) for cell_tokens in cells]


def _group_into_regions(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not lines:
        return []

    heights = sorted(ln["y1"] - ln["y0"] for ln in lines if ln["y1"] > ln["y0"])
    median_height = heights[len(heights) // 2] if heights else _DEFAULT_LINE_HEIGHT

    regions: list[list[dict[str, Any]]] = [[lines[0]]]
    for prev, curr in zip(lines, lines[1:]):
        # Table rows are typically padded more generously than prose lines
        # (border/cell padding), so a plain vertical-gap threshold tuned for
        # paragraph spacing splits every row of a real table into its own
        # single-line region - which then can never reach _looks_like_table()
        # (only considered for regions with 2+ lines already grouped
        # together). Consecutive multi-cell lines with the *same* cell
        # count are the signature of tabular data (an aligned column grid),
        # so keep those grouped regardless of gap size; requiring an exact
        # cell-count match (not just ">=2 cells" each) avoids stitching
        # together unrelated multi-cell prose (e.g. paragraphs that
        # incidentally split into 2 cells on one line and 7 on the next)
        # into one bogus "table" - only fall back to the gap heuristic when
        # cell counts differ or either side is a plain single-cell line.
        if len(prev["cells"]) >= 2 and len(prev["cells"]) == len(curr["cells"]):
            regions[-1].append(curr)
            continue
        gap = curr["y0"] - prev["y1"]
        if gap > median_height * _REGION_GAP_MULTIPLIER:
            regions.append([curr])
        else:
            regions[-1].append(curr)
    return regions


def _is_field_line(line: dict[str, Any]) -> bool:
    if len(line["cells"]) >= 2 and line["cells"][0].rstrip().endswith(":"):
        return True
    return bool(_FIELD_LINE_PATTERN.match(line["text"]))


def _looks_like_table(lines: list[dict[str, Any]]) -> bool:
    if len(lines) < _TABLE_MIN_LINES:
        return False
    multi_cell_lines = sum(1 for ln in lines if len(ln["cells"]) >= 2)
    return multi_cell_lines / len(lines) >= _TABLE_MIN_MULTI_CELL_RATIO


def _explode_field_line_stacks(regions: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """A block of vertically-close "Label: value" lines (a compact form) is
    grouped by _group_into_regions() as one multi-line region purely on
    proximity, before field-shape is ever considered per line - which then
    hides every individual field behind a single "paragraph" classification.
    Split such a block back into one field region per line; leave tables
    and genuine multi-line prose untouched.
    """
    exploded: list[list[dict[str, Any]]] = []
    for region_lines in regions:
        if len(region_lines) > 1 and not _looks_like_table(region_lines) and all(
            _is_field_line(ln) for ln in region_lines
        ):
            exploded.extend([[line] for line in region_lines])
        else:
            exploded.append(region_lines)
    return exploded


def _classify_region(lines: list[dict[str, Any]]) -> str:
    if len(lines) == 1:
        line = lines[0]
        if _is_field_line(line):
            return "field"
        words = line["text"].split()
        word_count = len(words)
        # Single Title-Case tokens ("Trauma", "Center", surname fragments
        # from a split letterhead) are almost never real section headers —
        # requiring ALL-CAPS or 2+ words stops logo/name fragments from
        # becoming blue banner headings in the twin.
        if word_count <= _HEADING_MAX_WORDS and line["text"].isupper() and word_count >= 1:
            return "heading"
        if 2 <= word_count <= _HEADING_MAX_WORDS and line["text"].istitle():
            return "heading"
        return "paragraph"

    if _looks_like_table(lines):
        return "table"
    return "paragraph"
