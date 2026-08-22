"""Mode 2b orchestration: visual/pixel redaction -
runs/<run_id>/documents/<doc_id>/redacted_image.pdf and
document_image_redaction_report.json.

Distinct from run_document_deidentification (deidentification_service.py,
"redacted.pdf"): that stage reconstructs a fresh, still-selectable-text
PDF from document_template.json's already region-split masked_preview
values. A single logical value can be fragmented across multiple
template regions when the source PDF happens to wrap it onto several
lines (e.g. a name printed on its own line right after a "Doctor Name:"
label) - layout_engine.py has no way to know two adjacent one-word
regions are really one value, and each fragment is then too short on its
own for a PERSON/DATE pattern to fire, so it survives unredacted. This
stage sidesteps that entirely: it never reads document_template.json at
all. It re-extracts the source PDF's positioned spans fresh, reconstructs
each PAGE's full text (not just one line/region at a time - the same
document-wide scope document_profile.json's own findings already use),
runs the same PII/entity detection once against that, then maps every
finding's character range back onto the specific source span(s) it
covers - so a value split across lines is still caught as a single
match - and finally rasterizes the source page (via pdf2image/poppler,
already an optional dependency for OCR) and paints a solid black
rectangle over each match's true source position. The output is a
pixel-only PDF: no text survives anywhere, redacted or not, so there is
no way for a black box to be sitting on top of still-selectable text
underneath it (the classic "redaction" flaw). This matches the visual
style of the reference fixtures in examples/fixtures/pdfs/ (Original.pdf ->
Synthetic Twin.pdf), which are themselves fully rasterized per-page
images with black boxes drawn over the source layout.

Reuses template_compiler._redact_inline_sensitive_values as the
canonical "what counts as sensitive" decision for prose text, so this
stage's redaction policy never drifts from Mode 2's. Bare numeric dates
(e.g. "24/05/1977", "02/08/2024") are not covered by that function
though - entities.py's DATE patterns deliberately only recognise ISO and
month-name dates, never slash-numeric ones (see entities.py's
_DATE_PATTERNS) - so this module adds its own supplementary inline
pattern for those, mirroring template_compiler._DATE_PATTERN's shape but
applied to substrings of prose/table text rather than a whole field
value.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.documents.pdf.errors import DocumentAdapterError, OCRUnavailableError
from synth_platform.engine.documents.pdf.layout_backends.registry import resolve_layout_backend
from synth_platform.engine.documents.pdf.native_layout import extract_acroform_field_spans
from synth_platform.engine.documents.pdf.preflight import run_pdf_preflight
from synth_platform.engine.documents.pdf.template_compiler import _redact_inline_sensitive_values

STAGE_NAME = "image_redaction"
IMAGE_REDACTED_PDF_FILENAME = "redacted_image.pdf"
IMAGE_REDACTION_REPORT_FILENAME = "document_image_redaction_report.json"

_REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "redacted_at", "page_count", "redaction_box_count"}

# Same y0-clustering tolerance as layout_engine._group_into_lines - spans
# whose top y-coordinate differ by less than this are considered part of
# the same visual line.
_LINE_Y_TOLERANCE = 0.012

# entities.py's DATE patterns deliberately never match slash-numeric dates
# (see its module docstring) - this fills that specific gap for this
# module only, matching template_compiler._DATE_PATTERN's shape but
# unanchored so it can match inside a larger prose/table string.
_INLINE_DATE_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")

# A bare age number has no distinctive shape of its own (detect_pii/
# extract_entities have nothing to match), but "46-year-old"/"46 years
# old" phrasing is an unambiguous, safe-to-generalise structural signal -
# age is one of HIPAA Safe Harbor's enumerated identifiers, same tier as
# name/DOB/SSN. Only the number itself is captured (a lookahead, not a
# consuming match), so "-year-old"/" years old" stays legible.
_INLINE_AGE_PATTERN = re.compile(r"\b\d{1,3}(?=-year-old\b|\syears?\sold\b)")

# Field labels whose value is directly, unconditionally redacted in place
# (no pattern match required) wherever that exact "Label: value" line
# appears - mirrors HIPAA Safe Harbor's age identifier, which (unlike
# name/date) has no distinctive shape detect_pii/extract_entities/the
# date pattern above could ever catch on its own.
_DIRECT_REDACT_FIELD_LABELS = {"age"}

# entities.py's PERSON pattern only reaches its >=0.8 "strong entity" bar
# (the threshold _redact_inline_sensitive_values requires) with extra
# context like a title prefix - a bare "Kimberly Lawrence" is a
# 0.6-confidence "capitalised_sequence" match, deliberately excluded
# there because that same weak signal fires on ordinary section headings
# ("Patient Summary", "Doctor Information") too, and blindly lowering the
# bar would black those out as well. A field explicitly labelled
# "Name"/"Doctor Name" is a much stronger, structural signal that its
# value is a real person's name - so this module identifies the name from
# there once, then sweeps for that exact string everywhere else it
# recurs (running headers, narrative summaries), the same way a human
# redactor would black out a name they already know is sensitive.
_FIELD_LABEL_LINE_PATTERN = re.compile(r"^(?P<label>[A-Za-z][A-Za-z /]{1,40}):\s*(?P<value>.*)$")
# A source PDF can wrap a field's value onto its own following line(s)
# (e.g. "Name: Kimberly" then "Lawrence" on the next line) - each
# constituent line still looks like a plausible name fragment (Title-Case
# word(s), no colon), so they're absorbed into the field's full value
# rather than left as a separate, too-short-to-match fragment.
_CONTINUATION_WORD_PATTERN = re.compile(r"^[A-Z][a-zA-Z'\-]*(?:\s+[A-Z][a-zA-Z'\-]*){0,2}$")
_MAX_CONTINUATION_LINES = 2

_RASTER_DPI = 200

# native_layout.py's extract_spans() deliberately gives each span a tight
# vertical bbox centred on the text baseline (+/- 0.2 * font_size, see its
# _ASCENDER_DESCENDER_RATIO) - good enough for line-clustering/layout
# analysis, but far too short to fully occlude a glyph's actual ascenders/
# descenders when drawn as a solid box (it would render as a thin
# underline with the text still legible above it). Padding the box
# vertically by this ratio on each side approximates a real font's full
# cap-height-to-descender extent for redaction-drawing purposes only -
# this constant is local to this module and never changes the underlying
# span data other stages rely on.
_VERTICAL_PAD_RATIO = 0.9


class DocumentImageRedactionReportLoadError(Exception):
    """Raised when document_image_redaction_report.json is missing, corrupted, or malformed."""


def _group_spans_into_lines(spans: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(spans, key=lambda s: s["y0"])
    clusters: list[list[dict[str, Any]]] = []
    for span in ordered:
        for cluster in clusters:
            if abs(cluster[0]["y0"] - span["y0"]) <= _LINE_Y_TOLERANCE:
                cluster.append(span)
                break
        else:
            clusters.append([span])
    for cluster in clusters:
        cluster.sort(key=lambda s: s["x0"])
    clusters.sort(key=lambda cluster: cluster[0]["y0"])

    # A y0-tolerance cluster can still contain two unrelated columns that
    # merely happen to sit at the same height (e.g. a logo's wrapped text
    # in the left margin next to a running "Name / DOB" header in the top
    # right) - joining those into one "line" risks a PERSON/ORG/date
    # pattern spuriously bridging two genuinely unrelated pieces of text
    # (see the module docstring). Genuine same-line word/field spacing is
    # always far smaller than the gap between two distinct columns, so
    # any two horizontally-adjacent spans separated by more than this
    # fraction of the page width are split into separate lines instead.
    _COLUMN_GAP_THRESHOLD = 0.05
    split_clusters: list[list[dict[str, Any]]] = []
    for cluster in clusters:
        current = [cluster[0]]
        for span in cluster[1:]:
            if span["x0"] - current[-1]["x1"] > _COLUMN_GAP_THRESHOLD:
                split_clusters.append(current)
                current = [span]
            else:
                current.append(span)
        split_clusters.append(current)
    return split_clusters


def _line_text_and_span_offsets(
    line_spans: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, int, dict[str, Any]]]]:
    """Join a line's spans with a single space (same convention
    layout_engine._group_into_lines and template_compiler's own
    heading/paragraph joins already use), tracking each span's character
    range within the joined text so a later character-offset match can be
    traced back to the exact originating span.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    for index, span in enumerate(line_spans):
        if index > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(span["text"])
        cursor += len(span["text"])
        offsets.append((start, cursor, span))
    return "".join(parts), offsets


def _sub_bbox(span: dict[str, Any], local_start: int, local_end: int) -> dict[str, float]:
    """Approximate the pixel region a [local_start, local_end) character
    range occupies within one span, by weighted-character-width
    interpolation across the span's own bbox width - not a
    font-metrics-exact position, but good enough to separate a "Label: "
    prefix from the value that follows it on the same line without
    needing word-level extraction spans.

    A plain equal-width-per-character estimate breaks down badly on a
    run of alignment spaces (e.g. a "Name" field's fixed-width label
    column padded with several spaces before its value) - a space glyph
    is visually much narrower than a letter/digit, so counting it as a
    full character systematically pushes the value's estimated start too
    far right, leaving its first character(s) exposed. Weighting a space
    as a fraction of a normal character's width corrects most of that.
    Real fonts are still not monospace even for non-space characters, so
    a generous pad (favouring over-redaction over under-redaction, since
    leaving a PII fragment legible is the far worse failure mode here) is
    still added on each side, clamped to the span's own bounds.
    """
    text = span["text"]
    length = len(text)
    if length == 0:
        return {"x0": span["x0"], "y0": span["y0"], "x1": span["x1"], "y1": span["y1"]}
    width = span["x1"] - span["x0"]
    weights = [0.3 if ch == " " else 1.0 for ch in text]
    total_weight = sum(weights) or float(length)
    cumulative = [0.0]
    for w in weights:
        cumulative.append(cumulative[-1] + w)
    frac_start = cumulative[max(local_start, 0)] / total_weight
    frac_end = cumulative[min(local_end, length)] / total_weight
    avg_char_width = width / length
    pad = avg_char_width * 3.0
    raw_x0 = span["x0"] + frac_start * width - pad
    raw_x1 = span["x0"] + frac_end * width + pad
    return {
        "x0": max(span["x0"], raw_x0),
        "y0": span["y0"],
        "x1": min(span["x1"], raw_x1),
        "y1": span["y1"],
    }


def _sensitive_spans_in_text(text: str) -> list[tuple[int, int]]:
    _, variable_spans = _redact_inline_sensitive_values(text)
    ranges = [(f["start"], f["end"]) for f in variable_spans]
    occupied = list(ranges)

    def _add(start: int, end: int) -> None:
        if any(start < o_end and end > o_start for o_start, o_end in occupied):
            return
        occupied.append((start, end))
        ranges.append((start, end))

    for match in _INLINE_DATE_PATTERN.finditer(text):
        _add(*match.span())
    for match in _INLINE_AGE_PATTERN.finditer(text):
        _add(*match.span())
    return ranges


def _known_name_ranges_in_text(text: str, known_names: set[str]) -> list[tuple[int, int]]:
    """Sweep for a literal known-name string (see
    _extract_known_person_names) using its own space-joined text, kept
    entirely separate from _sensitive_spans_in_text's newline-joined text:
    an exact-string search carries none of the cross-line false-merge risk
    a generic PERSON/ORG pattern has (see the module docstring's
    "Medical Tests"/ORG example), so it can safely bridge line breaks -
    which is required to catch the name in its own origin field when the
    source PDF wraps it onto two lines (e.g. "Name: Kimberly" / 
    "Lawrence"), not just later recurrences elsewhere in the document.
    """
    ranges: list[tuple[int, int]] = []
    for name in known_names:
        for match in re.finditer(rf"\b{re.escape(name)}\b", text):
            ranges.append(match.span())
    return ranges


def _direct_field_value_ranges(
    line_texts: list[str], line_page_ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Some identifying fields (Age - see _DIRECT_REDACT_FIELD_LABELS)
    have no distinctive shape any pattern above could ever catch, so
    their value is redacted unconditionally wherever that exact labelled
    line appears, purely from the field's structural "Label: value"
    shape - independent of _redact_inline_sensitive_values/known-name
    matching.
    """
    ranges: list[tuple[int, int]] = []
    for index, line_text in enumerate(line_texts):
        match = _FIELD_LABEL_LINE_PATTERN.match(line_text.strip())
        if not match or match.group("label").strip().lower() not in _DIRECT_REDACT_FIELD_LABELS:
            continue
        value = match.group("value").strip()
        if not value:
            continue
        # Re-anchor the value's offset by searching the (unstripped) line
        # text directly, rather than trusting the stripped match's
        # indices, so a line with any leading whitespace never produces
        # an off-by-N box.
        value_offset = line_text.find(value)
        if value_offset == -1:
            continue
        line_start = line_page_ranges[index][0]
        ranges.append((line_start + value_offset, line_start + value_offset + len(value)))
    return ranges


def _extract_known_person_names(pages_line_texts: list[list[str]]) -> set[str]:
    known: set[str] = set()
    for line_texts in pages_line_texts:
        index = 0
        while index < len(line_texts):
            match = _FIELD_LABEL_LINE_PATTERN.match(line_texts[index].strip())
            if match and "name" in match.group("label").lower():
                value_parts = [match.group("value").strip()]
                lookahead = index + 1
                consumed = 0
                while (
                    lookahead < len(line_texts)
                    and consumed < _MAX_CONTINUATION_LINES
                    and ":" not in line_texts[lookahead]
                    and _CONTINUATION_WORD_PATTERN.match(line_texts[lookahead].strip())
                ):
                    value_parts.append(line_texts[lookahead].strip())
                    lookahead += 1
                    consumed += 1
                full_value = " ".join(part for part in value_parts if part).strip()
                if full_value and _CONTINUATION_WORD_PATTERN.match(full_value):
                    known.add(full_value)
            index += 1
    return known


def _collect_redaction_boxes(
    spans: list[dict[str, Any]], page_count: int
) -> dict[int, list[dict[str, float]]]:
    boxes_by_page: dict[int, list[dict[str, float]]] = {p: [] for p in range(1, page_count + 1)}

    pages_line_texts: dict[int, list[str]] = {}
    pages_line_span_offsets: dict[int, list[list[tuple[int, int, dict[str, Any]]]]] = {}
    for page_number in range(1, page_count + 1):
        page_spans = [s for s in spans if s["page_number"] == page_number]
        lines = _group_spans_into_lines(page_spans)
        line_texts: list[str] = []
        line_span_offsets: list[list[tuple[int, int, dict[str, Any]]]] = []
        for line_spans in lines:
            text, offsets = _line_text_and_span_offsets(line_spans)
            line_texts.append(text)
            line_span_offsets.append(offsets)
        pages_line_texts[page_number] = line_texts
        pages_line_span_offsets[page_number] = line_span_offsets

    # A document-wide pass: a field's own value (e.g. the "Name"/"Doctor
    # Name" field on page 1) is the authoritative source for a person's
    # real name, and the same name commonly recurs on every page (running
    # headers) or in narrative text - so this is gathered once across all
    # pages before any redaction boxes are collected.
    known_names = _extract_known_person_names(list(pages_line_texts.values()))

    for page_number in range(1, page_count + 1):
        line_texts = pages_line_texts[page_number]
        line_span_offsets = pages_line_span_offsets[page_number]
        if not line_texts:
            continue

        # Lines are joined with a newline, matching entities.py's own
        # deliberate design: its word-separator patterns use [ \t]+, never
        # \s+, specifically so a PERSON/ORG match never bridges two
        # genuinely separate lines (e.g. a section heading immediately
        # followed by an unrelated institution name must NOT merge into
        # one ORG match). A value that got wrapped onto its own line by
        # the source PDF (e.g. a name immediately following a "Label: "
        # line) is instead handled by the dedicated known-name
        # continuation sweep above, which targets only genuine
        # label-adjacent continuations rather than blindly bridging every
        # line.
        def _join_lines(separator: str) -> tuple[str, list[tuple[int, int]]]:
            parts: list[str] = []
            ranges: list[tuple[int, int]] = []
            cursor = 0
            for index, text in enumerate(line_texts):
                if index > 0:
                    parts.append(separator)
                    cursor += len(separator)
                start = cursor
                parts.append(text)
                cursor += len(text)
                ranges.append((start, cursor))
            return "".join(parts), ranges

        page_text, line_page_ranges = _join_lines("\n")
        space_text, space_line_page_ranges = _join_lines(" ")

        all_ranges = [(*r, line_page_ranges) for r in _sensitive_spans_in_text(page_text)]
        all_ranges += [(*r, line_page_ranges) for r in _direct_field_value_ranges(line_texts, line_page_ranges)]
        all_ranges += [
            (*r, space_line_page_ranges) for r in _known_name_ranges_in_text(space_text, known_names)
        ]

        for f_start, f_end, ranges_for_match in all_ranges:
            for line_index, (line_start, line_end) in enumerate(ranges_for_match):
                overlap_start = max(f_start, line_start)
                overlap_end = min(f_end, line_end)
                if overlap_start >= overlap_end:
                    continue
                local_start = overlap_start - line_start
                local_end = overlap_end - line_start
                for span_start, span_end, span in line_span_offsets[line_index]:
                    span_overlap_start = max(local_start, span_start)
                    span_overlap_end = min(local_end, span_end)
                    if span_overlap_start >= span_overlap_end:
                        continue
                    boxes_by_page[page_number].append(
                        _sub_bbox(span, span_overlap_start - span_start, span_overlap_end - span_start)
                    )

    return boxes_by_page


def _rasterize_and_redact(
    source_path: Path, boxes_by_page: dict[int, list[dict[str, float]]], output_path: Path
) -> int:
    from pdf2image import convert_from_path
    from PIL import ImageDraw

    images = convert_from_path(str(source_path), dpi=_RASTER_DPI)
    if not images:
        raise DocumentAdapterError(f"pdf2image produced no pages for {source_path}")

    total_boxes = 0
    for page_number, image in enumerate(images, start=1):
        image = image.convert("RGB")
        images[page_number - 1] = image
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for box in boxes_by_page.get(page_number, []):
            box_height = box["y1"] - box["y0"]
            pad = box_height * _VERTICAL_PAD_RATIO
            rect = (
                box["x0"] * width,
                (box["y0"] - pad) * height,
                box["x1"] * width,
                (box["y1"] + pad) * height,
            )
            draw.rectangle(rect, fill="black")
            total_boxes += 1

    images[0].save(str(output_path), save_all=True, append_images=images[1:])
    return total_boxes


def run_image_redaction(
    file_path: str | Path,
    doc_id: str,
    manifest: RunManifest,
    extraction_method: str,
) -> StageResult:
    output_path = manifest.output_path(f"documents/{doc_id}/{IMAGE_REDACTED_PDF_FILENAME}")
    report_output_path = manifest.output_path(f"documents/{doc_id}/{IMAGE_REDACTION_REPORT_FILENAME}")
    if output_path.exists() or report_output_path.exists():
        raise RuntimeError(
            f"image-redacted output already exists for document {doc_id!r} in run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    source_path = Path(file_path)

    if shutil.which("pdftoppm") is None:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[str(source_path)],
            output_references=[],
            errors=[
                "poppler (pdftoppm) is not available on PATH; cannot rasterize the source PDF "
                "for visual redaction."
            ],
        )
        manifest.record_stage(result)
        return result

    try:
        spans = resolve_layout_backend(extraction_method).extract_spans(source_path)
        preflight = run_pdf_preflight(source_path)
        if preflight.get("has_form_fields"):
            spans = spans + extract_acroform_field_spans(source_path)
    except (DocumentAdapterError, OCRUnavailableError) as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[str(source_path)],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    page_count = preflight["page_count"]
    boxes_by_page = _collect_redaction_boxes(spans, page_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_boxes = _rasterize_and_redact(source_path, boxes_by_page, output_path)

    report = {
        "doc_id": doc_id,
        "source_reference": str(source_path),
        "redacted_at": datetime.now(UTC).isoformat(),
        "page_count": page_count,
        "redaction_box_count": total_boxes,
        "boxes_by_page": {str(page): len(page_boxes) for page, page_boxes in boxes_by_page.items()},
    }
    with report_output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[str(source_path)],
        output_references=[str(output_path), str(report_output_path)],
        metrics={"page_count": page_count, "redaction_box_count": total_boxes},
        evidence={"redacted_at": report["redacted_at"]},
    )
    manifest.record_stage(result)
    return result


def load_document_image_redaction_report(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a document_image_redaction_report.json file."""
    report_path = Path(path)
    if not report_path.is_file():
        raise DocumentImageRedactionReportLoadError(
            f"document_image_redaction_report.json not found: {report_path}"
        )

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DocumentImageRedactionReportLoadError(
            f"document_image_redaction_report.json is not valid JSON: {report_path}"
        ) from exc

    if not isinstance(data, dict):
        raise DocumentImageRedactionReportLoadError(
            f"document_image_redaction_report.json must be a JSON object: {report_path}"
        )

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DocumentImageRedactionReportLoadError(
            f"document_image_redaction_report.json missing required keys {sorted(missing)}: {report_path}"
        )

    return data
