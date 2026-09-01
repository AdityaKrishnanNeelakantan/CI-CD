"""Positioned text extraction for text-layer PDFs, via pypdf's
visitor_text callback - its render-time hook that exposes the raw PDF
text matrix per text-showing operator, unlike extract_text(), which only
returns a flat string.

Each text-showing operation becomes one positioned "span". Granularity
depends on how the PDF's producer grouped its Tj/TJ operators - anywhere
from one word to a full line - not a promise of word-level detail. This
mirrors src/documents/ocr_engine.py's positioned_spans output (same keys,
same fractional/top-left-origin coordinate space) so
src/documents/layout_engine.py can consume either without caring which
extraction path produced it.

Span widths are estimated from character count and font size (pypdf's
visitor does not expose per-glyph widths); heights come from font size.
Both are rough - enough for line/region clustering, not pixel-exact
layout, which is all layout analysis needs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pypdf

from synth_platform.engine.documents.pdf.errors import ExtractionError

_ASCENDER_DESCENDER_RATIO = 0.2
_AVG_CHAR_WIDTH_RATIO = 0.5
_DEFAULT_FONT_SIZE = 10.0


def _open_reader_for_form_reading(file_path: Path) -> pypdf.PdfReader:
    try:
        reader = pypdf.PdfReader(str(file_path))
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF for form-field extraction: {exc}") from exc
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            pass
        if reader.is_encrypted:
            raise ExtractionError(f"PDF is encrypted and could not be opened: {file_path}")
    return reader


def extract_pdf_info_metadata(path: str | Path) -> list[dict[str, str]]:
    """The PDF "Info dictionary"'s Author/Title/Subject/Keywords fields -
    a real, well-documented metadata-leakage vector distinct from a
    document's own rendered content: Office exporters and many PDF
    producers routinely auto-populate /Author with the OS username or the
    actual author's name, and /Title or /Subject with an internal project
    name, none of which is ever visible when merely reading the page text.

    Only these 4 human-authored fields are read - not /Producer or
    /Creator, which are software product names (e.g. "Microsoft Word"),
    not personal data, and would only add noise to PII/entity scanning.

    Returned as {name, value} pairs (not merged into any page here) so the
    caller decides how to use them; PDFDocumentAdapter merges them into
    page 1's text purely for PII/entity/text_hash detection coverage -
    the rendered twin never reads the source PDF at all (see
    src/documents/pdf_renderer.py), so this can never leak into the twin
    itself, only into the *profiling* stage's own findings.
    """
    reader = _open_reader_for_form_reading(Path(path))
    metadata = reader.metadata
    if metadata is None:
        return []

    entries: list[dict[str, str]] = []
    for label, key in (
        ("Author", "/Author"),
        ("Title", "/Title"),
        ("Subject", "/Subject"),
        ("Keywords", "/Keywords"),
    ):
        try:
            value = metadata.get(key)
        except Exception:
            continue
        if value is None:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        entries.append({"name": label, "value": value_str})
    return entries


def extract_acroform_field_values(path: str | Path) -> list[dict[str, Any]]:
    """Every AcroForm widget annotation with a non-empty value, across all
    pages: {page_number, name, value, rect} (rect is the raw PDF-points
    [x0, y0, x1, y1], bottom-left origin - pypdf's native Rect convention).

    AcroForm field values are NOT part of a page's content stream, so
    page.extract_text() never sees them (verified empirically: a real
    fillable-field PDF with a filled "customer_name" field extracts as
    completely empty text) - this is the dedicated AcroForm extraction
    path the PDF pipeline spec requires, used by preflight (native char
    count), PDFDocumentAdapter (plain-text profile/PII scanning) and
    extract_positioned_spans's caller (template/layout - see
    extract_acroform_field_spans below) so a form's real data is never
    silently invisible to the rest of the pipeline.
    """
    reader = _open_reader_for_form_reading(Path(path))
    fields: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        annots = page.get("/Annots")
        if not annots:
            continue
        for annot_ref in annots:
            try:
                annot = annot_ref.get_object()
            except Exception:
                continue
            if annot.get("/Subtype") != "/Widget":
                continue
            name = annot.get("/T")
            value = annot.get("/V")
            if (name is None or value is None) and annot.get("/Parent") is not None:
                try:
                    parent = annot["/Parent"].get_object()
                except Exception:
                    parent = None
                if parent is not None:
                    name = name if name is not None else parent.get("/T")
                    value = value if value is not None else parent.get("/V")
            if name is None or value is None:
                continue
            value_str = str(value).strip()
            if not value_str:
                continue
            rect = annot.get("/Rect")
            fields.append(
                {
                    "page_number": page_number,
                    "name": str(name),
                    "value": value_str,
                    "rect": [float(c) for c in rect] if rect else None,
                }
            )
    return fields


def _humanize_field_name(name: str) -> str:
    """AcroForm field names are internal identifiers (snake_case,
    camelCase, or dotted "form1.page1.customer_name" XFA-style paths),
    not printed labels - src/documents/layout_engine.py's field-line
    pattern only recognises a natural "Label: value" shape (its allowed
    label characters don't include "_" or "."), so a raw field name like
    "customer_name:" silently fell through to plain "paragraph"
    classification instead of "field" - which then runs the boilerplate
    text redaction sweep meant for narrative wording and masked the
    field's own value in the compiled template. Humanizing the name
    first is what makes it match the same field-recognition path an
    ordinary printed "Account Number: 12345" line already takes.
    """
    leaf = name.split(".")[-1]
    spaced = re.sub(r"[_\-]+", " ", leaf)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    return spaced.title().strip() or leaf


def extract_acroform_field_spans(path: str | Path) -> list[dict[str, Any]]:
    """AcroForm field values as positioned spans, same shape as
    extract_positioned_spans's output - so a form's filled values flow
    through src/documents/layout_engine.py exactly like ordinary text.
    """
    file_path = Path(path)
    reader = _open_reader_for_form_reading(file_path)
    page_dims: dict[int, tuple[float, float]] = {}
    for page_number, page in enumerate(reader.pages, start=1):
        mediabox = page.mediabox
        page_dims[page_number] = (float(mediabox.width) or 1.0, float(mediabox.height) or 1.0)

    spans: list[dict[str, Any]] = []
    for field in extract_acroform_field_values(file_path):
        rect = field["rect"]
        if rect is None:
            continue
        page_width, page_height = page_dims.get(field["page_number"], (1.0, 1.0))
        x0, y0, x1, y1 = rect
        spans.append(
            # "name: value" (not the bare value) so this flows through
            # src/documents/template_compiler.py's existing "Label: value"
            # field-line pattern unchanged, the same path an ordinary
            # printed "Account Number: 12345" line already takes - no
            # separate AcroForm-specific region type needed anywhere else
            # in the layout/template/binding pipeline.
            {
                "text": f"{_humanize_field_name(field['name'])}: {field['value']}",
                "x0": round(max(min(x0, x1) / page_width, 0.0), 6),
                "y0": round(max(1.0 - (max(y0, y1) / page_height), 0.0), 6),
                "x1": round(min(max(x0, x1) / page_width, 1.0), 6),
                "y1": round(min(1.0 - (min(y0, y1) / page_height), 1.0), 6),
                "page_number": field["page_number"],
                "confidence": None,
            }
        )
    return spans


def extract_positioned_spans(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    try:
        reader = pypdf.PdfReader(str(file_path))
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF for layout extraction: {exc}") from exc

    if reader.is_encrypted:
        # Only ever try an empty password, same rule as PDFDocumentAdapter -
        # never attempt to guess a real one.
        try:
            reader.decrypt("")
        except Exception:
            pass
        if reader.is_encrypted:
            raise ExtractionError(f"PDF is encrypted and could not be opened: {file_path}")

    spans: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        mediabox = page.mediabox
        page_width = float(mediabox.width) or 1.0
        page_height = float(mediabox.height) or 1.0
        page_spans: list[dict[str, Any]] = []

        # pypdf's visitor_text hands back (cm, tm) memoized from the start
        # of the text run being flushed - which is usually fine, but is
        # measurably stale for certain back-to-back "q BT <Td> <Tj> ET Q"
        # sequences (one per table cell is a common producer pattern,
        # e.g. reportlab-drawn grids): the memo occasionally still reflects
        # the *pre-Td* (identity) matrix rather than the cell's own
        # position, collapsing that cell's span to (0, 0). visitor_operand_
        # before fires per-operator with pypdf's *live* (not memoized)
        # cm/tm, which does not exhibit this staleness - confirmed against
        # this project's own generated fixtures (a table row's second+
        # column consistently reported tm=(0,0) via visitor_text while
        # visitor_operand_before reported the correct, non-zero position
        # for the exact same Tj). So: track live (cm, tm) ourselves from
        # every text-showing operator, and prefer that over whatever
        # visitor_text supplies, falling back to visitor_text's own values
        # only before any text-showing operator has been observed yet.
        live_state: list[Any] = [None, None]

        def _track_live_position(operator: bytes, operands: list[Any], cm: Any, tm: Any) -> None:
            if operator in (b"Tj", b"TJ", b"'", b'"'):
                live_state[0] = cm
                live_state[1] = tm

        def _visitor(text: str, cm: Any, tm: Any, font_dict: Any, font_size: float) -> None:
            if not text.strip():
                return
            if live_state[0] is not None:
                cm, tm = live_state[0], live_state[1]
            size = font_size or _DEFAULT_FONT_SIZE
            # tm is the text matrix; cm is the current transformation matrix
            # in effect (identity at the page's top level, but non-identity
            # for text drawn inside a nested Form XObject). Compose them to
            # land in absolute page coordinates rather than assuming cm is
            # always identity.
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            width = max(len(text) * size * _AVG_CHAR_WIDTH_RATIO, 1.0)
            y_top = y + size * _ASCENDER_DESCENDER_RATIO
            y_bottom = y - size * _ASCENDER_DESCENDER_RATIO
            page_spans.append(
                {
                    "text": text,
                    "x0": round(max(x / page_width, 0.0), 6),
                    "y0": round(max(1.0 - (y_top / page_height), 0.0), 6),
                    "x1": round(min((x + width) / page_width, 1.0), 6),
                    "y1": round(min(1.0 - (y_bottom / page_height), 1.0), 6),
                    "page_number": page_number,
                    "confidence": None,
                }
            )

        try:
            page.extract_text(visitor_text=_visitor, visitor_operand_before=_track_live_position)
        except Exception:
            pass  # a single unparseable page must not fail the whole document

        spans.extend(page_spans)

    return spans
