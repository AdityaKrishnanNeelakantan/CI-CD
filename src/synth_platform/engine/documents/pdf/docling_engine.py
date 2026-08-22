"""Docling-backed PDF extraction.

This module adapts Docling's ``DoclingDocument`` into the platform's two
existing document contracts:

1. page-scoped text used by document profiling; and
2. positioned text spans used by template compilation/layout analysis.

Docling is imported lazily so the rest of the PDF engine can still be
imported in environments where the optional PDF dependency set has not
been installed yet.
"""

from __future__ import annotations

import importlib.util
import os
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from synth_platform.engine.documents.pdf.errors import (
    DoclingUnavailableError,
    ExtractionError,
)
from synth_platform.engine.documents.pdf.native_layout import (
    extract_acroform_field_values,
    extract_pdf_info_metadata,
)


def is_docling_available() -> bool:
    """Return whether the local Python environment can import Docling."""
    return importlib.util.find_spec("docling") is not None


def docling_version() -> str | None:
    """Return the installed Docling version when available."""
    if not is_docling_available():
        return None
    try:
        return metadata.version("docling")
    except metadata.PackageNotFoundError:
        return "unknown"


def extract_with_docling(file_path: str | Path) -> dict[str, Any]:
    """Convert one PDF with Docling and return text plus normalized spans.

    The returned ``positioned_spans`` use the platform layout contract:
    fractional coordinates in ``[0, 1]`` with a top-left origin.
    """
    path = Path(file_path)
    if not is_docling_available():
        raise DoclingUnavailableError(
            "Docling extraction was requested but the 'docling' package is not installed. "
            "Install the Docling dependencies with: pip install -e '.[docling]'"
        )

    try:
        converter = _build_docling_converter()
        result = converter.convert(str(path))
        document = result.document
    except Exception as exc:
        raise ExtractionError(f"Docling failed to convert PDF {path}: {exc}") from exc

    page_texts = _extract_page_texts(document)
    page_texts = _augment_pdf_sensitive_sources(path, page_texts)
    spans = _extract_positioned_spans(document)

    return {
        "pages_text": page_texts,
        "positioned_spans": spans,
        "page_count": max(len(page_texts), _document_page_count(document)),
        "version": docling_version(),
    }



def _build_docling_converter() -> Any:
    """Create a Docling converter, honoring a pre-fetched artifact directory.

    Docling's standard PDF pipeline uses model artifacts for layout/table
    understanding. When ``DOCLING_SERVE_ARTIFACTS_PATH`` is set, pass that
    directory explicitly through ``PdfPipelineOptions`` so air-gapped or
    pre-warmed company deployments do not need a model download at runtime.
    """
    from docling.document_converter import DocumentConverter

    artifacts_path = os.getenv("DOCLING_SERVE_ARTIFACTS_PATH")
    if not artifacts_path:
        return DocumentConverter()

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption

    pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

def _extract_page_texts(document: Any) -> list[str]:
    page_count = _document_page_count(document)
    chunks: dict[int, list[str]] = {page_no: [] for page_no in range(1, page_count + 1)}

    for item, _level in document.iterate_items():
        text = _item_text(item)
        if not text:
            continue
        page_no = _first_page_number(item)
        if page_no is None:
            continue
        chunks.setdefault(page_no, []).append(text)

    if not any(chunks.values()):
        # Provenance can be unavailable for some documents/backends. Keep a
        # usable profiling text contract rather than returning an empty PDF.
        try:
            full_text = (document.export_to_text() or "").strip()
        except Exception:
            full_text = ""
        if page_count == 0:
            page_count = 1 if full_text else 0
        chunks = {page_no: [] for page_no in range(1, page_count + 1)}
        if full_text and chunks:
            chunks[1].append(full_text)

    if page_count == 0 and chunks:
        page_count = max(chunks)

    return ["\n".join(chunks.get(page_no, [])).strip() for page_no in range(1, page_count + 1)]


def _item_text(item: Any) -> str:
    text = getattr(item, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    data = getattr(item, "data", None)
    cells = getattr(data, "table_cells", None)
    if cells:
        rows: dict[int, list[tuple[int, str]]] = {}
        for cell in cells:
            cell_text = str(getattr(cell, "text", "") or "").strip()
            if not cell_text:
                continue
            row = int(getattr(cell, "start_row_offset_idx", 0) or 0)
            col = int(getattr(cell, "start_col_offset_idx", 0) or 0)
            rows.setdefault(row, []).append((col, cell_text))
        lines = ["\t".join(text for _col, text in sorted(row_cells)) for _row, row_cells in sorted(rows.items())]
        return "\n".join(line for line in lines if line).strip()

    return ""


def _first_page_number(item: Any) -> int | None:
    provenance = getattr(item, "prov", None) or []
    for prov in provenance:
        page_no = getattr(prov, "page_no", None)
        if page_no is not None:
            try:
                return int(page_no)
            except (TypeError, ValueError):
                continue
    return None


def _document_page_count(document: Any) -> int:
    pages = getattr(document, "pages", None)
    if isinstance(pages, dict):
        return len(pages)
    if isinstance(pages, (list, tuple)):
        return len(pages)

    max_page = 0
    try:
        for item, _level in document.iterate_items():
            page_no = _first_page_number(item)
            if page_no is not None:
                max_page = max(max_page, page_no)
    except Exception:
        pass
    return max_page


def _page_size(document: Any, page_no: int) -> tuple[float, float] | None:
    pages = getattr(document, "pages", None)
    page = None
    if isinstance(pages, dict):
        page = pages.get(page_no) or pages.get(str(page_no))
    elif isinstance(pages, (list, tuple)) and 0 < page_no <= len(pages):
        page = pages[page_no - 1]
    if page is None:
        return None

    size = getattr(page, "size", None)
    width = getattr(size, "width", None)
    height = getattr(size, "height", None)
    try:
        width_f = float(width)
        height_f = float(height)
    except (TypeError, ValueError):
        return None
    if width_f <= 0 or height_f <= 0:
        return None
    return width_f, height_f


def _extract_positioned_spans(document: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []

    for item, _level in document.iterate_items():
        text = getattr(item, "text", None)
        if isinstance(text, str) and text.strip():
            for prov in getattr(item, "prov", None) or []:
                span = _span_from_bbox(
                    document=document,
                    text=text.strip(),
                    page_no=getattr(prov, "page_no", None),
                    bbox=getattr(prov, "bbox", None),
                )
                if span is not None:
                    spans.append(span)

        # Docling represents table cell geometry on the table data rather
        # than as independent text node provenance. Preserve those cells as
        # individual spans so the platform's existing line/cell grouping can
        # still recognize tables.
        data = getattr(item, "data", None)
        cells: Iterable[Any] = getattr(data, "table_cells", None) or []
        table_page = _first_page_number(item)
        for cell in cells:
            cell_text = str(getattr(cell, "text", "") or "").strip()
            if not cell_text:
                continue
            span = _span_from_bbox(
                document=document,
                text=cell_text,
                page_no=table_page,
                bbox=getattr(cell, "bbox", None),
            )
            if span is not None:
                spans.append(span)

    return _deduplicate_spans(spans)


def _span_from_bbox(*, document: Any, text: str, page_no: Any, bbox: Any) -> dict[str, Any] | None:
    if bbox is None or page_no is None:
        return None
    try:
        page_no_i = int(page_no)
    except (TypeError, ValueError):
        return None

    size = _page_size(document, page_no_i)
    if size is None:
        return None
    page_width, page_height = size

    try:
        if hasattr(bbox, "to_top_left_origin"):
            bbox = bbox.to_top_left_origin(page_height=page_height)
        left = float(getattr(bbox, "l"))
        top = float(getattr(bbox, "t"))
        right = float(getattr(bbox, "r"))
        bottom = float(getattr(bbox, "b"))
    except (TypeError, ValueError, AttributeError):
        return None

    x0 = _clamp01(min(left, right) / page_width)
    x1 = _clamp01(max(left, right) / page_width)
    y0 = _clamp01(min(top, bottom) / page_height)
    y1 = _clamp01(max(top, bottom) / page_height)

    if x1 <= x0 or y1 <= y0:
        return None

    return {
        "text": text,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "page_number": page_no_i,
        "confidence": 1.0,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _deduplicate_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    for span in spans:
        key = (
            span["page_number"],
            span["text"],
            round(span["x0"], 6),
            round(span["y0"], 6),
            round(span["x1"], 6),
            round(span["y1"], 6),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(span)
    return output


def _augment_pdf_sensitive_sources(path: Path, page_texts: list[str]) -> list[str]:
    """Include form values and PDF metadata in privacy profiling text.

    Docling focuses on document content/layout. The platform separately
    treats AcroForm values and the PDF Info dictionary as privacy-relevant
    sources, so preserve that existing behavior when Docling is selected.
    """
    pages = list(page_texts)
    if not pages:
        pages = [""]

    try:
        for field in extract_acroform_field_values(path):
            index = int(field["page_number"]) - 1
            while len(pages) <= index:
                pages.append("")
            pages[index] = _append_line(pages[index], f"{field['name']}: {field['value']}")
    except ExtractionError:
        pass

    try:
        for entry in extract_pdf_info_metadata(path):
            pages[0] = _append_line(pages[0], f"{entry['name']}: {entry['value']}")
    except ExtractionError:
        pass

    return pages


def _append_line(existing: str, line: str) -> str:
    return f"{existing}\n{line}".strip() if existing else line.strip()
