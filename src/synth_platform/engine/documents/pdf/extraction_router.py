"""Routes PDF extraction across native, Docling, and OCR backends.

The default ``auto`` behavior preserves the original product contract:
validate/extract native PDF text first, then use OCR only when the PDF is
classified as scanned or native extraction produced no usable text.

Callers may explicitly request ``docling``. In that mode the source is still
validated through ``PDFDocumentAdapter`` first (so file-size/encryption/input
guards stay consistent), then Docling becomes the content extraction engine.
The selected extraction method is persisted in ``document_profile.json`` and
is reused by template compilation for geometry extraction.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.documents.pdf.base import DocumentAdapter
from synth_platform.engine.documents.pdf.docling_engine import extract_with_docling
from synth_platform.engine.documents.pdf.errors import ExtractionError, OCRUnavailableError
from synth_platform.engine.documents.pdf.ocr_engine import run_ocr
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.pdf_classifier import classify_pdf_type
from synth_platform.engine.documents.pdf.preflight import run_pdf_preflight

EXTRACTION_METHOD_NATIVE = "native"
EXTRACTION_METHOD_OCR = "ocr"
EXTRACTION_METHOD_DOCLING = "docling"
EXTRACTION_METHOD_AUTO = "auto"

_SUPPORTED_PREFERRED_METHODS = {
    EXTRACTION_METHOD_AUTO,
    EXTRACTION_METHOD_NATIVE,
    EXTRACTION_METHOD_OCR,
    EXTRACTION_METHOD_DOCLING,
}


def extract_with_fallback(
    adapter: DocumentAdapter,
    path: str | Path,
    preferred_method: str | None = None,
) -> dict[str, Any]:
    """Extract a document using the requested or automatically selected path.

    ``preferred_method=None`` and ``"auto"`` are equivalent and preserve the
    existing native->OCR fallback behavior. Explicit ``docling`` never silently
    falls back to another engine: if Docling is unavailable or conversion fails,
    the stage fails so the profile cannot claim it used Docling when it did not.
    """
    method = (preferred_method or EXTRACTION_METHOD_AUTO).strip().lower()
    if method not in _SUPPORTED_PREFERRED_METHODS:
        known = ", ".join(sorted(_SUPPORTED_PREFERRED_METHODS))
        raise ExtractionError(f"Unknown PDF extraction method {method!r}. Expected one of: {known}.")

    # Native adapter extraction always runs first because it owns source-file
    # validation, encryption handling, file-size limits, and metadata/form PII.
    native = adapter.extract(path)

    if not isinstance(adapter, PDFDocumentAdapter):
        if method not in {EXTRACTION_METHOD_AUTO, EXTRACTION_METHOD_NATIVE}:
            raise ExtractionError(
                f"Extraction method {method!r} is PDF-specific and cannot be used with "
                f"{type(adapter).__name__}."
            )
        return {
            **native,
            "extraction_method": EXTRACTION_METHOD_NATIVE,
            "pdf_classification": None,
            "ocr": None,
        }

    preflight = run_pdf_preflight(path)
    classification = classify_pdf_type(preflight)

    if method == EXTRACTION_METHOD_DOCLING:
        docling_result = extract_with_docling(path)
        return _normalise_extracted_pages(
            native=native,
            pages_text=docling_result["pages_text"],
            extraction_method=EXTRACTION_METHOD_DOCLING,
            classification=classification,
            warnings=[],
            ocr=None,
        )

    if method == EXTRACTION_METHOD_NATIVE:
        return {
            **native,
            "extraction_method": EXTRACTION_METHOD_NATIVE,
            "pdf_classification": classification,
            "ocr": None,
        }

    needs_ocr = (
        method == EXTRACTION_METHOD_OCR
        or classification["pdf_type"] == "scanned"
        or "empty_extracted_text" in native["extraction_warnings"]
    )
    if not needs_ocr:
        return {
            **native,
            "extraction_method": EXTRACTION_METHOD_NATIVE,
            "pdf_classification": classification,
            "ocr": None,
        }

    try:
        ocr_result = run_ocr(path)
    except OCRUnavailableError as exc:
        if method == EXTRACTION_METHOD_OCR:
            raise
        warnings = list(native["extraction_warnings"]) + ["ocr_required", f"ocr_unavailable:{exc}"]
        return {
            **native,
            "extraction_warnings": warnings,
            "extraction_method": EXTRACTION_METHOD_NATIVE,
            "pdf_classification": classification,
            "ocr": None,
        }

    warnings = [w for w in native["extraction_warnings"] if w != "empty_extracted_text"]
    if ocr_result["mean_confidence"] > 0 and ocr_result["mean_confidence"] < 60.0:
        warnings.append(f"low_ocr_confidence={ocr_result['mean_confidence']}")

    return _normalise_extracted_pages(
        native=native,
        pages_text=ocr_result["pages_text"],
        extraction_method=EXTRACTION_METHOD_OCR,
        classification=classification,
        warnings=warnings,
        ocr={"mean_confidence": ocr_result["mean_confidence"], "dpi": ocr_result["dpi"]},
    )


def _normalise_extracted_pages(
    *,
    native: dict[str, Any],
    pages_text: list[str],
    extraction_method: str,
    classification: dict[str, Any],
    warnings: list[str],
    ocr: dict[str, Any] | None,
) -> dict[str, Any]:
    full_text, page_offsets = _join_pages(pages_text)
    normalized_warnings = list(warnings)
    if not full_text.strip() and "empty_extracted_text" not in normalized_warnings:
        normalized_warnings.append("empty_extracted_text")

    return {
        "source_reference": native["source_reference"],
        "source_format": native["source_format"],
        "extracted_at": datetime.now(UTC).isoformat(),
        "text": full_text,
        "pages": [
            {"page_number": i + 1, "start": offset["start"], "end": offset["end"]}
            for i, offset in enumerate(page_offsets)
        ],
        "text_hash": "sha256:" + hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        "file_size_bytes": native["file_size_bytes"],
        "extraction_warnings": normalized_warnings,
        "extraction_method": extraction_method,
        "pdf_classification": classification,
        "ocr": ocr,
    }


def _join_pages(pages: list[str]) -> tuple[str, list[dict[str, int]]]:
    text_parts: list[str] = []
    offsets: list[dict[str, int]] = []
    cursor = 0
    for page_text in pages:
        start = cursor
        text_parts.append(page_text)
        cursor += len(page_text)
        offsets.append({"start": start, "end": cursor})
        text_parts.append("\n")
        cursor += 1
    full_text = "".join(text_parts).rstrip("\n") if text_parts else ""
    return full_text, offsets
