"""Contract: extract_with_fallback() always returns the same NormalisedDocument
shape (everything PDFDocumentAdapter.extract() returns) plus exactly three
extra keys, regardless of whether the native or OCR path was taken - so
every downstream detector (statistics, language, PII, entities,
classification) can keep consuming the dict without caring which path ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.extraction_router import (
    EXTRACTION_METHOD_DOCLING,
    EXTRACTION_METHOD_NATIVE,
    EXTRACTION_METHOD_OCR,
    extract_with_fallback,
)
from synth_platform.engine.documents.pdf.ocr_engine import is_ocr_available
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from tests.fixtures.pdf_factory import make_pdf, make_scanned_pdf

pytestmark = pytest.mark.contract

REQUIRED_KEYS = {
    "source_reference",
    "source_format",
    "extracted_at",
    "text",
    "pages",
    "text_hash",
    "file_size_bytes",
    "extraction_warnings",
    "extraction_method",
    "pdf_classification",
    "ocr",
}


def test_native_path_shape(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Real embedded text content."])
    result = extract_with_fallback(PDFDocumentAdapter(), pdf_path)
    assert set(result) == REQUIRED_KEYS
    assert result["extraction_method"] == EXTRACTION_METHOD_NATIVE
    assert result["ocr"] is None
    assert result["pdf_classification"]["pdf_type"] == "text"


@pytest.mark.skipif(not is_ocr_available(), reason="Tesseract/poppler not installed on this machine")
def test_ocr_path_shape(tmp_path: Path):
    pdf_path = make_scanned_pdf(tmp_path / "scan.pdf", [["Some scanned text."]])
    result = extract_with_fallback(PDFDocumentAdapter(), pdf_path)
    assert set(result) == REQUIRED_KEYS
    assert result["extraction_method"] == EXTRACTION_METHOD_OCR
    assert result["ocr"] is not None
    assert result["pdf_classification"]["pdf_type"] == "scanned"


def test_ocr_unavailable_path_shape_still_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import synth_platform.engine.documents.pdf.extraction_router as router_module

    def _raise(*args, **kwargs):
        from synth_platform.engine.documents.pdf.errors import OCRUnavailableError

        raise OCRUnavailableError("no ocr toolchain")

    monkeypatch.setattr(router_module, "run_ocr", _raise)
    pdf_path = make_scanned_pdf(tmp_path / "scan.pdf", [["Some scanned text."]])
    result = extract_with_fallback(PDFDocumentAdapter(), pdf_path)
    assert set(result) == REQUIRED_KEYS
    assert result["extraction_method"] == EXTRACTION_METHOD_NATIVE
    assert result["ocr"] is None
    assert "ocr_required" in result["extraction_warnings"]


def test_docling_path_preserves_normalized_document_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import synth_platform.engine.documents.pdf.extraction_router as router_module

    monkeypatch.setattr(
        router_module,
        "extract_with_docling",
        lambda _path: {
            "pages_text": ["Account Number: 8823471"],
            "positioned_spans": [],
            "page_count": 1,
            "version": "test",
        },
    )
    pdf_path = make_pdf(tmp_path / "docling.pdf", ["Native source text."])
    result = extract_with_fallback(
        PDFDocumentAdapter(), pdf_path, preferred_method="docling"
    )
    assert set(result) == REQUIRED_KEYS
    assert result["extraction_method"] == EXTRACTION_METHOD_DOCLING
    assert result["text"] == "Account Number: 8823471"
    assert result["ocr"] is None
    assert result["pdf_classification"]["pdf_type"] == "text"
