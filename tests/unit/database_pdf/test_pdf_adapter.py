from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.errors import (
    DocumentAdapterError,
    ExtractionError,
    UnsupportedFileTypeError,
)
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from tests.fixtures.pdf_factory import make_acroform_pdf, make_pdf, make_pdf_with_metadata

pytestmark = pytest.mark.unit


def test_extract_returns_full_text_and_correct_offsets(tmp_path: Path):
    pdf_path = make_pdf(
        tmp_path / "doc.pdf",
        ["First page text.", "Second page text."],
    )
    doc = PDFDocumentAdapter().extract(pdf_path)

    assert doc["source_format"] == "pdf"
    assert len(doc["pages"]) == 2
    page1, page2 = doc["pages"]

    assert doc["text"][page1["start"] : page1["end"]] == "First page text."
    assert doc["text"][page2["start"] : page2["end"]] == "Second page text."


def test_extract_computes_correct_text_hash(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Hello world."])
    doc = PDFDocumentAdapter().extract(pdf_path)
    expected = "sha256:" + hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
    assert doc["text_hash"] == expected


def test_extract_records_file_size_and_no_warnings_for_normal_doc(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Some content."])
    doc = PDFDocumentAdapter().extract(pdf_path)
    assert doc["file_size_bytes"] == pdf_path.stat().st_size
    assert doc["extraction_warnings"] == []


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(DocumentAdapterError):
        PDFDocumentAdapter().extract(tmp_path / "does_not_exist.pdf")


def test_unsupported_extension_raises(tmp_path: Path):
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        PDFDocumentAdapter().extract(txt_path)


def test_empty_file_raises(tmp_path: Path):
    empty_path = tmp_path / "empty.pdf"
    empty_path.write_bytes(b"")
    with pytest.raises(ExtractionError):
        PDFDocumentAdapter().extract(empty_path)


def test_oversized_file_raises(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Some content that takes up space."])
    with pytest.raises(DocumentAdapterError):
        PDFDocumentAdapter(max_file_size_bytes=10).extract(pdf_path)


def test_corrupt_pdf_raises_extraction_error(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4 this is not a real pdf structure" + b"\x00" * 50)
    with pytest.raises(ExtractionError):
        PDFDocumentAdapter().extract(corrupt_path)


def test_document_with_no_extractable_text_is_flagged(tmp_path: Path):
    # A single blank page produces near-empty text.
    pdf_path = make_pdf(tmp_path / "blank.pdf", [""])
    doc = PDFDocumentAdapter().extract(pdf_path)
    assert "empty_extracted_text" in doc["extraction_warnings"]


def test_acroform_field_values_appear_in_extracted_text(tmp_path: Path):
    """Regression: page.extract_text() cannot see AcroForm field values
    (verified empirically), so a real fillable form's actual data used to
    be completely absent from the document profile - invisible to PII
    scanning, entity detection, and the text_hash alike.
    """
    pdf_path = make_acroform_pdf(
        tmp_path / "acroform.pdf", [("customer_name", "Ada Lovelace", [100, 700, 300, 720])]
    )
    doc = PDFDocumentAdapter().extract(pdf_path)
    assert "Ada Lovelace" in doc["text"]
    assert "empty_extracted_text" not in doc["extraction_warnings"]


def test_pdf_info_metadata_appears_in_extracted_text(tmp_path: Path):
    """A PDF's Info dictionary (Author/Title/...) is a real, well-known
    metadata-leakage vector distinct from page content - Office/PDF
    exporters routinely auto-populate /Author with the actual author's
    real name, invisible to plain page.extract_text() alone.
    """
    pdf_path = make_pdf_with_metadata(
        tmp_path / "meta.pdf",
        ["Ordinary body text."],
        author="Ada Lovelace",
        title="Confidential Project Titan",
    )
    doc = PDFDocumentAdapter().extract(pdf_path)
    assert "Ada Lovelace" in doc["text"]
    assert "Confidential Project Titan" in doc["text"]


def test_pdf_without_metadata_fields_extracts_normally(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Plain content only."])
    doc = PDFDocumentAdapter().extract(pdf_path)
    assert doc["text"].strip() == "Plain content only."
