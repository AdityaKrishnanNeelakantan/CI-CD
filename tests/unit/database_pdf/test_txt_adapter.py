from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.errors import DocumentAdapterError, UnsupportedFileTypeError
from synth_platform.engine.documents.pdf.txt_adapter import TXTDocumentAdapter

pytestmark = pytest.mark.unit


def test_extract_returns_full_text_and_correct_offsets(tmp_path: Path):
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("First page text.\fSecond page text.", encoding="utf-8")
    doc = TXTDocumentAdapter().extract(txt_path)

    assert doc["source_format"] == "txt"
    assert len(doc["pages"]) == 2
    page1, page2 = doc["pages"]
    assert doc["text"][page1["start"] : page1["end"]] == "First page text."
    assert doc["text"][page2["start"] : page2["end"]] == "Second page text."


def test_extract_without_form_feed_is_a_single_page(tmp_path: Path):
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("Just one page of plain text.", encoding="utf-8")
    doc = TXTDocumentAdapter().extract(txt_path)
    assert len(doc["pages"]) == 1


def test_extract_computes_correct_text_hash(tmp_path: Path):
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("Hello world.", encoding="utf-8")
    doc = TXTDocumentAdapter().extract(txt_path)
    expected = "sha256:" + hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
    assert doc["text_hash"] == expected


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(DocumentAdapterError):
        TXTDocumentAdapter().extract(tmp_path / "does_not_exist.txt")


def test_wrong_extension_raises(tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 not a real pdf")
    with pytest.raises(UnsupportedFileTypeError):
        TXTDocumentAdapter().extract(pdf_path)


def test_empty_file_raises(tmp_path: Path):
    txt_path = tmp_path / "empty.txt"
    txt_path.write_text("", encoding="utf-8")
    with pytest.raises(DocumentAdapterError):
        TXTDocumentAdapter().extract(txt_path)
