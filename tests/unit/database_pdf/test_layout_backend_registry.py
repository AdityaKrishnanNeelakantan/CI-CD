from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.layout_backends.base import LayoutBackend
from synth_platform.engine.documents.pdf.layout_backends.docling_backend import DoclingLayoutBackend
from synth_platform.engine.documents.pdf.layout_backends.native_backend import NativeLayoutBackend
from synth_platform.engine.documents.pdf.layout_backends.ocr_backend import OcrLayoutBackend
from synth_platform.engine.documents.pdf.layout_backends.registry import (
    UnknownLayoutBackendError,
    get_layout_backend_class,
    resolve_layout_backend,
)
from synth_platform.engine.documents.pdf.docling_engine import is_docling_available
from synth_platform.engine.documents.pdf.native_layout import extract_positioned_spans
from synth_platform.engine.documents.pdf.ocr_engine import is_ocr_available
from tests.fixtures.pdf_factory import make_pdf, make_scanned_pdf

pytestmark = pytest.mark.unit


def test_resolve_layout_backend_native_returns_native_backend_instance():
    backend = resolve_layout_backend("native")
    assert isinstance(backend, NativeLayoutBackend)
    assert isinstance(backend, LayoutBackend)
    assert backend.backend_name == "native"


def test_resolve_layout_backend_ocr_returns_ocr_backend_instance():
    backend = resolve_layout_backend("ocr")
    assert isinstance(backend, OcrLayoutBackend)
    assert isinstance(backend, LayoutBackend)
    assert backend.backend_name == "ocr"


def test_resolve_layout_backend_docling_returns_docling_backend_instance():
    backend = resolve_layout_backend("docling")
    assert isinstance(backend, DoclingLayoutBackend)
    assert isinstance(backend, LayoutBackend)
    assert backend.backend_name == "docling"


def test_get_layout_backend_class_rejects_unknown_name():
    with pytest.raises(UnknownLayoutBackendError, match="not-a-backend"):
        get_layout_backend_class("not-a-backend")


def test_native_backend_produces_same_spans_as_extract_positioned_spans(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Some text near the top of the page."])
    backend_spans = NativeLayoutBackend().extract_spans(pdf_path)
    direct_spans = extract_positioned_spans(pdf_path)
    assert backend_spans == direct_spans


@pytest.mark.skipif(not is_ocr_available(), reason="Tesseract/poppler not installed on this machine")
def test_ocr_backend_produces_positioned_spans(tmp_path: Path):
    pdf_path = make_scanned_pdf(tmp_path / "scan.pdf", [["INVOICE NUMBER 8823"]])
    spans = OcrLayoutBackend().extract_spans(pdf_path)
    assert spans
    assert all({"text", "x0", "y0", "x1", "y1", "page_number"} <= set(span) for span in spans)


@pytest.mark.skipif(not is_docling_available(), reason="Docling not installed on this machine")
def test_docling_backend_produces_positioned_spans(tmp_path: Path):
    pdf_path = make_pdf(
        tmp_path / "docling.pdf",
        ["Account Number: 8823471\nCustomer Name: Grace Hopper"],
    )
    spans = DoclingLayoutBackend().extract_spans(pdf_path)
    assert spans
    assert all({"text", "x0", "y0", "x1", "y1", "page_number"} <= set(span) for span in spans)
    assert all(0.0 <= span["x0"] <= span["x1"] <= 1.0 for span in spans)
    assert all(0.0 <= span["y0"] <= span["y1"] <= 1.0 for span in spans)
