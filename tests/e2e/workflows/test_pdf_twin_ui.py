"""End-to-end UI test for the PDF Twin Streamlit page."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from synth_platform.engine.documents.pdf.ocr_engine import is_ocr_available
from tests.fixtures.pdf_factory import make_scanned_pdf

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/pdf_twin.py"


def _click(at: AppTest, label: str) -> None:
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.skipif(not is_ocr_available(), reason="Tesseract/poppler not installed on this machine")
def test_full_pdf_twin_wizard_on_the_real_scanned_statement(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MOCK", "1")
    sample_pdf = make_scanned_pdf(
        tmp_path / "dummy_statement.pdf",
        [
            [
                "Bank Statement",
                "Customer: Jordan Lee",
                "Account Number: 123456789",
                "Balance: 2450.75",
                "Transaction: Grocery Market 54.20",
            ]
        ],
    )
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload(sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")
    at.run()
    assert not at.exception

    # This regression specifically protects the original native/OCR path.
    # Docling has its own full backend integration test.
    extraction_engine = next(s for s in at.selectbox if s.label == "Extraction engine")
    extraction_engine.set_value("Automatic (native/OCR)").run()
    assert not at.exception

    _click(at, "Discover document")
    _click(at, "Build twin template")
    _click(at, "Understand fields")
    _click(at, "Generate values")
    _click(at, "Build twin document")
    _click(at, "Validate twin")

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Extraction method"] == "ocr"
    assert metrics["Integrity checks"] == "passed"
    assert metrics["Field accuracy"] == "100%"
    assert metrics["Overflow failures"] == "0"
