"""End-to-end UI test: drives the actual src/synth_platform/interfaces/streamlit/pages/pdf_twin.py through
Streamlit's own AppTest harness against the real scanned bank statement
fixture (examples/fixtures/pdfs/dummy_statement.pdf) - the same document used throughout
this project's backend test suite as the real-world proof case.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/pdf_twin.py"
SAMPLE_PDF = Path("examples/fixtures/pdfs/dummy_statement.pdf")


def _click(at: AppTest, label: str) -> None:
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, [str(e) for e in at.exception]


def test_full_pdf_twin_wizard_on_the_real_scanned_statement():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload("dummy_statement.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")
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
