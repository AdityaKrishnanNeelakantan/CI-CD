from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.errors import OCRUnavailableError
from synth_platform.engine.documents.pdf.ocr_engine import is_ocr_available, resolve_tesseract_cmd, run_ocr
from tests.fixtures.pdf_factory import make_scanned_pdf

pytestmark = pytest.mark.unit


def test_resolve_tesseract_cmd_prefers_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_binary = tmp_path / "tesseract.exe"
    fake_binary.write_bytes(b"")
    resolve_tesseract_cmd.cache_clear()
    monkeypatch.setenv("TESSERACT_CMD", str(fake_binary))
    assert resolve_tesseract_cmd() == str(fake_binary)
    resolve_tesseract_cmd.cache_clear()


def test_resolve_tesseract_cmd_ignores_nonexistent_env_var(monkeypatch: pytest.MonkeyPatch):
    resolve_tesseract_cmd.cache_clear()
    monkeypatch.setenv("TESSERACT_CMD", "Z:/does/not/exist/tesseract.exe")
    result = resolve_tesseract_cmd()
    assert result != "Z:/does/not/exist/tesseract.exe"
    resolve_tesseract_cmd.cache_clear()


def test_run_ocr_raises_ocr_unavailable_when_no_tesseract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import synth_platform.engine.documents.pdf.ocr_engine as ocr_engine_module

    monkeypatch.setattr(ocr_engine_module, "resolve_tesseract_cmd", lambda: None)
    pdf_path = make_scanned_pdf(tmp_path / "scan.pdf", [["Some text."]])
    with pytest.raises(OCRUnavailableError):
        run_ocr(pdf_path)


@pytest.mark.skipif(not is_ocr_available(), reason="Tesseract/poppler not installed on this machine")
def test_run_ocr_reads_back_rendered_text(tmp_path: Path):
    pdf_path = make_scanned_pdf(
        tmp_path / "scan.pdf", [["INVOICE NUMBER 8823"], ["SECOND PAGE TEXT"]]
    )
    result = run_ocr(pdf_path)
    assert len(result["pages_text"]) == 2
    assert "INVOICE" in result["pages_text"][0]
    assert "SECOND" in result["pages_text"][1]
    assert result["mean_confidence"] > 0
