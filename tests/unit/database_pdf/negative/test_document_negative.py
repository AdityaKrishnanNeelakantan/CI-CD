from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.errors import OCRUnavailableError
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.service import (
    DOCUMENT_PROFILE_FILENAME,
    DocumentProfileLoadError,
    load_document_profile,
    run_document_profiling,
)
from tests.fixtures.pdf_factory import make_pdf, make_scanned_pdf

pytestmark = pytest.mark.negative


def test_document_profiling_fails_closed_on_missing_file(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_document_profiling(
        PDFDocumentAdapter(), tmp_path / "missing.pdf", "doc1", manifest, tmp_path / "metadata"
    )
    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}").exists()


def test_document_profiling_fails_closed_on_unsupported_file_type(tmp_path: Path):
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("not a pdf", encoding="utf-8")
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_document_profiling(
        PDFDocumentAdapter(), txt_path, "doc1", manifest, tmp_path / "metadata"
    )
    assert result.status == "failed"
    assert result.errors


def test_document_profiling_fails_closed_on_corrupt_pdf(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4 not a real structure" + b"\x00" * 50)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_document_profiling(
        PDFDocumentAdapter(), corrupt_path, "doc1", manifest, tmp_path / "metadata"
    )
    assert result.status == "failed"
    assert result.errors


def test_document_profiling_fails_closed_on_oversized_file(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Some content that takes up more than a few bytes."])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    tiny_adapter = PDFDocumentAdapter(max_file_size_bytes=10)
    result = run_document_profiling(tiny_adapter, pdf_path, "doc1", manifest, tmp_path / "metadata")
    assert result.status == "failed"
    assert result.errors


def test_document_profiling_never_overwrites_existing_output(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Some content."])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, tmp_path / "metadata")
    with pytest.raises(RuntimeError):
        run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, tmp_path / "metadata")


def test_load_document_profile_rejects_missing_file(tmp_path: Path):
    with pytest.raises(DocumentProfileLoadError):
        load_document_profile(tmp_path / "does_not_exist.json")


def test_load_document_profile_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "document_profile.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DocumentProfileLoadError):
        load_document_profile(path)


def test_load_document_profile_rejects_missing_top_level_keys(tmp_path: Path):
    path = tmp_path / "document_profile.json"
    path.write_text(json.dumps({"doc_id": "x"}), encoding="utf-8")
    with pytest.raises(DocumentProfileLoadError):
        load_document_profile(path)


def test_scanned_pdf_degrades_cleanly_when_ocr_toolchain_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A scanned PDF must never silently report a successful empty-text
    profile: when OCR can't run, the profile must still show pdf_type
    "scanned", extraction_method "native", char_count 0, and an explicit
    ocr_required warning - the stage itself still succeeds (there is
    genuine, honestly-reported evidence to persist), it just can't extract.
    """
    import synth_platform.engine.documents.pdf.extraction_router as router_module

    def _raise(*args, **kwargs):
        raise OCRUnavailableError("no ocr toolchain on this machine")

    monkeypatch.setattr(router_module, "run_ocr", _raise)

    pdf_path = make_scanned_pdf(tmp_path / "scan.pdf", [["Some scanned text."]])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_document_profiling(PDFDocumentAdapter(), pdf_path, "scan1", manifest, tmp_path / "metadata")
    assert result.is_success()

    profile = load_document_profile(manifest.output_path(f"documents/scan1/{DOCUMENT_PROFILE_FILENAME}"))
    assert profile["extraction_method"] == "native"
    assert profile["pdf_classification"]["pdf_type"] == "scanned"
    assert profile["ocr"] is None
    assert profile["statistics"]["char_count"] == 0
    assert "ocr_required" in profile["extraction_warnings"]
