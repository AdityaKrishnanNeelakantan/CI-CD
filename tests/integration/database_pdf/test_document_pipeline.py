"""Integration: PDFDocumentAdapter -> all detectors -> service -> manifest,
plus duplicate detection working across two separate documents/runs
sharing the durable metadata/ index.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.ocr_engine import is_ocr_available
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
from tests.fixtures.pdf_factory import make_pdf, make_scanned_pdf

pytestmark = pytest.mark.integration


def test_document_profiling_produces_consistent_evidence(tmp_path: Path):
    pdf_path = make_pdf(
        tmp_path / "letter.pdf",
        ["Dear Dr. Grace Hopper,\nPlease reply to grace@example.com by 2024-06-01.\nSincerely, the team."],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    result = run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "letter1", manifest, tmp_path / "metadata"
    )
    assert result.is_success()

    profile = load_document_profile(
        manifest.output_path(f"documents/letter1/{DOCUMENT_PROFILE_FILENAME}")
    )
    assert profile["classification"]["document_type"] == "letter"
    assert profile["language"]["language"] == "en"
    assert any(f["type"] == "email" for f in profile["pii_findings"])
    assert any(f["type"] == "PERSON" for f in profile["entities"])
    assert any(f["type"] == "DATE" for f in profile["entities"])
    assert profile["duplicate"]["is_duplicate"] is False

    manifest_data_path = manifest.run_dir / "run_manifest.json"
    import json

    manifest_data = json.loads(manifest_data_path.read_text())
    assert manifest_data["stages"][0]["stage_name"] == "document_profiling"
    assert manifest_data["stages"][0]["status"] == "success"


def test_duplicate_document_is_detected_across_separate_runs(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    content = ["This exact same content appears in two different files."]

    pdf_a = make_pdf(tmp_path / "a.pdf", content)
    manifest_a = RunManifest.create(runs_dir=tmp_path / "runs")
    result_a = run_document_profiling(PDFDocumentAdapter(), pdf_a, "doc_a", manifest_a, metadata_dir)
    profile_a = load_document_profile(manifest_a.output_path(f"documents/doc_a/{DOCUMENT_PROFILE_FILENAME}"))
    assert profile_a["duplicate"]["is_duplicate"] is False

    pdf_b = make_pdf(tmp_path / "b.pdf", content)
    manifest_b = RunManifest.create(runs_dir=tmp_path / "runs")
    result_b = run_document_profiling(PDFDocumentAdapter(), pdf_b, "doc_b", manifest_b, metadata_dir)
    profile_b = load_document_profile(manifest_b.output_path(f"documents/doc_b/{DOCUMENT_PROFILE_FILENAME}"))

    assert profile_b["duplicate"]["is_duplicate"] is True
    assert profile_b["duplicate"]["duplicate_of"] == str(pdf_a)
    assert result_a.is_success() and result_b.is_success()


@pytest.mark.skipif(not is_ocr_available(), reason="Tesseract/poppler not installed on this machine")
def test_scanned_pdf_falls_back_to_ocr_and_downstream_stages_still_run(tmp_path: Path):
    pdf_path = make_scanned_pdf(
        tmp_path / "scan.pdf",
        [["Dear Dr. Grace Hopper,", "Please reply by 2024-06-01.", "Sincerely, the team."]],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    result = run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "scan1", manifest, tmp_path / "metadata"
    )
    assert result.is_success()

    profile = load_document_profile(manifest.output_path(f"documents/scan1/{DOCUMENT_PROFILE_FILENAME}"))
    assert profile["extraction_method"] == "ocr"
    assert profile["pdf_classification"]["pdf_type"] == "scanned"
    assert profile["ocr"]["mean_confidence"] > 0
    assert profile["statistics"]["char_count"] > 0
    assert any(f["type"] == "PERSON" for f in profile["entities"])
    assert any(f["type"] == "DATE" for f in profile["entities"])
