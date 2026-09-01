"""End-to-end: the entire currently-available unstructured/PDF pipeline succeeds.

Parallel to tests/e2e/test_pipeline_e2e.py's structured Connect->Clean
chain, but for documents: Extract -> Statistics/Language/PII/Entities/
Classification -> Duplicate check -> persisted evidence, re-read entirely
from disk afterward, plus duplicate detection proven across two
independent runs (the document pipeline's analogue of "override survives
a rerun").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.ocr_engine import is_ocr_available
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
from tests.fixtures.pdf_factory import make_pdf, make_pdf_with_metadata, make_scanned_pdf

pytestmark = pytest.mark.e2e


def test_document_pipeline_is_fully_traceable(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    pdf_path = make_pdf(
        tmp_path / "contract.pdf",
        [
            "AGREEMENT\nThis Agreement is entered into by and between the Parties, "
            "WHEREAS the Parties wish to define their rights hereby.\n"
            "Signed by Dr. Grace Hopper of Acme Corp on 2024-06-01. "
            "Contact: grace@example.com."
        ],
    )

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "contract1", manifest, metadata_dir
    )
    assert result.is_success()

    # Simulate a fresh process: re-read every artifact from disk only.
    manifest_on_disk = json.loads((manifest.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    profile_on_disk = load_document_profile(
        manifest.output_path(f"documents/contract1/{DOCUMENT_PROFILE_FILENAME}")
    )

    assert manifest_on_disk["stages"][0]["stage_name"] == "document_profiling"
    assert manifest_on_disk["stages"][0]["status"] == "success"
    assert manifest_on_disk["stages"][0]["output_references"] == [
        str(manifest.output_path(f"documents/contract1/{DOCUMENT_PROFILE_FILENAME}"))
    ]

    assert profile_on_disk["classification"]["document_type"] == "contract"
    assert any(f["type"] == "email" for f in profile_on_disk["pii_findings"])
    assert any(f["type"] == "PERSON" for f in profile_on_disk["entities"])
    assert any(f["type"] == "ORG" for f in profile_on_disk["entities"])
    assert any(f["type"] == "DATE" for f in profile_on_disk["entities"])
    assert "text" not in profile_on_disk  # never persists raw extracted content


def test_pdf_info_metadata_pii_is_detected_but_never_persisted_as_raw_text(tmp_path: Path):
    """A PDF's Info dictionary (Author/Title/...) is a real, well-known
    metadata-leakage vector distinct from page content - Office/PDF
    exporters routinely auto-populate /Author with the actual author's
    real name. Proves the profiling stage now scans it (PII/entity
    findings surface it) while still upholding the project's existing
    never-persist-raw-text contract for every other field.
    """
    metadata_dir = tmp_path / "metadata"
    pdf_path = make_pdf_with_metadata(
        tmp_path / "meta.pdf",
        ["Ordinary body text with no PII of its own."],
        author="Grace Hopper",
        title="Contact grace.hopper@example.com for questions",
    )

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_document_profiling(PDFDocumentAdapter(), pdf_path, "meta1", manifest, metadata_dir)
    assert result.is_success()

    profile_on_disk = load_document_profile(
        manifest.output_path(f"documents/meta1/{DOCUMENT_PROFILE_FILENAME}")
    )
    assert any(f["type"] == "email" for f in profile_on_disk["pii_findings"])
    assert any(f["type"] == "PERSON" for f in profile_on_disk["entities"])

    serialized_profile = json.dumps(profile_on_disk)
    assert "grace.hopper@example.com" not in serialized_profile
    assert "Grace Hopper" not in serialized_profile
    assert "text" not in profile_on_disk


def test_duplicate_detection_holds_across_independent_pipeline_runs(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    content = ["Identical content for duplicate-detection proof across two full runs."]

    pdf_1 = make_pdf(tmp_path / "first.pdf", content)
    manifest_1 = RunManifest.create(runs_dir=tmp_path / "runs")
    run_document_profiling(PDFDocumentAdapter(), pdf_1, "doc1", manifest_1, metadata_dir)

    # A second, completely independent run (new RunManifest/run_id) must
    # still see the first run's document in the durable metadata/ index.
    pdf_2 = make_pdf(tmp_path / "second.pdf", content)
    manifest_2 = RunManifest.create(runs_dir=tmp_path / "runs")
    run_document_profiling(PDFDocumentAdapter(), pdf_2, "doc2", manifest_2, metadata_dir)

    profile_2 = load_document_profile(
        manifest_2.output_path(f"documents/doc2/{DOCUMENT_PROFILE_FILENAME}")
    )
    assert profile_2["duplicate"]["is_duplicate"] is True
    assert profile_2["duplicate"]["duplicate_of"] == str(pdf_1)


@pytest.mark.skipif(not is_ocr_available(), reason="Tesseract/poppler not installed on this machine")
def test_scanned_document_pipeline_is_fully_traceable(tmp_path: Path):
    """Same traceability proof as test_document_pipeline_is_fully_traceable,
    but for a PDF with no native text layer at all - the OCR fallback path
    must persist the same evidence shape, re-read entirely from disk.
    """
    metadata_dir = tmp_path / "metadata"
    pdf_path = make_scanned_pdf(
        tmp_path / "scan.pdf",
        [["AGREEMENT", "Signed by Dr. Grace Hopper of Acme Corp on 2024-06-01."]],
    )

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_document_profiling(PDFDocumentAdapter(), pdf_path, "scan1", manifest, metadata_dir)
    assert result.is_success()

    profile_on_disk = load_document_profile(
        manifest.output_path(f"documents/scan1/{DOCUMENT_PROFILE_FILENAME}")
    )

    assert profile_on_disk["extraction_method"] == "ocr"
    assert profile_on_disk["pdf_classification"]["pdf_type"] == "scanned"
    assert profile_on_disk["ocr"]["mean_confidence"] > 0
    assert any(f["type"] == "PERSON" for f in profile_on_disk["entities"])
    assert any(f["type"] == "DATE" for f in profile_on_disk["entities"])
    assert "text" not in profile_on_disk  # OCR'd text must never be persisted either
