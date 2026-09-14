"""Golden-fixture regression suite: runs the COMPLETE PDF twin pipeline
(profiling -> template_compilation -> semantic_binding -> value_generation
-> document_rendering -> document_validation -> document_deidentification)
against every real-world PDF checked into examples/fixtures/pdfs/, end to end, with no
mocking.

This is deliberately independent of tests/e2e/test_pipeline_e2e.py and
tests/e2e/test_render_and_validation_e2e.py (which use small synthetic
fixtures): those prove the pipeline's *logic*; this proves it survives
contact with real, messy, third-party PDFs (scanned images, multi-column
academic layouts, encrypted files, oversized files) without crashing, and
that no detected PII value ever survives verbatim into the rendered twin
or the redacted PDF.

Two of the six fixtures are *expected* to fail at document_profiling by
deliberate, tested input guards (not defects) - see
src/documents/base.py's DEFAULT_MAX_FILE_SIZE_BYTES and
src/documents/pdf_adapter.py's encrypted-PDF handling. Those are asserted
as expected failures here, not skipped, so a regression that makes them
start (or stop) failing is caught.

Fixed (previously a known, accepted limitation): entities.py's PERSON
pattern only matched "First Last" / "First M. Last" name shapes and
missed bibliographic "Surname Initial" citation-list author names (e.g.
"Di Gregorio C, Frattini M"), exercised by health_report.pdf's reference
list. entities.py now also detects runs of 2+ "Surname Initial" units
chained only by list-separator punctuation (comma/"and") as a single
high-confidence PERSON span (see _find_citation_author_lists), so those
names are redacted like any other detected PERSON entity. See
/memories/repo for the full write-up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.binding_service import DOCUMENT_BINDING_MAP_FILENAME, load_document_binding_map, run_semantic_binding
from synth_platform.engine.documents.pdf.deidentification_service import (
    DEIDENTIFICATION_REPORT_FILENAME,
    REDACTED_PDF_FILENAME,
    run_document_deidentification,
)
from synth_platform.engine.documents.pdf.generation_service import (
    DOCUMENT_SYNTHETIC_VALUES_FILENAME,
    load_document_synthetic_values,
    run_value_generation,
)
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.render_service import (
    GROUND_TRUTH_FILENAME,
    RENDERED_PDF_FILENAME,
    load_document_ground_truth,
    run_document_rendering,
)
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
from synth_platform.engine.documents.pdf.template_service import DOCUMENT_TEMPLATE_FILENAME, load_document_template, run_template_compilation
from synth_platform.engine.documents.pdf.validation_service import run_document_validation
from synth_platform.application.services.transfer_service import TransferService
from synth_platform.errors import TransferBlockedError

PDFS_DIR = Path(__file__).resolve().parents[3] / "examples" / "fixtures" / "pdfs"
LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "local-data"
HEALTHCARE_SUMMARY_PDF = LOCAL_DATA_DIR / "healthcare_patient_summary.pdf"

# Real, sensitive PII categories only (src/documents/pii.py) - never the
# generic entities.py PERSON/ORG/DATE heuristics, which are documented as
# having known false positives (generic headings misread as names) and
# false negatives (citation-list author names) that are unrelated to this
# suite's purpose of proving *structured, field/table-level* PII never
# leaks.
_SUCCEEDS_END_TO_END = {
    "bank_form.pdf",
    "dummy_statement.pdf",
    "healthcare_trauma_report.pdf",
    "health_report.pdf",
    "financial_statement_analysis.pdf",
}
# Always expected to fail profiling when present in examples/fixtures/pdfs/.
_FAILS_AT_PROFILING = {
    "sample_tables.pdf": "encrypted",
}
# Large optional fixture (gitignored for GitHub size); included only if on disk.
_OPTIONAL_FAILS_AT_PROFILING = {
    "Test Case PDF.pdf": "exceeds max",
}

_FAILS_AT_PROFILING = {
    **_FAILS_AT_PROFILING,
    **{
        name: reason
        for name, reason in _OPTIONAL_FAILS_AT_PROFILING.items()
        if (PDFS_DIR / name).is_file()
    },
}

_ALL_FIXTURES = sorted(p.name for p in PDFS_DIR.glob("*.pdf")) if PDFS_DIR.is_dir() else []
_PRESENT_SUCCEEDS_END_TO_END = sorted(set(_ALL_FIXTURES) & _SUCCEEDS_END_TO_END)
_PRESENT_FAILS_AT_PROFILING = sorted(set(_ALL_FIXTURES) & set(_FAILS_AT_PROFILING))


@pytest.fixture()
def manifest(tmp_path: Path) -> RunManifest:
    return RunManifest.create(runs_dir=tmp_path / "runs")


def test_fixture_inventory_matches_expected_set() -> None:
    """Guards against a new/removed PDF fixture silently changing the
    universe this suite covers without anyone noticing."""
    expected_when_present = _SUCCEEDS_END_TO_END | set(_FAILS_AT_PROFILING)
    assert set(_ALL_FIXTURES) <= expected_when_present


@pytest.mark.parametrize("filename", _PRESENT_FAILS_AT_PROFILING)
def test_deliberately_guarded_pdfs_fail_cleanly_at_profiling(filename: str, manifest: RunManifest, tmp_path: Path) -> None:
    pdf_path = PDFS_DIR / filename
    result = run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, tmp_path / "metadata")
    assert not result.is_success()
    assert result.errors
    expected_substring = _FAILS_AT_PROFILING[filename]
    assert any(expected_substring in err for err in result.errors), result.errors


@pytest.mark.parametrize("filename", _PRESENT_SUCCEEDS_END_TO_END)
def test_real_pdf_completes_full_pipeline_with_no_pii_leakage(filename: str, manifest: RunManifest, tmp_path: Path) -> None:
    pdf_path = PDFS_DIR / filename
    doc_id = "doc1"
    metadata_dir = tmp_path / "metadata"

    profiling_result = run_document_profiling(PDFDocumentAdapter(), pdf_path, doc_id, manifest, metadata_dir)
    assert profiling_result.is_success(), profiling_result.errors
    profile = load_document_profile(manifest.output_path(f"documents/{doc_id}/{DOCUMENT_PROFILE_FILENAME}"))

    template_result = run_template_compilation(
        pdf_path, doc_id, manifest, extraction_method=profile["extraction_method"]
    )
    assert template_result.is_success(), template_result.errors
    template = load_document_template(manifest.output_path(f"documents/{doc_id}/{DOCUMENT_TEMPLATE_FILENAME}"))

    binding_result = run_semantic_binding(template, doc_id, manifest, template_reference=str(pdf_path))
    assert binding_result.is_success(), binding_result.errors
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/{doc_id}/{DOCUMENT_BINDING_MAP_FILENAME}")
    )

    generation_result = run_value_generation(
        template, binding_map, doc_id, manifest, binding_map_reference=str(pdf_path), seed=7
    )
    assert generation_result.is_success(), generation_result.errors
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/{doc_id}/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )

    rendering_result = run_document_rendering(
        template, binding_map, values, doc_id, manifest, synthetic_values_reference=str(pdf_path)
    )
    assert rendering_result.is_success(), rendering_result.errors
    ground_truth = load_document_ground_truth(manifest.output_path(f"documents/{doc_id}/{GROUND_TRUTH_FILENAME}"))
    rendered_pdf_path = manifest.output_path(f"documents/{doc_id}/{RENDERED_PDF_FILENAME}")
    assert rendered_pdf_path.is_file()

    validation_result = run_document_validation(
        rendered_pdf_path, ground_truth, doc_id, manifest, ground_truth_reference=str(pdf_path)
    )
    assert validation_result.is_success(), validation_result.errors
    assert validation_result.metrics["overflow_failure_count"] == 0

    deidentification_result = run_document_deidentification(template, doc_id, manifest, template_reference=str(pdf_path))
    assert deidentification_result.is_success(), deidentification_result.errors
    redacted_pdf_path = manifest.output_path(f"documents/{doc_id}/{REDACTED_PDF_FILENAME}")
    assert redacted_pdf_path.is_file()

    # --- Structural equivalence: the twin must be a plausible reconstruction,
    # not degenerate (0 pages, 0 regions). ---
    assert ground_truth["page_count"] >= 1
    assert len(ground_truth["regions"]) == sum(len(page["regions"]) for page in template["pages"])

    # --- PII leakage: every true-PII finding (email/phone/ssn/credit_card/
    # street_address/iban/routing_number) detected in the source must never
    # appear verbatim in the rendered twin or the redacted PDF. ---
    source_text = PDFDocumentAdapter().extract(pdf_path)["text"]
    pii_values = {source_text[f["start"] : f["end"]] for f in profile["pii_findings"]}
    pii_values = {v for v in pii_values if v}

    rendered_text = PDFDocumentAdapter().extract(rendered_pdf_path)["text"]
    redacted_text = PDFDocumentAdapter().extract(redacted_pdf_path)["text"]

    leaked_in_twin = pii_values & _substrings_present(rendered_text, pii_values)
    leaked_in_redacted = pii_values & _substrings_present(redacted_text, pii_values)
    assert not leaked_in_twin, f"Real PII leaked into rendered twin for {filename}: {leaked_in_twin}"
    assert not leaked_in_redacted, f"Real PII leaked into redacted PDF for {filename}: {leaked_in_redacted}"


def test_local_healthcare_pdf_redacts_phi_and_obeys_transfer_gate(manifest: RunManifest, tmp_path: Path) -> None:
    assert HEALTHCARE_SUMMARY_PDF.exists(), f"local PDF fixture is missing: {HEALTHCARE_SUMMARY_PDF}"
    doc_id = "doc1"
    metadata_dir = tmp_path / "metadata"

    profiling_result = run_document_profiling(PDFDocumentAdapter(), HEALTHCARE_SUMMARY_PDF, doc_id, manifest, metadata_dir)
    assert profiling_result.is_success(), profiling_result.errors
    profile = load_document_profile(manifest.output_path(f"documents/{doc_id}/{DOCUMENT_PROFILE_FILENAME}"))
    source_text = PDFDocumentAdapter().extract(HEALTHCARE_SUMMARY_PDF)["text"]

    structured_pii = {source_text[f["start"] : f["end"]] for f in profile["pii_findings"]}
    assert "567-45-5412" in structured_pii
    assert any("(402) 738-5912" in value for value in structured_pii)

    template_result = run_template_compilation(
        HEALTHCARE_SUMMARY_PDF, doc_id, manifest, extraction_method=profile["extraction_method"]
    )
    assert template_result.is_success(), template_result.errors
    template = load_document_template(manifest.output_path(f"documents/{doc_id}/{DOCUMENT_TEMPLATE_FILENAME}"))

    binding_result = run_semantic_binding(template, doc_id, manifest, template_reference=str(HEALTHCARE_SUMMARY_PDF))
    assert binding_result.is_success(), binding_result.errors
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/{doc_id}/{DOCUMENT_BINDING_MAP_FILENAME}")
    )

    generation_result = run_value_generation(
        template, binding_map, doc_id, manifest, binding_map_reference=str(HEALTHCARE_SUMMARY_PDF), seed=7
    )
    assert generation_result.is_success(), generation_result.errors
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/{doc_id}/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )

    rendering_result = run_document_rendering(
        template, binding_map, values, doc_id, manifest, synthetic_values_reference=str(HEALTHCARE_SUMMARY_PDF)
    )
    assert rendering_result.is_success(), rendering_result.errors
    ground_truth = load_document_ground_truth(manifest.output_path(f"documents/{doc_id}/{GROUND_TRUTH_FILENAME}"))
    rendered_pdf_path = manifest.output_path(f"documents/{doc_id}/{RENDERED_PDF_FILENAME}")

    with pytest.raises(TransferBlockedError, match="validation has not run"):
        TransferService().downloadable_file(
            workflow="pdf_twin",
            output_id="synthetic_twin.pdf",
            validation_report=None,
            path=rendered_pdf_path,
        )

    validation_result = run_document_validation(
        rendered_pdf_path, ground_truth, doc_id, manifest, ground_truth_reference=str(HEALTHCARE_SUMMARY_PDF)
    )
    assert validation_result.is_success(), validation_result.errors
    validation_report = {"hard_checks_passed": True, "report": validation_result.metrics}
    transfer = TransferService().downloadable_file(
        workflow="pdf_twin",
        output_id="synthetic_twin.pdf",
        validation_report=validation_report,
        path=rendered_pdf_path,
    )
    assert transfer.approval.validation_status == "PASS"

    deidentification_result = run_document_deidentification(
        template, doc_id, manifest, template_reference=str(HEALTHCARE_SUMMARY_PDF)
    )
    assert deidentification_result.is_success(), deidentification_result.errors
    redacted_pdf_path = manifest.output_path(f"documents/{doc_id}/{REDACTED_PDF_FILENAME}")
    redacted_text = PDFDocumentAdapter().extract(redacted_pdf_path)["text"]

    for sensitive_value in ["Kimberly Lawrence", "24/05/1977", "567-45-5412"]:
        assert sensitive_value in source_text
        assert sensitive_value not in redacted_text


def _substrings_present(haystack: str, candidates: set[str]) -> set[str]:
    return {value for value in candidates if value in haystack}
