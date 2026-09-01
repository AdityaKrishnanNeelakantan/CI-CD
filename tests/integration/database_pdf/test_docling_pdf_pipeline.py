"""Full PDF Twin backend chain through the Docling extraction path."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.binding_service import (
    DOCUMENT_BINDING_MAP_FILENAME,
    load_document_binding_map,
    run_semantic_binding,
)
from synth_platform.engine.documents.pdf.deidentification_service import (
    DEIDENTIFICATION_REPORT_FILENAME,
    REDACTED_PDF_FILENAME,
    load_document_deidentification_report,
    run_document_deidentification,
)
from synth_platform.engine.documents.pdf.docling_engine import is_docling_available
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
from synth_platform.engine.documents.pdf.service import (
    DOCUMENT_PROFILE_FILENAME,
    load_document_profile,
    run_document_profiling,
)
from synth_platform.engine.documents.pdf.template_service import (
    DOCUMENT_TEMPLATE_FILENAME,
    load_document_template,
    run_template_compilation,
)
from synth_platform.engine.documents.pdf.validation_service import (
    VALIDATION_REPORT_FILENAME,
    load_document_validation_report,
    run_document_validation,
)
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not is_docling_available(), reason="Docling not installed on this machine")
def test_docling_path_completes_profile_to_validation(tmp_path: Path):
    pdf_path = make_pdf(
        tmp_path / "docling_statement.pdf",
        [
            "Account Number: 8823471\n"
            "Total Due: 401.50\n"
            "Customer Name: Grace Hopper\n"
            "Reference Code: RC-88291\n"
            "Issued Date: 2024-06-01"
        ],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    doc_id = "doc1"

    profile_result = run_document_profiling(
        PDFDocumentAdapter(),
        pdf_path,
        doc_id,
        manifest,
        tmp_path / "metadata",
        preferred_extraction_method="docling",
    )
    if not profile_result.is_success() and any("Failed to download" in err for err in profile_result.errors):
        pytest.skip("Docling/RapidOCR model assets are not available locally")
    assert profile_result.is_success(), profile_result.errors
    profile = load_document_profile(
        manifest.output_path(f"documents/{doc_id}/{DOCUMENT_PROFILE_FILENAME}")
    )
    assert profile["extraction_method"] == "docling"

    template_result = run_template_compilation(
        pdf_path,
        doc_id,
        manifest,
        extraction_method=profile["extraction_method"],
    )
    assert template_result.is_success(), template_result.errors
    assert template_result.metrics["field_count"] >= 1
    template = load_document_template(
        manifest.output_path(f"documents/{doc_id}/{DOCUMENT_TEMPLATE_FILENAME}")
    )
    assert template["extraction_method"] == "docling"

    binding_result = run_semantic_binding(template, doc_id, manifest, template_reference=str(pdf_path))
    assert binding_result.is_success(), binding_result.errors
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/{doc_id}/{DOCUMENT_BINDING_MAP_FILENAME}")
    )

    generation_result = run_value_generation(
        template,
        binding_map,
        doc_id,
        manifest,
        binding_map_reference=str(pdf_path),
        seed=42,
    )
    assert generation_result.is_success(), generation_result.errors
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/{doc_id}/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )

    rendering_result = run_document_rendering(
        template,
        binding_map,
        values,
        doc_id,
        manifest,
        synthetic_values_reference=str(pdf_path),
    )
    assert rendering_result.is_success(), rendering_result.errors
    rendered_pdf = manifest.output_path(f"documents/{doc_id}/{RENDERED_PDF_FILENAME}")
    ground_truth = load_document_ground_truth(
        manifest.output_path(f"documents/{doc_id}/{GROUND_TRUTH_FILENAME}")
    )

    validation_result = run_document_validation(
        rendered_pdf,
        ground_truth,
        doc_id,
        manifest,
        ground_truth_reference=str(pdf_path),
    )
    assert validation_result.is_success(), validation_result.errors
    report = load_document_validation_report(
        manifest.output_path(f"documents/{doc_id}/{VALIDATION_REPORT_FILENAME}")
    )
    assert report["hard_checks_passed"] is True
    assert report["report"]["field_accuracy"] == 1.0
    assert report["report"]["overflow_failure_count"] == 0

    deidentification_result = run_document_deidentification(
        template, doc_id, manifest, template_reference=str(pdf_path)
    )
    assert deidentification_result.is_success(), deidentification_result.errors
    assert manifest.output_path(f"documents/{doc_id}/{REDACTED_PDF_FILENAME}").is_file()
    deidentification_report = load_document_deidentification_report(
        manifest.output_path(f"documents/{doc_id}/{DEIDENTIFICATION_REPORT_FILENAME}")
    )
    assert deidentification_report["doc_id"] == doc_id
