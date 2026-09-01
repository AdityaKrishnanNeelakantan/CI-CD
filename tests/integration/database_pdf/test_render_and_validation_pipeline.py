"""Integration: the full six-stage PDF track - document_profiling ->
template_compilation -> semantic_binding -> value_generation ->
document_rendering -> document_validation. Proves the rendered twin
round-trips cleanly and the validation report reflects a real pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.binding_service import (
    DOCUMENT_BINDING_MAP_FILENAME,
    load_document_binding_map,
    run_semantic_binding,
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
from synth_platform.engine.documents.pdf.service import run_document_profiling
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


def test_full_pdf_track_renders_and_validates_successfully(tmp_path: Path):
    pdf_path = make_pdf(
        tmp_path / "statement.pdf",
        [
            "Account Number: 8823471\n"
            "Total Due: 401.50\n"
            "Customer Name: Grace Hopper\n"
            "Reference Code: RC-88291\n"
            "Issued Date: 2024-06-01\n"
            "Beginning balance 69.96\n"
            "Ending balance 5340.43\n"
            "Fees 12.00"
        ],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, metadata_dir)
    run_template_compilation(pdf_path, "doc1", manifest, extraction_method="native")
    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))

    run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}")
    )

    run_value_generation(template, binding_map, "doc1", manifest, binding_map_reference=str(pdf_path), seed=42)
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/doc1/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )

    render_result = run_document_rendering(
        template, binding_map, values, "doc1", manifest, synthetic_values_reference=str(pdf_path)
    )
    assert render_result.is_success()

    ground_truth = load_document_ground_truth(manifest.output_path(f"documents/doc1/{GROUND_TRUTH_FILENAME}"))
    rendered_pdf_path = manifest.output_path(f"documents/doc1/{RENDERED_PDF_FILENAME}")
    assert rendered_pdf_path.is_file()

    validation_result = run_document_validation(
        rendered_pdf_path, ground_truth, "doc1", manifest, ground_truth_reference=str(pdf_path)
    )
    assert validation_result.is_success()

    report = load_document_validation_report(
        manifest.output_path(f"documents/doc1/{VALIDATION_REPORT_FILENAME}")
    )
    assert report["hard_checks_passed"] is True
    assert report["report"]["field_accuracy"] == 1.0
    assert report["report"]["overflow_failure_count"] == 0


def test_rendered_twin_never_contains_the_source_documents_real_values(tmp_path: Path):
    source_values = ["8823471", "401.50", "Grace Hopper", "RC-88291", "2024-06-01"]
    pdf_path = make_pdf(
        tmp_path / "statement.pdf",
        [
            "Account Number: 8823471\n"
            "Total Due: 401.50\n"
            "Customer Name: Grace Hopper\n"
            "Reference Code: RC-88291\n"
            "Issued Date: 2024-06-01"
        ],
    )
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, metadata_dir)
    run_template_compilation(pdf_path, "doc1", manifest, extraction_method="native")
    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))
    run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}")
    )
    run_value_generation(template, binding_map, "doc1", manifest, binding_map_reference=str(pdf_path), seed=1)
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/doc1/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )
    run_document_rendering(template, binding_map, values, "doc1", manifest, synthetic_values_reference=str(pdf_path))

    rendered_pdf_path = manifest.output_path(f"documents/doc1/{RENDERED_PDF_FILENAME}")
    extracted = PDFDocumentAdapter().extract(rendered_pdf_path)
    for secret in source_values:
        assert secret not in extracted["text"]
