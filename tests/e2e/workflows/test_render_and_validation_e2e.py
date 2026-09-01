"""End-to-end: the complete PDF track - document_profiling ->
template_compilation -> semantic_binding -> value_generation ->
document_rendering -> document_validation - re-read entirely from disk
afterward, same proof pattern as the other tests/e2e/test_*_e2e.py files.
This is the full pipeline the master spec describes for Deliverable 2.
"""

from __future__ import annotations

import json
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
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
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

pytestmark = pytest.mark.e2e

_A4_WIDTH_PT = 595.28
_A4_HEIGHT_PT = 841.89
_LETTER_WIDTH_PT = 612.0
_LETTER_HEIGHT_PT = 792.0


def test_complete_pdf_track_is_fully_traceable_end_to_end(tmp_path: Path):
    metadata_dir = tmp_path / "metadata"
    source_secrets = ["8823471", "401.50", "Grace Hopper", "RC-88291", "2024-06-01", "69.96", "5340.43", "12.00"]

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

    run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, metadata_dir)
    profile = load_document_profile(manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}"))

    run_template_compilation(pdf_path, "doc1", manifest, extraction_method=profile["extraction_method"])
    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))

    run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}")
    )

    run_value_generation(template, binding_map, "doc1", manifest, binding_map_reference=str(pdf_path), seed=7)
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/doc1/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )

    run_document_rendering(template, binding_map, values, "doc1", manifest, synthetic_values_reference=str(pdf_path))
    ground_truth = load_document_ground_truth(manifest.output_path(f"documents/doc1/{GROUND_TRUTH_FILENAME}"))
    rendered_pdf_path = manifest.output_path(f"documents/doc1/{RENDERED_PDF_FILENAME}")

    run_document_validation(rendered_pdf_path, ground_truth, "doc1", manifest, ground_truth_reference=str(pdf_path))

    # Simulate a fresh process: re-read every artifact from disk only.
    manifest_on_disk = json.loads((manifest.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    stage_names = [s["stage_name"] for s in manifest_on_disk["stages"]]
    assert stage_names == [
        "document_profiling",
        "template_compilation",
        "semantic_binding",
        "value_generation",
        "document_rendering",
        "document_validation",
    ]
    assert all(s["status"] == "success" for s in manifest_on_disk["stages"])

    validation_report = load_document_validation_report(
        manifest.output_path(f"documents/doc1/{VALIDATION_REPORT_FILENAME}")
    )
    assert validation_report["hard_checks_passed"] is True

    # The definitive privacy proof: the rendered PDF, its ground truth,
    # and the template must never contain any of the source's real values.
    extracted_from_twin = PDFDocumentAdapter().extract(rendered_pdf_path)
    serialized_ground_truth = json.dumps(ground_truth)
    serialized_template = json.dumps(template)
    for secret in source_secrets:
        assert secret not in extracted_from_twin["text"], f"{secret!r} leaked into the rendered twin PDF"
        assert secret not in serialized_ground_truth, f"{secret!r} leaked into document_ground_truth.json"
        assert secret not in serialized_template, f"{secret!r} leaked into document_template.json"


def test_rendered_twin_page_size_matches_the_source_not_a_hardcoded_default(tmp_path: Path):
    """Regression: the renderer used to hardcode FPDF(format="Letter")
    regardless of the source PDF's actual page size - invisible to every
    other test in this file because make_pdf() itself defaults to A4
    (fpdf2's own default), and none of them asserted on page size. A
    source page size must be propagated end to end: preflight ->
    layout_engine -> template_compiler -> pdf_renderer.
    """
    metadata_dir = tmp_path / "metadata"
    pdf_path = make_pdf(tmp_path / "a4_statement.pdf", ["Account Number: 1234567\nCustomer Name: Ada Lovelace"])

    import pypdf

    source_mediabox = pypdf.PdfReader(str(pdf_path)).pages[0].mediabox
    assert abs(float(source_mediabox.width) - _A4_WIDTH_PT) < 1, "test fixture assumption changed: expected A4"

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_document_profiling(PDFDocumentAdapter(), pdf_path, "doc1", manifest, metadata_dir)
    profile = load_document_profile(manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}"))

    run_template_compilation(pdf_path, "doc1", manifest, extraction_method=profile["extraction_method"])
    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))
    assert abs(template["pages"][0]["width_pt"] - _A4_WIDTH_PT) < 1
    assert abs(template["pages"][0]["height_pt"] - _A4_HEIGHT_PT) < 1

    run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
    binding_map = load_document_binding_map(manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}"))

    run_value_generation(template, binding_map, "doc1", manifest, binding_map_reference=str(pdf_path), seed=3)
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/doc1/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )

    run_document_rendering(template, binding_map, values, "doc1", manifest, synthetic_values_reference=str(pdf_path))
    rendered_pdf_path = manifest.output_path(f"documents/doc1/{RENDERED_PDF_FILENAME}")

    rendered_mediabox = pypdf.PdfReader(str(rendered_pdf_path)).pages[0].mediabox
    is_a4 = abs(float(rendered_mediabox.width) - _A4_WIDTH_PT) < 2 and abs(
        float(rendered_mediabox.height) - _A4_HEIGHT_PT
    ) < 2
    is_letter = abs(float(rendered_mediabox.width) - _LETTER_WIDTH_PT) < 2 and abs(
        float(rendered_mediabox.height) - _LETTER_HEIGHT_PT
    ) < 2
    assert is_a4, f"rendered twin page size {rendered_mediabox.width}x{rendered_mediabox.height}pt is not A4"
    assert not is_letter, "rendered twin fell back to hardcoded Letter size instead of matching the source"
