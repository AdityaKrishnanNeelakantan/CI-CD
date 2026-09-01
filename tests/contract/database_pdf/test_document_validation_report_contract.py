"""Contract: run_document_validation() always writes the same top-level
shape.
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
from synth_platform.engine.documents.pdf.layout_engine import analyze_layout
from synth_platform.engine.documents.pdf.native_layout import extract_positioned_spans
from synth_platform.engine.documents.pdf.preflight import run_pdf_preflight
from synth_platform.engine.documents.pdf.render_service import (
    GROUND_TRUTH_FILENAME,
    RENDERED_PDF_FILENAME,
    load_document_ground_truth,
    run_document_rendering,
)
from synth_platform.engine.documents.pdf.template_compiler import compile_document_template
from synth_platform.engine.documents.pdf.validation_service import (
    VALIDATION_REPORT_FILENAME,
    load_document_validation_report,
    run_document_validation,
)
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.contract

REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "validated_at", "hard_checks_passed", "report"}
REQUIRED_REPORT_KEYS = {
    "field_accuracy",
    "total_fields",
    "matched_fields",
    "unmatched_fields",
    "overflow_failure_count",
    "overflow_failures",
    "extraction_warnings",
    "hard_checks_passed",
}


def test_document_validation_report_shape(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Account Number: 8823471\nTotal Due: 401.50"])

    spans = extract_positioned_spans(pdf_path)
    preflight = run_pdf_preflight(pdf_path)
    layout = analyze_layout(spans, preflight["page_count"])
    template = compile_document_template(layout)

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}")
    )
    run_value_generation(template, binding_map, "doc1", manifest, binding_map_reference=str(pdf_path), seed=1)
    values = load_document_synthetic_values(
        manifest.output_path(f"documents/doc1/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )
    run_document_rendering(template, binding_map, values, "doc1", manifest, synthetic_values_reference=str(pdf_path))
    ground_truth = load_document_ground_truth(manifest.output_path(f"documents/doc1/{GROUND_TRUTH_FILENAME}"))
    rendered_pdf_path = manifest.output_path(f"documents/doc1/{RENDERED_PDF_FILENAME}")

    result = run_document_validation(rendered_pdf_path, ground_truth, "doc1", manifest, ground_truth_reference=str(pdf_path))
    assert result.is_success()

    report = load_document_validation_report(
        manifest.output_path(f"documents/doc1/{VALIDATION_REPORT_FILENAME}")
    )
    assert set(report) == REQUIRED_TOP_LEVEL_KEYS
    assert REQUIRED_REPORT_KEYS <= set(report["report"])
