"""Contract: run_document_rendering() always writes the same top-level
shape, and every region entry (field/heading/paragraph/table) shares the
common shape needed for src/documents/render_validator.py's round trip.
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
    load_document_ground_truth,
    run_document_rendering,
)
from synth_platform.engine.documents.pdf.template_compiler import compile_document_template
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.contract

REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "rendered_at", "page_count", "regions"}


def _build_chain(pdf_path, tmp_path):
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
    return manifest, template, binding_map, values


@pytest.mark.parametrize(
    "pages",
    [
        ["Account Number: 8823471\nTotal Due: 401.50"],
        ["Beginning balance 69.96\nEnding balance 5340.43\nFees 12.00"],
    ],
    ids=["fields", "table_or_paragraph"],
)
def test_document_ground_truth_shape_is_invariant(pages: list[str], tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", pages)
    manifest, template, binding_map, values = _build_chain(pdf_path, tmp_path)

    result = run_document_rendering(template, binding_map, values, "doc1", manifest, synthetic_values_reference=str(pdf_path))
    assert result.is_success()

    ground_truth = load_document_ground_truth(manifest.output_path(f"documents/doc1/{GROUND_TRUTH_FILENAME}"))
    assert set(ground_truth) == REQUIRED_TOP_LEVEL_KEYS

    for region in ground_truth["regions"]:
        assert "region_id" in region
        assert "region_type" in region
        if region["region_type"] == "table":
            assert "rows" in region
            assert "column_widths" in region
            for row in region["rows"]:
                assert {"row_index", "page", "bbox", "rendered_cells"} <= set(row)
        else:
            assert {"page", "bbox", "rendered_text"} <= set(region)
