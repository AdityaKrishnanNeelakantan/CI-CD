"""Contract: run_semantic_binding() always writes the same top-level
shape, and every binding entry (field or table) shares a common shape.
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
from synth_platform.engine.documents.pdf.layout_engine import analyze_layout
from synth_platform.engine.documents.pdf.native_layout import extract_positioned_spans
from synth_platform.engine.documents.pdf.preflight import run_pdf_preflight
from synth_platform.engine.documents.pdf.template_compiler import compile_document_template
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.contract

REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "bound_at", "page_count", "bindings"}
REQUIRED_FIELD_BINDING_KEYS = {
    "region_id",
    "binding_type",
    "label",
    "semantic_role",
    "semantic_status",
    "semantic_confidence",
    "generator_strategy",
}
REQUIRED_TABLE_BINDING_KEYS = {"region_id", "binding_type", "row_count", "column_count", "columns"}
REQUIRED_INLINE_SPANS_BINDING_KEYS = {"region_id", "binding_type", "spans"}


def _compile_template(pdf_path):
    spans = extract_positioned_spans(pdf_path)
    preflight = run_pdf_preflight(pdf_path)
    layout = analyze_layout(spans, preflight["page_count"])
    return compile_document_template(layout)


@pytest.mark.parametrize(
    "pages",
    [
        ["Account Number: 8823471\nTotal Due: 401.50"],
        ["Beginning balance 69.96\nEnding balance 5340.43\nFees 12.00"],
        [""],
    ],
    ids=["fields", "table", "blank"],
)
def test_document_binding_map_shape_is_invariant(pages: list[str], tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", pages)
    template = _compile_template(pdf_path)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    result = run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
    assert result.is_success()

    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}")
    )
    assert set(binding_map) == REQUIRED_TOP_LEVEL_KEYS

    for binding in binding_map["bindings"]:
        if binding["binding_type"] == "field":
            assert REQUIRED_FIELD_BINDING_KEYS <= set(binding)
        elif binding["binding_type"] == "table":
            assert REQUIRED_TABLE_BINDING_KEYS <= set(binding)
            for column in binding["columns"]:
                assert {"column_index", "semantic_role", "generator_strategy"} <= set(column)
        else:
            assert binding["binding_type"] == "inline_spans"
            assert REQUIRED_INLINE_SPANS_BINDING_KEYS <= set(binding)
            for span in binding["spans"]:
                assert {"start", "end", "semantic_role", "generator_strategy"} <= set(span)
