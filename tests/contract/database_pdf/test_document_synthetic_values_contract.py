"""Contract: run_value_generation() always writes the same top-level
shape, and every generated field carries a semantic_role alongside its
value, regardless of which generator strategy produced it.
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
from synth_platform.engine.documents.pdf.template_compiler import compile_document_template
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.contract

REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "generated_at", "seed", "fields", "tables", "inline_spans"}
REQUIRED_FIELD_VALUE_KEYS = {"label", "semantic_role"}


def _compile_template(pdf_path):
    spans = extract_positioned_spans(pdf_path)
    preflight = run_pdf_preflight(pdf_path)
    layout = analyze_layout(spans, preflight["page_count"])
    return compile_document_template(layout)


@pytest.mark.parametrize(
    "pages",
    [
        ["Account Number: 8823471\nTotal Due: 401.50\nCustomer Name: Grace Hopper"],
        ["Beginning balance 69.96\nEnding balance 5340.43\nFees 12.00"],
    ],
    ids=["fields", "table"],
)
def test_document_synthetic_values_shape_is_invariant(pages: list[str], tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", pages)
    template = _compile_template(pdf_path)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    run_semantic_binding(template, "doc1", manifest, template_reference=str(pdf_path))
    binding_map = load_document_binding_map(
        manifest.output_path(f"documents/doc1/{DOCUMENT_BINDING_MAP_FILENAME}")
    )

    result = run_value_generation(
        template, binding_map, "doc1", manifest, binding_map_reference=str(pdf_path), seed=42
    )
    assert result.is_success()

    values = load_document_synthetic_values(
        manifest.output_path(f"documents/doc1/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    )
    assert set(values) == REQUIRED_TOP_LEVEL_KEYS
    assert values["seed"] == 42

    for field in values["fields"].values():
        assert REQUIRED_FIELD_VALUE_KEYS <= set(field)

    for rows in values["tables"].values():
        for row in rows:
            for cell in row:
                assert "semantic_role" in cell
