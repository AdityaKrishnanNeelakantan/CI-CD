"""Contract: run_document_deidentification() always writes the same
top-level shape, and every region entry shares the common shape needed
for src/documents/render_validator.py-style consumers - same pattern as
tests/contract/test_document_ground_truth_contract.py, but for Mode 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.deidentification_service import (
    DEIDENTIFICATION_REPORT_FILENAME,
    load_document_deidentification_report,
    run_document_deidentification,
)
from synth_platform.engine.documents.pdf.layout_engine import analyze_layout
from synth_platform.engine.documents.pdf.native_layout import extract_positioned_spans
from synth_platform.engine.documents.pdf.preflight import run_pdf_preflight
from synth_platform.engine.documents.pdf.template_compiler import compile_document_template
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.contract

REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "redacted_at", "page_count", "regions"}


@pytest.mark.parametrize(
    "pages",
    [
        ["Account Number: 8823471\nTotal Due: 401.50"],
        ["Beginning balance 69.96\nEnding balance 5340.43\nFees 12.00"],
    ],
    ids=["fields", "table_or_paragraph"],
)
def test_document_deidentification_report_shape_is_invariant(pages: list[str], tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", pages)
    spans = extract_positioned_spans(pdf_path)
    preflight = run_pdf_preflight(pdf_path)
    layout = analyze_layout(spans, preflight["page_count"])
    template = compile_document_template(layout)

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_document_deidentification(template, "doc1", manifest, template_reference=str(pdf_path))
    assert result.is_success()

    report = load_document_deidentification_report(
        manifest.output_path(f"documents/doc1/{DEIDENTIFICATION_REPORT_FILENAME}")
    )
    assert set(report) == REQUIRED_TOP_LEVEL_KEYS

    for region in report["regions"]:
        assert "region_id" in region
        assert "region_type" in region
        if region["region_type"] == "table":
            assert "rows" in region
            assert "column_widths" in region
