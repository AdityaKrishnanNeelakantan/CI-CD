"""Contract: run_template_compilation() always writes the same top-level
shape, and every region shares a common shape regardless of its type, so
a future renderer/generator stage can rely on document_template.json's
structure without special-casing per document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.template_service import (
    DOCUMENT_TEMPLATE_FILENAME,
    load_document_template,
    run_template_compilation,
)
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.contract

REQUIRED_TOP_LEVEL_KEYS = {
    "doc_id",
    "source_reference",
    "compiled_at",
    "extraction_method",
    "page_count",
    "pages",
}
REQUIRED_REGION_KEYS = {"region_id", "region_type", "bbox", "reading_order_index"}
REQUIRED_BBOX_KEYS = {"x0", "y0", "x1", "y1"}


@pytest.mark.parametrize(
    "pages",
    [
        ["Account Number: 8823471\nTotal Due: 401.50"],
        ["Dear Dr. Grace Hopper,\nPlease reply by 2024-06-01.\nSincerely, the team."],
        [""],
    ],
    ids=["field_heavy", "letter", "blank_page"],
)
def test_document_template_shape_is_invariant(pages: list[str], tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", pages)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    result = run_template_compilation(pdf_path, "doc1", manifest, extraction_method="native")
    assert result.is_success()

    template = load_document_template(manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}"))
    assert set(template) == REQUIRED_TOP_LEVEL_KEYS
    assert "text" not in template

    for page in template["pages"]:
        for region in page["regions"]:
            assert REQUIRED_REGION_KEYS <= set(region)
            assert set(region["bbox"]) == REQUIRED_BBOX_KEYS
            if region["region_type"] == "field":
                assert "label" in region and "masked_preview" in region
            elif region["region_type"] == "table":
                assert "rows" in region and "row_count" in region
            else:
                assert "text" in region
