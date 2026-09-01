"""Contract: run_document_profiling() always writes the same top-level
shape, and every PII/entity finding shares the same shape, regardless of
document content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.pdf_adapter import PDFDocumentAdapter
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.contract

REQUIRED_TOP_LEVEL_KEYS = {
    "doc_id",
    "source_reference",
    "source_format",
    "profiled_at",
    "text_hash",
    "file_size_bytes",
    "pages",
    "extraction_warnings",
    "statistics",
    "language",
    "pii_findings",
    "entities",
    "classification",
    "archetype",
    "duplicate",
    "extraction_method",
    "pdf_classification",
    "ocr",
}
REQUIRED_PII_FINDING_KEYS = {"type", "start", "end", "confidence", "redaction_preview"}
REQUIRED_ENTITY_FINDING_KEYS = {"type", "start", "end", "confidence", "redaction_preview", "evidence"}

FIXTURES = {
    "with_pii_and_entities": [
        "Contact Ada Lovelace at ada@example.com on 2024-01-01 regarding Acme Corp."
    ],
    "plain_text": ["The quick brown fox jumps over the lazy dog."],
    "blank_page": [""],
    "multi_page": ["First page.", "Second page."],
}


@pytest.mark.parametrize("pages", FIXTURES.values(), ids=FIXTURES.keys())
def test_document_profile_shape_is_invariant(pages: list[str], tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", pages)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    result = run_document_profiling(
        PDFDocumentAdapter(), pdf_path, "doc1", manifest, tmp_path / "metadata"
    )
    assert result.is_success()

    profile = load_document_profile(manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}"))
    assert set(profile) == REQUIRED_TOP_LEVEL_KEYS
    assert "text" not in profile  # raw extracted text must never be persisted

    for finding in profile["pii_findings"]:
        assert set(finding) == REQUIRED_PII_FINDING_KEYS
    for finding in profile["entities"]:
        assert set(finding) == REQUIRED_ENTITY_FINDING_KEYS
