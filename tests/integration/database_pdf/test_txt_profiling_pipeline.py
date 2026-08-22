"""Integration: run_document_profiling() works unmodified with a
non-PDF DocumentAdapter (TXTDocumentAdapter) - concrete proof that the
plugin architecture (src/documents/base.py, src/documents/extraction_router.py)
needs no redesign to add a second document type, per this project's
existing "any DocumentAdapter that isn't PDFDocumentAdapter is passed
through unchanged" contract in extraction_router.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.service import DOCUMENT_PROFILE_FILENAME, load_document_profile, run_document_profiling
from synth_platform.engine.documents.pdf.txt_adapter import TXTDocumentAdapter

pytestmark = pytest.mark.integration


def test_document_profiling_accepts_a_txt_file_via_the_plugin_seam(tmp_path: Path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Contact grace@example.com about account 8823471.", encoding="utf-8")

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_document_profiling(TXTDocumentAdapter(), txt_path, "doc1", manifest, tmp_path / "metadata")
    assert result.is_success()

    profile = load_document_profile(manifest.output_path(f"documents/doc1/{DOCUMENT_PROFILE_FILENAME}"))
    assert profile["source_format"] == "txt"
    assert profile["extraction_method"] == "native"
    assert profile["pdf_classification"] is None
    assert profile["ocr"] is None
    assert any(f["type"] == "email" for f in profile["pii_findings"])
