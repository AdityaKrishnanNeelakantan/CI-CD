from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.template_service import (
    DOCUMENT_TEMPLATE_FILENAME,
    DocumentTemplateLoadError,
    load_document_template,
    run_template_compilation,
)
from tests.fixtures.pdf_factory import make_pdf

pytestmark = pytest.mark.negative


def test_template_compilation_fails_closed_on_missing_file(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_template_compilation(
        tmp_path / "missing.pdf", "doc1", manifest, extraction_method="native"
    )
    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path(f"documents/doc1/{DOCUMENT_TEMPLATE_FILENAME}").exists()


def test_template_compilation_fails_closed_on_corrupt_pdf(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4 not a real structure" + b"\x00" * 50)
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_template_compilation(corrupt_path, "doc1", manifest, extraction_method="native")
    assert result.status == "failed"
    assert result.errors


def test_template_compilation_never_overwrites_existing_output(tmp_path: Path):
    pdf_path = make_pdf(tmp_path / "doc.pdf", ["Some content."])
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_template_compilation(pdf_path, "doc1", manifest, extraction_method="native")
    with pytest.raises(RuntimeError):
        run_template_compilation(pdf_path, "doc1", manifest, extraction_method="native")


def test_load_document_template_rejects_missing_file(tmp_path: Path):
    with pytest.raises(DocumentTemplateLoadError):
        load_document_template(tmp_path / "does_not_exist.json")


def test_load_document_template_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "document_template.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DocumentTemplateLoadError):
        load_document_template(path)


def test_load_document_template_rejects_missing_top_level_keys(tmp_path: Path):
    path = tmp_path / "document_template.json"
    path.write_text(json.dumps({"doc_id": "x"}), encoding="utf-8")
    with pytest.raises(DocumentTemplateLoadError):
        load_document_template(path)
