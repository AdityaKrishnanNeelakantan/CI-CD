"""Negative tests for src/documents/deidentification_service.py (Mode 2):
never-overwrite guard, and load_document_deidentification_report's
fail-closed error paths - same shape as
tests/negative/test_render_and_validation_negative.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.deidentification_service import (
    DocumentDeidentificationReportLoadError,
    load_document_deidentification_report,
    run_document_deidentification,
)

pytestmark = pytest.mark.negative

_EMPTY_TEMPLATE = {"page_count": 1, "pages": [{"page_number": 1, "regions": []}]}


def test_deidentification_never_overwrites_existing_output(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_document_deidentification(_EMPTY_TEMPLATE, "doc1", manifest, template_reference="src.pdf")
    with pytest.raises(RuntimeError):
        run_document_deidentification(_EMPTY_TEMPLATE, "doc1", manifest, template_reference="src.pdf")


def test_load_document_deidentification_report_rejects_missing_file(tmp_path: Path):
    with pytest.raises(DocumentDeidentificationReportLoadError):
        load_document_deidentification_report(tmp_path / "does_not_exist.json")


def test_load_document_deidentification_report_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "document_deidentification_report.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DocumentDeidentificationReportLoadError):
        load_document_deidentification_report(path)


def test_load_document_deidentification_report_rejects_missing_top_level_keys(tmp_path: Path):
    path = tmp_path / "document_deidentification_report.json"
    path.write_text(json.dumps({"doc_id": "x"}), encoding="utf-8")
    with pytest.raises(DocumentDeidentificationReportLoadError):
        load_document_deidentification_report(path)
