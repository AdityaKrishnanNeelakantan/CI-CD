from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.render_service import (
    DocumentGroundTruthLoadError,
    load_document_ground_truth,
    run_document_rendering,
)
from synth_platform.engine.documents.pdf.validation_service import (
    DocumentValidationReportLoadError,
    load_document_validation_report,
    run_document_validation,
)

pytestmark = pytest.mark.negative

_EMPTY_TEMPLATE = {"page_count": 1, "pages": [{"page_number": 1, "regions": []}]}
_EMPTY_BINDING_MAP = {"bindings": []}
_EMPTY_VALUES = {"fields": {}, "tables": {}, "inline_spans": {}}


def test_rendering_never_overwrites_existing_output(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_document_rendering(
        _EMPTY_TEMPLATE, _EMPTY_BINDING_MAP, _EMPTY_VALUES, "doc1", manifest, synthetic_values_reference="src.pdf"
    )
    with pytest.raises(RuntimeError):
        run_document_rendering(
            _EMPTY_TEMPLATE, _EMPTY_BINDING_MAP, _EMPTY_VALUES, "doc1", manifest, synthetic_values_reference="src.pdf"
        )


def test_validation_never_overwrites_existing_output(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    render_result = run_document_rendering(
        _EMPTY_TEMPLATE, _EMPTY_BINDING_MAP, _EMPTY_VALUES, "doc1", manifest, synthetic_values_reference="src.pdf"
    )
    pdf_path, ground_truth_path = render_result.output_references
    ground_truth = json.loads(Path(ground_truth_path).read_text())

    run_document_validation(pdf_path, ground_truth, "doc1", manifest, ground_truth_reference="src.pdf")
    with pytest.raises(RuntimeError):
        run_document_validation(pdf_path, ground_truth, "doc1", manifest, ground_truth_reference="src.pdf")


def test_validation_fails_closed_on_missing_pdf(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    ground_truth = {"doc_id": "doc1", "regions": []}
    result = run_document_validation(
        tmp_path / "does_not_exist.pdf", ground_truth, "doc1", manifest, ground_truth_reference="src.pdf"
    )
    assert result.status == "failed"
    assert result.errors


def test_load_document_ground_truth_rejects_missing_file(tmp_path: Path):
    with pytest.raises(DocumentGroundTruthLoadError):
        load_document_ground_truth(tmp_path / "does_not_exist.json")


def test_load_document_ground_truth_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "document_ground_truth.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DocumentGroundTruthLoadError):
        load_document_ground_truth(path)


def test_load_document_ground_truth_rejects_missing_top_level_keys(tmp_path: Path):
    path = tmp_path / "document_ground_truth.json"
    path.write_text(json.dumps({"doc_id": "x"}), encoding="utf-8")
    with pytest.raises(DocumentGroundTruthLoadError):
        load_document_ground_truth(path)


def test_load_document_validation_report_rejects_missing_file(tmp_path: Path):
    with pytest.raises(DocumentValidationReportLoadError):
        load_document_validation_report(tmp_path / "does_not_exist.json")


def test_load_document_validation_report_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "document_validation_report.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DocumentValidationReportLoadError):
        load_document_validation_report(path)


def test_load_document_validation_report_rejects_missing_top_level_keys(tmp_path: Path):
    path = tmp_path / "document_validation_report.json"
    path.write_text(json.dumps({"doc_id": "x"}), encoding="utf-8")
    with pytest.raises(DocumentValidationReportLoadError):
        load_document_validation_report(path)
