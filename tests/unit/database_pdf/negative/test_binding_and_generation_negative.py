from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.documents.pdf.binding_service import (
    DocumentBindingMapLoadError,
    load_document_binding_map,
    run_semantic_binding,
)
from synth_platform.engine.documents.pdf.generation_service import (
    DocumentSyntheticValuesLoadError,
    load_document_synthetic_values,
    run_value_generation,
)

pytestmark = pytest.mark.negative

_EMPTY_TEMPLATE = {"page_count": 0, "pages": []}
_EMPTY_BINDING_MAP = {"bindings": []}


def test_semantic_binding_never_overwrites_existing_output(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_semantic_binding(_EMPTY_TEMPLATE, "doc1", manifest, template_reference="src.pdf")
    with pytest.raises(RuntimeError):
        run_semantic_binding(_EMPTY_TEMPLATE, "doc1", manifest, template_reference="src.pdf")


def test_value_generation_never_overwrites_existing_output(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_value_generation(
        _EMPTY_TEMPLATE, _EMPTY_BINDING_MAP, "doc1", manifest, binding_map_reference="src.pdf", seed=1
    )
    with pytest.raises(RuntimeError):
        run_value_generation(
            _EMPTY_TEMPLATE, _EMPTY_BINDING_MAP, "doc1", manifest, binding_map_reference="src.pdf", seed=1
        )


def test_load_document_binding_map_rejects_missing_file(tmp_path: Path):
    with pytest.raises(DocumentBindingMapLoadError):
        load_document_binding_map(tmp_path / "does_not_exist.json")


def test_load_document_binding_map_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "document_binding_map.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DocumentBindingMapLoadError):
        load_document_binding_map(path)


def test_load_document_binding_map_rejects_missing_top_level_keys(tmp_path: Path):
    path = tmp_path / "document_binding_map.json"
    path.write_text(json.dumps({"doc_id": "x"}), encoding="utf-8")
    with pytest.raises(DocumentBindingMapLoadError):
        load_document_binding_map(path)


def test_load_document_synthetic_values_rejects_missing_file(tmp_path: Path):
    with pytest.raises(DocumentSyntheticValuesLoadError):
        load_document_synthetic_values(tmp_path / "does_not_exist.json")


def test_load_document_synthetic_values_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "document_synthetic_values.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DocumentSyntheticValuesLoadError):
        load_document_synthetic_values(path)


def test_load_document_synthetic_values_rejects_missing_top_level_keys(tmp_path: Path):
    path = tmp_path / "document_synthetic_values.json"
    path.write_text(json.dumps({"doc_id": "x"}), encoding="utf-8")
    with pytest.raises(DocumentSyntheticValuesLoadError):
        load_document_synthetic_values(path)
