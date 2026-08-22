from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.documents.pdf.duplicates import INDEX_FILENAME, check_duplicate, load_document_index, save_document_index
from synth_platform.engine.documents.pdf.errors import DocumentIndexError

pytestmark = pytest.mark.unit


def test_check_duplicate_returns_false_for_unknown_hash():
    result = check_duplicate("sha256:abc", index={})
    assert result["is_duplicate"] is False
    assert result["duplicate_of"] is None


def test_check_duplicate_returns_true_for_known_hash():
    index = {"sha256:abc": {"source_reference": "doc1.pdf"}}
    result = check_duplicate("sha256:abc", index)
    assert result["is_duplicate"] is True
    assert result["duplicate_of"] == "doc1.pdf"


def test_load_document_index_returns_empty_dict_when_missing(tmp_path: Path):
    assert load_document_index(tmp_path) == {}


def test_save_and_load_round_trip(tmp_path: Path):
    index = {"sha256:abc": {"source_reference": "doc1.pdf"}}
    save_document_index(index, tmp_path)
    loaded = load_document_index(tmp_path)
    assert loaded == index


def test_load_rejects_malformed_json_content(tmp_path: Path):
    index_path = tmp_path / INDEX_FILENAME
    index_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DocumentIndexError):
        load_document_index(tmp_path)


def test_load_rejects_non_object_json_content(tmp_path: Path):
    index_path = tmp_path / INDEX_FILENAME
    index_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(DocumentIndexError):
        load_document_index(tmp_path)


def test_save_wraps_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import synth_platform.engine.documents.pdf.duplicates as duplicates_module

    def _raising_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(duplicates_module.json, "dump", _raising_dump)
    with pytest.raises(DocumentIndexError):
        save_document_index({"sha256:abc": {"source_reference": "doc1.pdf"}}, tmp_path)
