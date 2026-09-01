"""Exact-duplicate detection over normalised document text.

Only exact-hash duplicate detection is implemented as the baseline; near-
duplicate detection (fuzzy similarity) is a documented limitation, not
attempted here. The index is a durable, cross-run lookup (like
metadata/dataset_contract.json) rather than per-run evidence, since
duplicate detection is only meaningful across the whole document corpus,
not within a single run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synth_platform.engine.documents.pdf.errors import DocumentIndexError

INDEX_FILENAME = "document_hash_index.json"


def check_duplicate(text_hash: str, index: dict[str, Any]) -> dict[str, Any]:
    existing = index.get(text_hash)
    return {
        "is_duplicate": existing is not None,
        "duplicate_of": existing["source_reference"] if existing else None,
    }


def load_document_index(metadata_dir: str | Path) -> dict[str, Any]:
    path = Path(metadata_dir) / INDEX_FILENAME
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise DocumentIndexError(f"document hash index is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise DocumentIndexError(f"document hash index must contain a JSON object: {path}")
    return data


def save_document_index(index: dict[str, Any], metadata_dir: str | Path) -> Path:
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / INDEX_FILENAME
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
    except OSError as exc:
        raise DocumentIndexError(f"could not write document hash index: {path}: {exc}") from exc
    return path
