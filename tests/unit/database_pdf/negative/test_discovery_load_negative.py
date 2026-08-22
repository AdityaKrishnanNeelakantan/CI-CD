"""Negative tests: corrupted or tampered discovery.json must be rejected.

This is the "corrupted artifact is rejected" case for Checkpoint 1's
output. The equivalent for the portable model artifact (Checkpoint 5,
checksum validation) will be added when that artifact format exists -
there is nothing to corrupt yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.service import DiscoveryLoadError, load_discovery

pytestmark = pytest.mark.negative


def test_missing_file_is_rejected(tmp_path: Path):
    with pytest.raises(DiscoveryLoadError):
        load_discovery(tmp_path / "does_not_exist.json")


def test_invalid_json_is_rejected(tmp_path: Path):
    path = tmp_path / "discovery.json"
    path.write_text("{not valid json at all", encoding="utf-8")
    with pytest.raises(DiscoveryLoadError):
        load_discovery(path)


def test_non_object_json_is_rejected(tmp_path: Path):
    path = tmp_path / "discovery.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(DiscoveryLoadError):
        load_discovery(path)


def test_missing_top_level_keys_are_rejected(tmp_path: Path):
    path = tmp_path / "discovery.json"
    path.write_text(json.dumps({"source_type": "sqlite"}), encoding="utf-8")
    with pytest.raises(DiscoveryLoadError):
        load_discovery(path)


def test_table_missing_required_keys_is_rejected(tmp_path: Path):
    path = tmp_path / "discovery.json"
    path.write_text(
        json.dumps(
            {
                "source_type": "sqlite",
                "discovered_at": "2026-01-01T00:00:00Z",
                "source_fingerprint": "sha256:" + "a" * 64,
                "tables": {"customers": {"columns": []}},  # missing primary_key etc.
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DiscoveryLoadError):
        load_discovery(path)


def test_malformed_fingerprint_is_rejected(tmp_path: Path):
    path = tmp_path / "discovery.json"
    path.write_text(
        json.dumps(
            {
                "source_type": "sqlite",
                "discovered_at": "2026-01-01T00:00:00Z",
                "source_fingerprint": "not-a-real-fingerprint",
                "tables": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DiscoveryLoadError):
        load_discovery(path)
