"""Integration: SQLite adapter -> DiscoveryService -> RunManifest -> load_discovery.

Verifies these components produce and consume a consistent discovery.json
together, not just that each one is individually correct. Negative-path
behaviour (missing source, overwrite protection) lives in tests/negative/ -
this file stays focused on the happy path across components.

As later checkpoints land (profiler reading discovery.json, semantic
inference reading profile.json, ...) this is where the integration chain
grows one link at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import DISCOVERY_FILENAME, load_discovery, run_discovery

pytestmark = pytest.mark.integration


def test_run_discovery_writes_discovery_json_and_updates_manifest(
    temp_sqlite_db: Path, tmp_path: Path
):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    result = run_discovery(adapter, manifest, config_path="config/project.yaml")

    assert result.is_success()
    output_path = manifest.output_path(DISCOVERY_FILENAME)
    assert output_path.exists()

    with output_path.open() as f:
        discovery_data = json.load(f)

    # No table or column name is hard-coded anywhere in src/ - the discovery
    # output must be entirely a reflection of the fixture database's schema.
    assert set(discovery_data["tables"]) == set(adapter.list_tables())
    assert discovery_data["tables"]["orders"]["primary_key"] == adapter.get_primary_keys("orders")
    assert discovery_data["source_fingerprint"].startswith("sha256:")

    assert manifest.source_fingerprint == discovery_data["source_fingerprint"]

    manifest_path = manifest.run_dir / "run_manifest.json"
    with manifest_path.open() as f:
        manifest_data = json.load(f)
    assert len(manifest_data["stages"]) == 1
    assert manifest_data["stages"][0]["stage_name"] == "discovery"
    assert manifest_data["stages"][0]["status"] == "success"


def test_each_run_gets_its_own_directory(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    runs_dir = tmp_path / "runs"

    manifest_a = RunManifest.create(runs_dir=runs_dir)
    manifest_b = RunManifest.create(runs_dir=runs_dir)
    assert manifest_a.run_dir != manifest_b.run_dir

    run_discovery(adapter, manifest_a, config_path="config/project.yaml")
    run_discovery(adapter, manifest_b, config_path="config/project.yaml")

    assert manifest_a.output_path(DISCOVERY_FILENAME).exists()
    assert manifest_b.output_path(DISCOVERY_FILENAME).exists()


def test_discovery_output_round_trips_through_load_discovery(
    temp_sqlite_db: Path, tmp_path: Path
):
    """What run_discovery() writes, load_discovery() must accept - this is
    the contract future stages (profiling, inference) will rely on instead
    of re-parsing discovery.json themselves.
    """
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    run_discovery(adapter, manifest, config_path="config/project.yaml")

    loaded = load_discovery(manifest.output_path(DISCOVERY_FILENAME))
    assert loaded["source_fingerprint"] == manifest.source_fingerprint
    assert set(loaded["tables"]) == set(adapter.list_tables())
