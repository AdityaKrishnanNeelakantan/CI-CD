"""Integration: SQLite adapter -> DiscoveryService -> ProfilingService -> RunManifest.

Extends the discovery integration chain by one link: discovery.json feeds
directly into profiling (both the table list and, for the orders table, the
primary_key used for the PK cross-check).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.profiling.database.service import PROFILE_FILENAME, load_profile, run_profiling

pytestmark = pytest.mark.integration


def test_profiling_consumes_discovery_output_and_writes_profile_json(
    temp_sqlite_db: Path, tmp_path: Path
):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_path = Path(discovery_result.output_references[0])
    discovery_data = json.loads(discovery_path.read_text(encoding="utf-8"))

    profiling_result = run_profiling(
        adapter,
        discovery_data,
        manifest,
        discovery_reference=str(discovery_path),
        sample_limit=100,
    )

    assert profiling_result.is_success()
    profile_path = manifest.output_path(PROFILE_FILENAME)
    assert profile_path.exists()

    profile_data = load_profile(profile_path)
    # Every table discovery found must show up in the profile - profiling
    # must not silently skip a table.
    assert set(profile_data["tables"]) == set(discovery_data["tables"])

    orders_profile = profile_data["tables"]["orders"]
    assert orders_profile["row_count"] == adapter.estimate_row_count("orders")
    assert set(orders_profile["columns"]) == {
        col["name"] for col in discovery_data["tables"]["orders"]["columns"]
    }

    manifest_data = json.loads((manifest.run_dir / "run_manifest.json").read_text())
    stage_names = [s["stage_name"] for s in manifest_data["stages"]]
    assert stage_names == ["discovery", "profiling"]


def test_profiling_with_chunk_size_produces_equivalent_row_counts(
    temp_sqlite_db: Path, tmp_path: Path
):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_path = Path(discovery_result.output_references[0])
    discovery_data = json.loads(discovery_path.read_text(encoding="utf-8"))

    profiling_result = run_profiling(
        adapter,
        discovery_data,
        manifest,
        discovery_reference=str(discovery_path),
        sample_limit=100,
        chunk_size=2,
    )

    assert profiling_result.is_success()
    profile_data = load_profile(manifest.output_path(PROFILE_FILENAME))
    orders_profile = profile_data["tables"]["orders"]
    assert orders_profile["row_count"] == adapter.estimate_row_count("orders")
    assert orders_profile["correlations"] == []
    assert "correlations_skipped_in_chunked_mode" in orders_profile["warnings"]
    assert orders_profile["execution_metadata"]["chunk_count"] == 3
