from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.profiling.database.service import ProfileLoadError, load_profile, run_profiling

pytestmark = pytest.mark.negative


def _discover(adapter: SQLiteSourceAdapter, manifest: RunManifest) -> dict:
    result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    return json.loads(Path(result.output_references[0]).read_text())


def test_profiling_never_overwrites_existing_output(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    discovery_data = _discover(adapter, manifest)

    run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)
    with pytest.raises(RuntimeError):
        run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)


def test_profiling_fails_closed_on_table_missing_from_source(
    temp_sqlite_db: Path, tmp_path: Path
):
    """discovery.json referencing a table the live source no longer has
    (e.g. dropped between discovery and profiling) must fail the whole
    stage, not silently profile the tables that do exist.
    """
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    discovery_data = _discover(adapter, manifest)
    discovery_data["tables"]["__ghost_table__"] = discovery_data["tables"]["orders"]

    result = run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("profile.json").exists()


def test_load_profile_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ProfileLoadError):
        load_profile(tmp_path / "does_not_exist.json")


def test_load_profile_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "profile.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ProfileLoadError):
        load_profile(path)


def test_run_discovery_fails_closed_on_write_error(
    temp_sqlite_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A disk-full/permission-denied failure while writing discovery.json
    must surface as a clean failed StageResult, not crash run_discovery()
    with a raw OSError - matching the module's own StageResult contract.
    """
    import synth_platform.engine.discovery.database.service as discovery_module

    real_dump = discovery_module.json.dump

    def _raising_dump(obj, fp, *args, **kwargs):
        if Path(fp.name).name == "discovery.json":
            raise OSError("disk full")
        return real_dump(obj, fp, *args, **kwargs)

    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    monkeypatch.setattr(discovery_module.json, "dump", _raising_dump)

    result = run_discovery(adapter, manifest, config_path="config/project.yaml")

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("discovery.json").exists()


def test_run_profiling_fails_closed_on_write_error(
    temp_sqlite_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Same guarantee as run_discovery's write-error handling, for
    profile.json - a disk failure must produce a failed StageResult.
    """
    import synth_platform.engine.profiling.database.service as profiling_module

    real_dump = profiling_module.json.dump

    def _raising_dump(obj, fp, *args, **kwargs):
        if Path(fp.name).name == "profile.json":
            raise OSError("disk full")
        return real_dump(obj, fp, *args, **kwargs)

    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    discovery_data = _discover(adapter, manifest)

    monkeypatch.setattr(profiling_module.json, "dump", _raising_dump)
    result = run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("profile.json").exists()


def test_load_profile_rejects_missing_top_level_keys(tmp_path: Path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"tables": {}}), encoding="utf-8")  # missing "profiled_at"
    with pytest.raises(ProfileLoadError):
        load_profile(path)


def test_load_profile_rejects_table_missing_required_keys(tmp_path: Path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiled_at": "2026-01-01T00:00:00Z",
                "tables": {"customers": {"row_count": 3}},  # missing columns, correlations, ...
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProfileLoadError):
        load_profile(path)
