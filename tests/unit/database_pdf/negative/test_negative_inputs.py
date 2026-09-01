"""Negative tests: incorrect inputs must fail correctly, not silently.

Each test here asserts both that failure happens AND that it happens
through the intended mechanism (a specific exception type, or a StageResult
recorded as failed with no output written) - a bare "it raised something"
is not enough per the anti-hallucination rule "fail with a clear validation
error instead of guessing".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.errors import SourceConfigError, UnknownTableError
from synth_platform.engine.discovery.database.adapters.registry import create_source_adapter
from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.config import ConfigError, load_project_config
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import DISCOVERY_FILENAME, run_discovery

pytestmark = pytest.mark.negative


def test_run_discovery_fails_closed_on_missing_source(tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(tmp_path / "missing.db")})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    result = run_discovery(adapter, manifest, config_path="config/project.yaml")

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path(DISCOVERY_FILENAME).exists()

    with (manifest.run_dir / "run_manifest.json").open() as f:
        manifest_data = json.load(f)
    assert manifest_data["stages"][0]["status"] == "failed"


def test_run_discovery_never_overwrites_existing_output(
    temp_sqlite_db: Path, tmp_path: Path
):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    run_discovery(adapter, manifest, config_path="config/project.yaml")
    with pytest.raises(RuntimeError):
        run_discovery(adapter, manifest, config_path="config/project.yaml")


def test_run_directory_is_never_reused(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    RunManifest.create(runs_dir=runs_dir, run_id="forced-collision")
    with pytest.raises(FileExistsError):
        RunManifest.create(runs_dir=runs_dir, run_id="forced-collision")


def test_unregistered_source_type_is_rejected():
    with pytest.raises(SourceConfigError):
        create_source_adapter("postgresql", {"host": "localhost"})


def test_malformed_yaml_config_is_rejected(tmp_path: Path):
    config_path = tmp_path / "project.yaml"
    config_path.write_text("project:\n  name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_project_config(config_path)


def test_non_integer_sampling_limit_is_rejected(tmp_path: Path):
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        """
project:
  name: "p"
  dataset_id: "d"
source:
  type: sqlite
  connection:
    path: "db.sqlite"
  sampling:
    default_limit: "not-a-number"
    max_limit: 1000
output:
  runs_dir: "runs"
  metadata_dir: "metadata"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_project_config(config_path)


@pytest.mark.parametrize(
    "operation",
    [
        lambda adapter: adapter.get_schema("does_not_exist"),
        lambda adapter: adapter.get_primary_keys("does_not_exist"),
        lambda adapter: adapter.get_foreign_keys("does_not_exist"),
        lambda adapter: adapter.get_indexes("does_not_exist"),
        lambda adapter: adapter.estimate_row_count("does_not_exist"),
        lambda adapter: adapter.read_sample("does_not_exist", limit=10),
    ],
)
def test_every_table_operation_rejects_unknown_table(temp_sqlite_db: Path, operation):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    with pytest.raises(UnknownTableError):
        operation(adapter)
