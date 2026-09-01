from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.common.database.config import ConfigError, load_project_config

pytestmark = pytest.mark.unit


def write_config(tmp_path: Path, connection_path_value: str) -> Path:
    config_path = tmp_path / "project.yaml"
    config_path.write_text(
        f"""
project:
  name: "test-project"
  dataset_id: "test-dataset"

source:
  type: sqlite
  connection:
    path: "{connection_path_value}"
  sampling:
    default_limit: 500
    max_limit: 5000

output:
  runs_dir: "runs"
  metadata_dir: "metadata"
""",
        encoding="utf-8",
    )
    return config_path


def test_load_project_config_with_literal_path(tmp_path: Path):
    config_path = write_config(tmp_path, "/data/mydb.sqlite")
    config = load_project_config(config_path)

    assert config.name == "test-project"
    assert config.source.type == "sqlite"
    assert config.source.connection["path"] == "/data/mydb.sqlite"
    assert config.source.sampling.default_limit == 500


def test_env_var_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_DB_PATH", "/secret/path/db.sqlite")
    config_path = write_config(tmp_path, "${TEST_DB_PATH}")
    config = load_project_config(config_path)
    assert config.source.connection["path"] == "/secret/path/db.sqlite"


def test_missing_env_var_raises_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TEST_DB_PATH_UNSET", raising=False)
    config_path = write_config(tmp_path, "${TEST_DB_PATH_UNSET}")
    with pytest.raises(ConfigError):
        load_project_config(config_path)


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_project_config(tmp_path / "does_not_exist.yaml")
