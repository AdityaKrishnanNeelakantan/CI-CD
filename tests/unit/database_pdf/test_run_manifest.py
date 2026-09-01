from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from synth_platform.engine.common.database.core.observability import LOGGER_NAME
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import StageResult

pytestmark = pytest.mark.unit


def test_create_makes_unique_run_directory(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    manifest_a = RunManifest.create(runs_dir=runs_dir)
    manifest_b = RunManifest.create(runs_dir=runs_dir)

    assert manifest_a.run_id != manifest_b.run_id
    assert manifest_a.run_dir.exists()
    assert manifest_b.run_dir.exists()
    assert (manifest_a.run_dir / "run_manifest.json").exists()


def test_record_stage_appends_without_losing_prior_entries(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    manifest.record_stage(
        StageResult(
            stage_name="discovery",
            status="success",
            input_references=["config/project.yaml"],
            output_references=["discovery.json"],
        )
    )
    manifest.record_stage(
        StageResult(
            stage_name="profiling",
            status="success",
            input_references=["discovery.json"],
            output_references=["profile.json"],
        )
    )

    with (manifest.run_dir / "run_manifest.json").open() as f:
        data = json.load(f)

    stage_names = [s["stage_name"] for s in data["stages"]]
    assert stage_names == ["discovery", "profiling"]


def test_manifest_records_code_version(tmp_path: Path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    assert manifest.code_version
    assert isinstance(manifest.code_version, str)


def test_record_stage_logs_outcome_via_observability(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        manifest.record_stage(
            StageResult(stage_name="discovery", status="success", input_references=[], output_references=[])
        )

    assert any(r.levelno == logging.INFO and "discovery" in r.message for r in caplog.records)
