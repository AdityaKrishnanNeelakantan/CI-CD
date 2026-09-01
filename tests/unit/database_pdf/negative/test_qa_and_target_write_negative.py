from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.validation.database.qa_service import QAReportLoadError, load_qa_report, run_qa_validation
from synth_platform.engine.generation.database.target_write_service import (
    TargetWriteReportLoadError,
    load_target_write_report,
    run_target_write,
)

pytestmark = pytest.mark.negative


def _contract():
    return {
        "tables": {
            "customers": {
                "primary_key": ["customer_id"],
                "foreign_keys": [],
                "columns": {"customer_id": {"physical_type": "TEXT", "nullable": False}},
            }
        }
    }


def _relational_report(tmp_path, duplicate_pk: bool = False):
    ids = ["a", "a", "c"] if duplicate_pk else ["a", "b", "c"]
    csv_path = tmp_path / "customers.csv"
    pd.DataFrame({"customer_id": ids}).to_csv(csv_path, index=False)
    return {
        "generation_order": ["customers"],
        "fk_validity": {"edges": {}, "overall_fk_validity": 1.0},
        "constraint_reports": {},
        "tables": {"customers": {"row_count": len(ids), "path": str(csv_path)}},
    }


def test_qa_validation_never_overwrites_existing_output(tmp_path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_qa_validation(_contract(), _relational_report(tmp_path), manifest, relational_report_reference="r.json")
    with pytest.raises(RuntimeError):
        run_qa_validation(_contract(), _relational_report(tmp_path), manifest, relational_report_reference="r.json")


def test_qa_validation_flags_duplicate_primary_keys_as_a_failed_hard_check(tmp_path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_qa_validation(
        _contract(), _relational_report(tmp_path, duplicate_pk=True), manifest, relational_report_reference="r.json"
    )
    assert result.is_success()  # the stage itself still succeeds - it recorded real evidence
    assert result.metrics["hard_checks_passed"] is False
    assert "hard_checks_failed" in result.warnings


def test_qa_validation_fails_closed_on_missing_input_csv(tmp_path):
    """A prior-stage table CSV that no longer exists on disk (e.g. deleted
    between relational generation and QA validation) must produce a clean
    failed StageResult, not crash run_qa_validation() with a raw
    FileNotFoundError.
    """
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    relational_report = _relational_report(tmp_path)
    relational_report["tables"]["customers"]["path"] = str(tmp_path / "does_not_exist.csv")

    result = run_qa_validation(_contract(), relational_report, manifest, relational_report_reference="r.json")

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("qa_report.json").exists()


def test_qa_validation_fails_closed_on_write_error(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A disk-full/permission-denied failure while writing qa_report.json
    must surface as a clean failed StageResult, not crash run_qa_validation()
    with a raw OSError.
    """
    import synth_platform.engine.validation.database.qa_service as qa_service_module

    real_dump = qa_service_module.json.dump

    def _raising_dump(obj, fp, *args, **kwargs):
        if Path(fp.name).name == "qa_report.json":
            raise OSError("disk full")
        return real_dump(obj, fp, *args, **kwargs)

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    monkeypatch.setattr(qa_service_module.json, "dump", _raising_dump)

    result = run_qa_validation(_contract(), _relational_report(tmp_path), manifest, relational_report_reference="r.json")

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("qa_report.json").exists()


def test_target_write_refuses_when_qa_hard_checks_failed(tmp_path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    relational_report = _relational_report(tmp_path)
    qa_report = {"hard_checks_passed": False}

    result = run_target_write(
        tmp_path / "target.db", _contract(), relational_report, qa_report, manifest, qa_report_reference="qa.json"
    )
    assert result.status == "failed"
    assert result.errors
    assert not (tmp_path / "target.db").exists() or sqlite3.connect(str(tmp_path / "target.db")).execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall() == []


def test_target_write_never_overwrites_existing_report(tmp_path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    relational_report = _relational_report(tmp_path)
    qa_report = {"hard_checks_passed": True}
    run_target_write(
        tmp_path / "target.db", _contract(), relational_report, qa_report, manifest, qa_report_reference="qa.json"
    )
    with pytest.raises(RuntimeError):
        run_target_write(
            tmp_path / "target.db", _contract(), relational_report, qa_report, manifest, qa_report_reference="qa.json"
        )


def test_target_write_fails_closed_on_incompatible_target_schema(tmp_path):
    target_path = tmp_path / "target.db"
    conn = sqlite3.connect(str(target_path))
    conn.execute("CREATE TABLE customers (totally_unrelated TEXT)")
    conn.commit()
    conn.close()

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_target_write(
        target_path, _contract(), _relational_report(tmp_path), {"hard_checks_passed": True},
        manifest, qa_report_reference="qa.json",
    )
    assert result.status == "failed"
    assert result.errors


def test_target_write_fails_closed_on_missing_input_csv(tmp_path):
    """A prior-stage table CSV that no longer exists on disk must produce a
    clean failed StageResult, not crash run_target_write() with a raw
    FileNotFoundError.
    """
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    relational_report = _relational_report(tmp_path)
    relational_report["tables"]["customers"]["path"] = str(tmp_path / "does_not_exist.csv")

    result = run_target_write(
        tmp_path / "target.db", _contract(), relational_report, {"hard_checks_passed": True},
        manifest, qa_report_reference="qa.json",
    )
    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("target_write_report.json").exists()


def test_target_write_fails_closed_on_write_error(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A disk-full/permission-denied failure while writing
    target_write_report.json must surface as a clean failed StageResult, not
    crash run_target_write() with a raw OSError.
    """
    import synth_platform.engine.generation.database.target_write_service as target_write_service_module

    real_dump = target_write_service_module.json.dump

    def _raising_dump(obj, fp, *args, **kwargs):
        if Path(fp.name).name == "target_write_report.json":
            raise OSError("disk full")
        return real_dump(obj, fp, *args, **kwargs)

    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    monkeypatch.setattr(target_write_service_module.json, "dump", _raising_dump)

    result = run_target_write(
        tmp_path / "target.db", _contract(), _relational_report(tmp_path), {"hard_checks_passed": True},
        manifest, qa_report_reference="qa.json",
    )
    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("target_write_report.json").exists()


def test_load_qa_report_rejects_missing_file(tmp_path):
    with pytest.raises(QAReportLoadError):
        load_qa_report(tmp_path / "does_not_exist.json")


def test_load_qa_report_rejects_missing_top_level_keys(tmp_path):
    path = tmp_path / "qa_report.json"
    path.write_text(json.dumps({"validated_at": "x"}), encoding="utf-8")
    with pytest.raises(QAReportLoadError):
        load_qa_report(path)


def test_load_target_write_report_rejects_missing_file(tmp_path):
    with pytest.raises(TargetWriteReportLoadError):
        load_target_write_report(tmp_path / "does_not_exist.json")


def test_load_target_write_report_rejects_missing_top_level_keys(tmp_path):
    path = tmp_path / "target_write_report.json"
    path.write_text(json.dumps({"written_at": "x"}), encoding="utf-8")
    with pytest.raises(TargetWriteReportLoadError):
        load_target_write_report(path)
