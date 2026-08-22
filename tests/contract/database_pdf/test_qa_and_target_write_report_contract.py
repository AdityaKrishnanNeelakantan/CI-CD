"""Contract: run_qa_validation() and run_target_write() always write the
same top-level shape.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.validation.database.qa_service import QA_REPORT_FILENAME, load_qa_report, run_qa_validation
from synth_platform.engine.generation.database.target_write_service import (
    TARGET_WRITE_REPORT_FILENAME,
    load_target_write_report,
    run_target_write,
)

pytestmark = pytest.mark.contract

QA_REQUIRED_TOP_LEVEL_KEYS = {"validated_at", "hard_checks_passed", "report", "release"}
TARGET_WRITE_REQUIRED_TOP_LEVEL_KEYS = {"written_at", "target_db_path", "write_report", "validation_report"}


def _contract():
    return {
        "tables": {
            "customers": {
                "primary_key": ["customer_id"],
                "foreign_keys": [],
                "columns": {
                    "customer_id": {"physical_type": "TEXT", "nullable": False},
                },
            }
        }
    }


def _relational_report(tmp_path):
    csv_path = tmp_path / "customers.csv"
    pd.DataFrame({"customer_id": ["a", "b", "c"]}).to_csv(csv_path, index=False)
    return {
        "generation_order": ["customers"],
        "fk_validity": {"edges": {}, "overall_fk_validity": 1.0},
        "constraint_reports": {},
        "tables": {"customers": {"row_count": 3, "path": str(csv_path)}},
    }


def test_qa_report_shape(tmp_path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_qa_validation(
        _contract(), _relational_report(tmp_path), manifest, relational_report_reference="r.json"
    )
    assert result.is_success()
    report = load_qa_report(manifest.output_path(QA_REPORT_FILENAME))
    assert set(report) == QA_REQUIRED_TOP_LEVEL_KEYS


def test_target_write_report_shape(tmp_path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    relational_report = _relational_report(tmp_path)
    qa_report = {"hard_checks_passed": True}

    target_db_path = tmp_path / "target.db"
    result = run_target_write(
        target_db_path, _contract(), relational_report, qa_report, manifest, qa_report_reference="qa.json"
    )
    assert result.is_success()

    report = load_target_write_report(manifest.output_path(TARGET_WRITE_REPORT_FILENAME))
    assert set(report) == TARGET_WRITE_REQUIRED_TOP_LEVEL_KEYS

    conn = sqlite3.connect(str(target_db_path))
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 3
    conn.close()
