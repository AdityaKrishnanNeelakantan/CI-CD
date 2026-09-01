"""Checkpoint 8 orchestration: writes a validated relational dataset into
a target database (Database B) and persists target_write_report.json -
the final stage of "Database A -> generate -> Database B".

Refuses to write at all unless the QA report's hard_checks_passed is
True - a dataset that failed integrity/business-rule validation must
never reach a target database. "Target write committed successfully" is
one of this project's reference design's acceptance criteria, never a
substitute for the integrity checks that must already have passed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.generation.database.target_writer import (
    TargetWriterError,
    validate_write,
    write_dataset,
)

STAGE_NAME = "target_write"
TARGET_WRITE_REPORT_FILENAME = "target_write_report.json"

_REQUIRED_TOP_LEVEL_KEYS = {"written_at", "target_db_path", "write_report", "validation_report"}


class TargetWriteReportLoadError(Exception):
    """Raised when target_write_report.json is missing, corrupted, or malformed."""


def run_target_write(
    target_db_path: str | Path,
    dataset_contract: dict[str, Any],
    relational_generation_report: dict[str, Any],
    qa_report: dict[str, Any],
    manifest: RunManifest,
    qa_report_reference: str,
) -> StageResult:
    output_path = manifest.output_path(TARGET_WRITE_REPORT_FILENAME)
    if output_path.exists():
        raise RuntimeError(
            f"{TARGET_WRITE_REPORT_FILENAME} already exists for run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    if not qa_report["hard_checks_passed"]:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[qa_report_reference],
            output_references=[],
            errors=["QA report hard_checks_passed is False; refusing to write to the target database"],
        )
        manifest.record_stage(result)
        return result

    try:
        tables = {
            table_name: pd.read_csv(entry["path"])
            for table_name, entry in relational_generation_report["tables"].items()
        }
        generation_order = relational_generation_report["generation_order"]

        write_report = write_dataset(target_db_path, tables, dataset_contract, generation_order)
        validation_report = validate_write(target_db_path, tables)

        report_data = {
            "written_at": datetime.now(UTC).isoformat(),
            "target_db_path": str(target_db_path),
            "write_report": write_report,
            "validation_report": validation_report,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, sort_keys=True)
    except (TargetWriterError, OSError, pd.errors.ParserError) as exc:
        output_path.unlink(missing_ok=True)
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[qa_report_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[qa_report_reference],
        output_references=[str(output_path)],
        metrics={
            "table_count": len(tables),
            "total_rows_written": sum(len(df) for df in tables.values()),
            "all_writes_confirmed": validation_report["all_writes_confirmed"],
        },
        evidence={"written_at": report_data["written_at"]},
    )
    manifest.record_stage(result)
    return result


def load_target_write_report(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a target_write_report.json file."""
    report_path = Path(path)
    if not report_path.is_file():
        raise TargetWriteReportLoadError(f"target_write_report.json not found: {report_path}")

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise TargetWriteReportLoadError(f"target_write_report.json is not valid JSON: {report_path}") from exc

    if not isinstance(data, dict):
        raise TargetWriteReportLoadError(f"target_write_report.json did not parse to an object: {report_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise TargetWriteReportLoadError(f"target_write_report.json missing required keys: {sorted(missing)}")

    return data
