"""Checkpoint 7 orchestration: turns a relational_generation_report.json
(FK validity + business-rule evidence, already on disk) plus the
generated table CSVs plus an optional reference_profile.json into one
persisted qa_report.json - the unified Validate stage for the database
track, mirroring the PDF track's document_validation_report.json.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.validation.database.qa_report import build_qa_report
from synth_platform.engine.validation.database.release_manager import MODE_LEARNED_RESTRICTED, evaluate_release

STAGE_NAME = "qa_validation"
QA_REPORT_FILENAME = "qa_report.json"

_REQUIRED_TOP_LEVEL_KEYS = {"validated_at", "hard_checks_passed", "report", "release"}


class QAReportLoadError(Exception):
    """Raised when qa_report.json is missing, corrupted, or malformed."""


def run_qa_validation(
    dataset_contract: dict[str, Any],
    relational_generation_report: dict[str, Any],
    manifest: RunManifest,
    relational_report_reference: str,
    reference_profile: dict[str, Any] | None = None,
    release_mode: str = MODE_LEARNED_RESTRICTED,
    dp_report: dict[str, Any] | None = None,
) -> StageResult:
    output_path = manifest.output_path(QA_REPORT_FILENAME)
    if output_path.exists():
        raise RuntimeError(
            f"{QA_REPORT_FILENAME} already exists for run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    try:
        use_streaming_paths = bool(relational_generation_report.get("streaming"))
        if use_streaming_paths:
            from synth_platform.engine.validation.database.qa_report import build_qa_report_from_paths

            report = build_qa_report_from_paths(
                table_entries=relational_generation_report["tables"],
                dataset_contract=dataset_contract,
                fk_validity=relational_generation_report["fk_validity"],
                constraint_reports=relational_generation_report["constraint_reports"],
                reference_profile=reference_profile,
                chunk_size=int(relational_generation_report.get("batch_size") or 10_000),
            )
            table_count = len(relational_generation_report["tables"])
        else:
            tables = {
                table_name: pd.read_csv(entry["path"])
                for table_name, entry in relational_generation_report["tables"].items()
            }

            report = build_qa_report(
                tables=tables,
                dataset_contract=dataset_contract,
                fk_validity=relational_generation_report["fk_validity"],
                constraint_reports=relational_generation_report["constraint_reports"],
                reference_profile=reference_profile,
            )
            table_count = len(tables)

        qa_report = {
            "validated_at": datetime.now(UTC).isoformat(),
            "hard_checks_passed": report["hard_checks_passed"],
            "report": report,
            "release": evaluate_release(
                {"hard_checks_passed": report["hard_checks_passed"], "report": report},
                release_mode=release_mode,
                dp_report=dp_report,
            ),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(qa_report, f, indent=2, sort_keys=True)
    except (OSError, pd.errors.ParserError, KeyError, ValueError) as exc:
        output_path.unlink(missing_ok=True)
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[relational_report_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[relational_report_reference],
        output_references=[str(output_path)],
        metrics={
            "overall_fk_validity": report["integrity"]["fk_validity"]["overall_fk_validity"],
            "hard_checks_passed": report["hard_checks_passed"],
            "table_count": table_count,
        },
        warnings=[] if report["hard_checks_passed"] else ["hard_checks_failed"],
        evidence={"validated_at": qa_report["validated_at"]},
    )
    manifest.record_stage(result)
    return result


def load_qa_report(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a qa_report.json file."""
    report_path = Path(path)
    if not report_path.is_file():
        raise QAReportLoadError(f"qa_report.json not found: {report_path}")

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise QAReportLoadError(f"qa_report.json is not valid JSON: {report_path}") from exc

    if not isinstance(data, dict):
        raise QAReportLoadError(f"qa_report.json did not parse to an object: {report_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise QAReportLoadError(f"qa_report.json missing required keys: {sorted(missing)}")

    return data
