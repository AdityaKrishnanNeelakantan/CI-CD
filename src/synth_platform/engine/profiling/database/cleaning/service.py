"""Orchestration for the cleaning stage: turns the approved dataset_contract
plus a fresh sample into recorded cleaning evidence.

Like profiling and inference, this stage re-reads a bounded sample through
the same SourceAdapter rather than receiving a DataFrame from an earlier
stage, and persists only derived evidence (counts, strategies applied) -
never the cleaned rows themselves. Checkpoint 4's trainer will call
DataCleaner directly on its own training read; this stage exists purely to
produce an auditable record of what cleaning would do to the approved
columns.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.engine.discovery.database.adapters.errors import SourceAdapterError
from synth_platform.engine.profiling.database.cleaning.cleaner import DataCleaner
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult

STAGE_NAME = "cleaning"
CLEANING_REPORT_FILENAME = "cleaning_report.json"

_REQUIRED_TOP_LEVEL_KEYS = {"cleaned_at", "tables"}
_REQUIRED_TABLE_KEYS = {
    "table_name",
    "row_count",
    "duplicate_row_count",
    "excluded_columns",
    "columns",
    "warnings",
}


class CleaningReportLoadError(Exception):
    """Raised when cleaning_report.json is missing, corrupted, or malformed."""


def run_cleaning(
    adapter: SourceAdapter,
    dataset_contract: dict[str, Any],
    manifest: RunManifest,
    contract_reference: str,
    sample_limit: int,
    cleaner: DataCleaner | None = None,
    chunk_size: int | None = None,
) -> StageResult:
    output_path = manifest.output_path(CLEANING_REPORT_FILENAME)
    if output_path.exists():
        raise RuntimeError(
            f"{CLEANING_REPORT_FILENAME} already exists for run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    cleaner = cleaner or DataCleaner()
    table_reports: dict[str, Any] = {}
    warnings: list[str] = []

    try:
        for table_name, table_contract in dataset_contract["tables"].items():
            if chunk_size is not None:
                chunks = adapter.read_sample_chunks(table_name, limit=sample_limit, chunk_size=chunk_size)
                report = cleaner.clean_table_chunked(chunks, table_name, table_contract)
            else:
                df = adapter.read_sample(table_name, limit=sample_limit)
                _cleaned_df, report = cleaner.clean_table(df, table_name, table_contract)
            table_reports[table_name] = report
            warnings.extend(f"{table_name}: {w}" for w in report["warnings"])
            if not report["columns"]:
                warnings.append(f"{table_name}: zero approved columns available for cleaning")
    except SourceAdapterError as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[contract_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    report_data = {
        "cleaned_at": datetime.now(UTC).isoformat(),
        "tables": table_reports,
    }
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, sort_keys=True)
    except OSError as exc:
        output_path.unlink(missing_ok=True)
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[contract_reference],
            output_references=[],
            errors=[f"could not write {CLEANING_REPORT_FILENAME}: {exc}"],
        )
        manifest.record_stage(result)
        return result

    total_nulls_imputed = sum(
        col["nulls_imputed"] for table in table_reports.values() for col in table["columns"].values()
    )
    total_duplicate_rows = sum(table["duplicate_row_count"] for table in table_reports.values())

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[contract_reference],
        output_references=[str(output_path)],
        metrics={
            "table_count": len(table_reports),
            "total_nulls_imputed": total_nulls_imputed,
            "total_duplicate_rows": total_duplicate_rows,
        },
        warnings=warnings,
        evidence={"cleaned_at": report_data["cleaned_at"]},
    )
    manifest.record_stage(result)
    return result


def load_cleaning_report(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a cleaning_report.json file."""
    report_path = Path(path)
    if not report_path.is_file():
        raise CleaningReportLoadError(f"cleaning_report.json not found: {report_path}")

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise CleaningReportLoadError(f"cleaning_report.json is not valid JSON: {report_path}") from exc

    if not isinstance(data, dict):
        raise CleaningReportLoadError(f"cleaning_report.json did not parse to an object: {report_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise CleaningReportLoadError(f"cleaning_report.json missing required keys: {sorted(missing)}")

    for table_name, table in data["tables"].items():
        if not isinstance(table, dict):
            raise CleaningReportLoadError(f"cleaning_report.json table {table_name!r} must be an object")
        missing_table_keys = _REQUIRED_TABLE_KEYS - table.keys()
        if missing_table_keys:
            raise CleaningReportLoadError(
                f"cleaning_report.json table {table_name!r} missing keys: {sorted(missing_table_keys)}"
            )

    return data
