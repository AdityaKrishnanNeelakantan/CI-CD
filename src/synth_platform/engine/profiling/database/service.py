"""Checkpoint 2: turn discovery.json + a fresh sample into recorded profiling evidence.

Reads: discovery.json (for the approved table/column list and schema, via
whatever already loaded it - typically src.discovery.service.load_discovery)
plus a bounded sample read directly through the same SourceAdapter used for
discovery. Writes: <run_dir>/profile.json, plus a StageResult appended to
the run manifest. No raw sampled rows are persisted - only derived
statistics ever reach disk.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.engine.discovery.database.adapters.errors import SourceAdapterError
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.profiling.database.profiler import StructuredProfiler

STAGE_NAME = "profiling"
PROFILE_FILENAME = "profile.json"

_REQUIRED_TOP_LEVEL_KEYS = {"profiled_at", "tables"}
_REQUIRED_TABLE_KEYS = {
    "table_name",
    "row_count",
    "columns",
    "correlations",
    "warnings",
    "execution_metadata",
}


class ProfileLoadError(Exception):
    """Raised when a profile.json file is missing, corrupted, or malformed."""


def run_profiling(
    adapter: SourceAdapter,
    discovery_data: dict[str, Any],
    manifest: RunManifest,
    discovery_reference: str,
    sample_limit: int,
    profiler: StructuredProfiler | None = None,
    chunk_size: int | None = None,
) -> StageResult:
    """Run profiling for every table in discovery_data.

    By default reads and profiles each table's sample in one shot
    (unchanged behavior). When ``chunk_size`` is given, the sample is
    instead streamed and profiled in bounded-memory pieces via
    ``adapter.read_sample_chunks()`` / ``StructuredProfiler.profile_table_chunked()``
    - see that method's docstring for the resulting trade-offs
    (correlations omitted, distinct_count approximate for high-cardinality
    columns).
    """
    output_path = manifest.output_path(PROFILE_FILENAME)
    if output_path.exists():
        raise RuntimeError(
            f"{PROFILE_FILENAME} already exists for run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    profiler = profiler or StructuredProfiler()
    table_reports: dict[str, Any] = {}
    warnings: list[str] = []

    try:
        for table_name, table_schema in discovery_data["tables"].items():
            if chunk_size is not None:
                chunks = adapter.read_sample_chunks(table_name, limit=sample_limit, chunk_size=chunk_size)
                report = profiler.profile_table_chunked(chunks, table_name, table_schema)
            else:
                df = adapter.read_sample(table_name, limit=sample_limit)
                report = profiler.profile_table(df, table_name, table_schema)
            table_reports[table_name] = report
            warnings.extend(f"{table_name}: {w}" for w in report["warnings"])
    except SourceAdapterError as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[discovery_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result
    except Exception as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[discovery_reference],
            output_references=[],
            errors=[f"profiling failed while analyzing source values: {exc}"],
        )
        manifest.record_stage(result)
        return result

    profile_data = {
        "profiled_at": datetime.now(UTC).isoformat(),
        "tables": table_reports,
    }
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, sort_keys=True)
    except OSError as exc:
        output_path.unlink(missing_ok=True)
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[discovery_reference],
            output_references=[],
            errors=[f"could not write {PROFILE_FILENAME}: {exc}"],
        )
        manifest.record_stage(result)
        return result

    total_rows_profiled = sum(report["row_count"] for report in table_reports.values())

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[discovery_reference],
        output_references=[str(output_path)],
        metrics={
            "table_count": len(table_reports),
            "total_rows_profiled": total_rows_profiled,
        },
        warnings=warnings,
        evidence={"profiled_at": profile_data["profiled_at"]},
    )
    manifest.record_stage(result)
    return result


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a profile.json produced by run_profiling().

    Downstream stages (semantic inference) must call this instead of
    parsing profile.json themselves, so a missing, corrupted, or
    hand-edited file is rejected in one place.
    """
    profile_path = Path(path)
    if not profile_path.is_file():
        raise ProfileLoadError(f"profile.json not found: {profile_path}")

    try:
        with profile_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ProfileLoadError(f"profile.json is not valid JSON: {profile_path}") from exc

    if not isinstance(data, dict):
        raise ProfileLoadError(f"profile.json did not parse to an object: {profile_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise ProfileLoadError(f"profile.json missing required keys: {sorted(missing)}")

    if not isinstance(data["tables"], dict):
        raise ProfileLoadError("profile.json 'tables' must be an object")

    for table_name, table in data["tables"].items():
        if not isinstance(table, dict):
            raise ProfileLoadError(f"profile.json table {table_name!r} must be an object")
        missing_table_keys = _REQUIRED_TABLE_KEYS - table.keys()
        if missing_table_keys:
            raise ProfileLoadError(
                f"profile.json table {table_name!r} missing keys: {sorted(missing_table_keys)}"
            )

    return data
