"""Checkpoint 1: turn a SourceAdapter into recorded discovery evidence.

Reads: an approved SourceAdapter (config already validated by the adapter's
constructor) and the config file path it was built from.
Writes: <run_dir>/discovery.json, plus a StageResult appended to the
run manifest. This is the only place discovery.json is produced - nothing
else in the pipeline should re-derive schema metadata from the source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synth_platform.engine.discovery.database.adapters.base import SourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult

STAGE_NAME = "discovery"
DISCOVERY_FILENAME = "discovery.json"

_REQUIRED_TOP_LEVEL_KEYS = {"source_type", "discovered_at", "tables", "source_fingerprint"}
_REQUIRED_TABLE_KEYS = {"columns", "primary_key", "foreign_keys", "indexes", "estimated_row_count"}


class DiscoveryLoadError(Exception):
    """Raised when a discovery.json file is missing, corrupted, or malformed."""


def run_discovery(
    adapter: SourceAdapter,
    manifest: RunManifest,
    config_path: str,
) -> StageResult:
    output_path = manifest.output_path(DISCOVERY_FILENAME)
    if output_path.exists():
        raise RuntimeError(
            f"{DISCOVERY_FILENAME} already exists for run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    health = adapter.test_connection()
    if not health.get("healthy", False):
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[config_path],
            output_references=[],
            errors=[health.get("error", "source connection health check failed")],
            evidence={"connection_health": health},
        )
        manifest.record_stage(result)
        return result

    discovery_data = adapter.discover()

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(discovery_data, f, indent=2, sort_keys=True)
    except OSError as exc:
        output_path.unlink(missing_ok=True)
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[config_path],
            output_references=[],
            errors=[f"could not write {DISCOVERY_FILENAME}: {exc}"],
        )
        manifest.record_stage(result)
        return result

    manifest.source_fingerprint = discovery_data["source_fingerprint"]
    manifest.write()

    table_count = len(discovery_data["tables"])
    total_estimated_rows = sum(
        table["estimated_row_count"] for table in discovery_data["tables"].values()
    )

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[config_path],
        output_references=[str(output_path)],
        metrics={
            "table_count": table_count,
            "total_estimated_rows": total_estimated_rows,
        },
        evidence={
            "connection_health": health,
            "source_fingerprint": discovery_data["source_fingerprint"],
            "discovered_at": discovery_data["discovered_at"],
        },
    )
    manifest.record_stage(result)
    return result


def load_discovery(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a discovery.json produced by run_discovery().

    Downstream stages (profiling, semantic inference, ...) must call this
    instead of parsing discovery.json themselves, so a missing, corrupted,
    or hand-edited file is rejected in one place instead of silently
    misinterpreted by each consumer.
    """
    discovery_path = Path(path)
    if not discovery_path.is_file():
        raise DiscoveryLoadError(f"discovery.json not found: {discovery_path}")

    try:
        with discovery_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DiscoveryLoadError(f"discovery.json is not valid JSON: {discovery_path}") from exc

    if not isinstance(data, dict):
        raise DiscoveryLoadError(f"discovery.json did not parse to an object: {discovery_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DiscoveryLoadError(f"discovery.json missing required keys: {sorted(missing)}")

    if not isinstance(data["tables"], dict):
        raise DiscoveryLoadError("discovery.json 'tables' must be an object")

    for table_name, table in data["tables"].items():
        if not isinstance(table, dict):
            raise DiscoveryLoadError(f"discovery.json table {table_name!r} must be an object")
        missing_table_keys = _REQUIRED_TABLE_KEYS - table.keys()
        if missing_table_keys:
            raise DiscoveryLoadError(
                f"discovery.json table {table_name!r} missing keys: {sorted(missing_table_keys)}"
            )

    fingerprint = data["source_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
        raise DiscoveryLoadError("discovery.json 'source_fingerprint' must be a 'sha256:<hex>' string")

    return data
