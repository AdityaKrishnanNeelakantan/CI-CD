"""Checkpoint 3: the approved dataset contract - the single source of
truth for semantic meaning that every later stage must read instead of
re-inferring its own interpretation.

metadata/dataset_contract.json is a living document (the "current approved
understanding"), distinct from the immutable per-run evidence files in
runs/<run_id>/. Every save also writes an immutable snapshot under
metadata/history/, so the current pointer can move forward without losing
the trail of what was approved when.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.inference.database.engine import STATUS_REVIEW_REQUIRED

CONTRACT_FORMAT_VERSION = "1.0"
CONTRACT_FILENAME = "dataset_contract.json"
SENSITIVE_SEMANTIC_TYPES = frozenset({"identifier", "email", "person_name", "phone_number"})

_REQUIRED_TOP_LEVEL_KEYS = {
    "contract_version",
    "revision",
    "dataset_id",
    "source_fingerprint",
    "updated_at",
    "tables",
}
_REQUIRED_TABLE_KEYS = {"primary_key", "foreign_keys", "columns", "business_rules"}
_REQUIRED_COLUMN_KEYS = {
    "physical_type",
    "semantic_type",
    "nullable",
    "sensitive",
    "inference_status",
    "confidence",
    "evidence",
    "alternatives",
}


class DatasetContractLoadError(Exception):
    """Raised when dataset_contract.json is missing, corrupted, or malformed."""


def _build_column_entry(
    physical_type: str,
    nullable: bool,
    semantic_type: str | None,
    inference_status: str,
    confidence: float | None,
    evidence: list[str],
    alternatives: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "physical_type": physical_type,
        "semantic_type": semantic_type,
        "nullable": nullable,
        "sensitive": semantic_type in SENSITIVE_SEMANTIC_TYPES,
        "inference_status": inference_status,
        "confidence": confidence,
        "evidence": evidence,
        "alternatives": alternatives,
    }


def build_or_update_dataset_contract(
    dataset_id: str,
    source_fingerprint: str,
    discovery_data: dict[str, Any],
    candidates_by_table: dict[str, dict[str, dict[str, Any]]],
    existing_contract: dict[str, Any] | None = None,
    decisions: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Merge fresh candidates with prior approvals and new human decisions.

    Rule: a column already marked "approved" in `existing_contract` is
    never silently replaced by a new candidate - only an explicit new
    decision for that exact column can change it. This is what makes an
    override survive a rerun of inference.
    """
    decisions = decisions or {}
    revision = (existing_contract.get("revision", 0) if existing_contract else 0) + 1

    tables_out: dict[str, Any] = {}
    for table_name, table_discovery in discovery_data["tables"].items():
        existing_table = ((existing_contract or {}).get("tables", {})).get(table_name, {})
        existing_columns = existing_table.get("columns", {})
        table_candidates = candidates_by_table.get(table_name, {})
        table_decisions = decisions.get(table_name, {})

        columns_out: dict[str, Any] = {}
        for column in table_discovery["columns"]:
            column_name = column["name"]
            physical_type = column["database_type"]
            nullable = column["nullable"]
            existing_entry = existing_columns.get(column_name)
            decision_value = table_decisions.get(column_name)
            candidate = table_candidates.get(column_name)

            if decision_value is not None:
                evidence = list((candidate or {}).get("evidence", []))
                evidence.append(
                    "human_accepted_proposal"
                    if candidate and candidate.get("semantic_type") == decision_value
                    else "human_override"
                )
                columns_out[column_name] = _build_column_entry(
                    physical_type,
                    nullable,
                    decision_value,
                    "approved",
                    (candidate or {}).get("confidence"),
                    evidence,
                    (candidate or {}).get("alternatives", []),
                )
            elif existing_entry is not None and existing_entry.get("inference_status") == "approved":
                columns_out[column_name] = existing_entry
            elif candidate is not None:
                columns_out[column_name] = _build_column_entry(
                    physical_type,
                    nullable,
                    candidate["semantic_type"],
                    "review_required" if candidate["status"] == STATUS_REVIEW_REQUIRED else "proposed",
                    candidate["confidence"],
                    candidate["evidence"],
                    candidate["alternatives"],
                )
            elif existing_entry is not None:
                columns_out[column_name] = existing_entry
            else:
                columns_out[column_name] = _build_column_entry(
                    physical_type, nullable, None, "review_required", 0.0, ["no_candidate_available"], []
                )

        tables_out[table_name] = {
            "primary_key": table_discovery.get("primary_key", []),
            "foreign_keys": table_discovery.get("foreign_keys", []),
            "columns": columns_out,
            "business_rules": existing_table.get("business_rules", []),
        }

    return {
        "contract_version": CONTRACT_FORMAT_VERSION,
        "revision": revision,
        "dataset_id": dataset_id,
        "source_fingerprint": source_fingerprint,
        "updated_at": datetime.now(UTC).isoformat(),
        "tables": tables_out,
    }


def save_dataset_contract(contract: dict[str, Any], metadata_dir: str | Path) -> Path:
    metadata_dir = Path(metadata_dir)
    history_dir = metadata_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    revision = contract["revision"]
    snapshot_path = history_dir / f"dataset_contract.rev{revision}.json"
    if snapshot_path.exists():
        raise RuntimeError(
            f"dataset_contract revision {revision} is already recorded in history; "
            "immutable history must never be overwritten."
        )
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, sort_keys=True)

    current_path = metadata_dir / CONTRACT_FILENAME
    with current_path.open("w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, sort_keys=True)
    return current_path


def _validate_contract_structure(data: Any, source_description: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise DatasetContractLoadError(f"{source_description} did not parse to an object")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DatasetContractLoadError(f"{source_description} missing required keys: {sorted(missing)}")

    if not isinstance(data["tables"], dict):
        raise DatasetContractLoadError(f"{source_description} 'tables' must be an object")

    for table_name, table in data["tables"].items():
        if not isinstance(table, dict):
            raise DatasetContractLoadError(f"{source_description} table {table_name!r} must be an object")
        missing_table_keys = _REQUIRED_TABLE_KEYS - table.keys()
        if missing_table_keys:
            raise DatasetContractLoadError(
                f"{source_description} table {table_name!r} missing keys: {sorted(missing_table_keys)}"
            )
        for column_name, column in table["columns"].items():
            if not isinstance(column, dict):
                raise DatasetContractLoadError(
                    f"{source_description} table {table_name!r} column {column_name!r} must be an object"
                )
            missing_column_keys = _REQUIRED_COLUMN_KEYS - column.keys()
            if missing_column_keys:
                raise DatasetContractLoadError(
                    f"{source_description} table {table_name!r} column {column_name!r} "
                    f"missing keys: {sorted(missing_column_keys)}"
                )

    return data


def load_dataset_contract(metadata_dir: str | Path) -> dict[str, Any] | None:
    """Load the current approved dataset contract, or None if it does not exist yet."""
    path = Path(metadata_dir) / CONTRACT_FILENAME
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DatasetContractLoadError(f"dataset_contract.json is not valid JSON: {path}") from exc

    return _validate_contract_structure(data, str(path))
