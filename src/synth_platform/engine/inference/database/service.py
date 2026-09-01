"""Checkpoint 3 orchestration: turns discovery+profile evidence into
recorded semantic-inference proposals, and separately turns proposals plus
human decisions into an approved dataset contract.

These are two distinct stages, matching "human decisions are stored
separately from automated proposals": run_inference() only ever writes
proposals (runs/<run_id>/semantic_candidates.json); run_contract_approval()
is the only place that can mark a column "approved", and it writes to
metadata/ (the durable, cross-run source of truth) rather than the
run-scoped evidence directory.
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
from synth_platform.engine.inference.database.contract import (
    CONTRACT_FILENAME,
    build_or_update_dataset_contract,
    load_dataset_contract,
    save_dataset_contract,
)
from synth_platform.engine.inference.database.engine import STATUS_REVIEW_REQUIRED, SemanticInferenceEngine

INFERENCE_STAGE_NAME = "inference"
APPROVAL_STAGE_NAME = "contract_approval"
CANDIDATES_FILENAME = "semantic_candidates.json"

_REQUIRED_TOP_LEVEL_KEYS = {"generated_at", "tables"}
_REQUIRED_CANDIDATE_KEYS = {"column", "semantic_type", "status", "confidence", "evidence", "alternatives"}


class CandidatesLoadError(Exception):
    """Raised when semantic_candidates.json is missing, corrupted, or malformed."""


def run_inference(
    adapter: SourceAdapter,
    discovery_data: dict[str, Any],
    profile_data: dict[str, Any],
    manifest: RunManifest,
    discovery_reference: str,
    profile_reference: str,
    sample_limit: int,
    engine: SemanticInferenceEngine | None = None,
    chunk_size: int | None = None,
) -> StageResult:
    output_path = manifest.output_path(CANDIDATES_FILENAME)
    if output_path.exists():
        raise RuntimeError(
            f"{CANDIDATES_FILENAME} already exists for run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    engine = engine or SemanticInferenceEngine()
    candidates_by_table: dict[str, Any] = {}
    warnings: list[str] = []

    try:
        for table_name, table_schema in discovery_data["tables"].items():
            profile_table = profile_data.get("tables", {}).get(table_name)
            if chunk_size is not None:
                chunks = adapter.read_sample_chunks(table_name, limit=sample_limit, chunk_size=chunk_size)
                table_candidates = engine.infer_table_chunked(chunks, table_name, table_schema, profile_table)
            else:
                df = adapter.read_sample(table_name, limit=sample_limit)
                table_candidates = engine.infer_table(df, table_name, table_schema, profile_table)
            candidates_by_table[table_name] = table_candidates
            warnings.extend(
                f"{table_name}: {col}: REVIEW_REQUIRED"
                for col, candidate in table_candidates.items()
                if candidate["status"] == STATUS_REVIEW_REQUIRED
            )
    except SourceAdapterError as exc:
        result = StageResult(
            stage_name=INFERENCE_STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[discovery_reference, profile_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    candidates_data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tables": candidates_by_table,
    }
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(candidates_data, f, indent=2, sort_keys=True)
    except OSError as exc:
        output_path.unlink(missing_ok=True)
        result = StageResult(
            stage_name=INFERENCE_STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[discovery_reference, profile_reference],
            output_references=[],
            errors=[f"could not write {CANDIDATES_FILENAME}: {exc}"],
        )
        manifest.record_stage(result)
        return result

    total_columns = sum(len(cols) for cols in candidates_by_table.values())
    review_required_count = sum(
        1
        for cols in candidates_by_table.values()
        for c in cols.values()
        if c["status"] == STATUS_REVIEW_REQUIRED
    )

    result = StageResult(
        stage_name=INFERENCE_STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[discovery_reference, profile_reference],
        output_references=[str(output_path)],
        metrics={"total_columns": total_columns, "review_required_count": review_required_count},
        warnings=warnings,
        evidence={"generated_at": candidates_data["generated_at"]},
    )
    manifest.record_stage(result)
    return result


def run_contract_approval(
    dataset_id: str,
    source_fingerprint: str,
    discovery_data: dict[str, Any],
    candidates_by_table: dict[str, Any],
    manifest: RunManifest,
    metadata_dir: str | Path,
    candidates_reference: str,
    decisions: dict[str, dict[str, str]] | None = None,
) -> StageResult:
    existing_contract = load_dataset_contract(metadata_dir)

    contract = build_or_update_dataset_contract(
        dataset_id,
        source_fingerprint,
        discovery_data,
        candidates_by_table,
        existing_contract=existing_contract,
        decisions=decisions,
    )
    try:
        current_path = save_dataset_contract(contract, metadata_dir)
    except OSError as exc:
        (Path(metadata_dir) / CONTRACT_FILENAME).unlink(missing_ok=True)
        result = StageResult(
            stage_name=APPROVAL_STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[candidates_reference],
            output_references=[],
            errors=[f"could not write {CONTRACT_FILENAME}: {exc}"],
        )
        manifest.record_stage(result)
        return result
    snapshot_path = Path(metadata_dir) / "history" / f"dataset_contract.rev{contract['revision']}.json"

    status_counts: dict[str, int] = {}
    for table in contract["tables"].values():
        for column in table["columns"].values():
            status_counts[column["inference_status"]] = status_counts.get(column["inference_status"], 0) + 1

    result = StageResult(
        stage_name=APPROVAL_STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[candidates_reference],
        output_references=[str(current_path), str(snapshot_path)],
        metrics={"revision": contract["revision"], **status_counts},
        evidence={"dataset_id": dataset_id, "updated_at": contract["updated_at"]},
    )
    manifest.record_stage(result)
    return result


def load_candidates(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a semantic_candidates.json file."""
    candidates_path = Path(path)
    if not candidates_path.is_file():
        raise CandidatesLoadError(f"semantic_candidates.json not found: {candidates_path}")

    try:
        with candidates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise CandidatesLoadError(f"semantic_candidates.json is not valid JSON: {candidates_path}") from exc

    if not isinstance(data, dict):
        raise CandidatesLoadError(f"semantic_candidates.json did not parse to an object: {candidates_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise CandidatesLoadError(f"semantic_candidates.json missing required keys: {sorted(missing)}")

    for table_name, columns in data["tables"].items():
        if not isinstance(columns, dict):
            raise CandidatesLoadError(f"semantic_candidates.json table {table_name!r} must be an object")
        for column_name, candidate in columns.items():
            if not isinstance(candidate, dict):
                raise CandidatesLoadError(
                    f"semantic_candidates.json table {table_name!r} column {column_name!r} must be an object"
                )
            missing_keys = _REQUIRED_CANDIDATE_KEYS - candidate.keys()
            if missing_keys:
                raise CandidatesLoadError(
                    f"semantic_candidates.json table {table_name!r} column {column_name!r} "
                    f"missing keys: {sorted(missing_keys)}"
                )

    return data
