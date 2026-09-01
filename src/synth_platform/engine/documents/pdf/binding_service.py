"""Checkpoint 9d orchestration: turns a persisted document_template.json
into a persisted document_binding_map.json (runs/<run_id>/documents/<doc_id>/
document_binding_map.json).

Reads only the already-redacted template - no source file, no raw value,
ever touched here. Same relationship to template_service.py as cleaning
has to profiling: an independently-rerunnable stage over a prior stage's
output, not a function call chained inside it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_SUCCESS, StageResult
from synth_platform.engine.documents.pdf.binding_engine import bind_document_template

STAGE_NAME = "semantic_binding"
DOCUMENT_BINDING_MAP_FILENAME = "document_binding_map.json"

_REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "bound_at", "page_count", "bindings"}


class DocumentBindingMapLoadError(Exception):
    """Raised when document_binding_map.json is missing, corrupted, or malformed."""


def run_semantic_binding(
    template: dict[str, Any],
    doc_id: str,
    manifest: RunManifest,
    template_reference: str,
) -> StageResult:
    output_path = manifest.output_path(f"documents/{doc_id}/{DOCUMENT_BINDING_MAP_FILENAME}")
    if output_path.exists():
        raise RuntimeError(
            f"{DOCUMENT_BINDING_MAP_FILENAME} already exists for document {doc_id!r} in run "
            f"{manifest.run_id}; a stage output must never be overwritten. Start a new run instead."
        )

    binding = bind_document_template(template)

    review_required_count = sum(
        1
        for b in binding["bindings"]
        if (b.get("semantic_status") == "REVIEW_REQUIRED")
        or any(c.get("semantic_status") == "REVIEW_REQUIRED" for c in b.get("columns", []))
    )
    warnings = [f"review_required_bindings={review_required_count}"] if review_required_count else []

    document_binding_map = {
        "doc_id": doc_id,
        "source_reference": template_reference,
        "bound_at": datetime.now(UTC).isoformat(),
        "page_count": binding["page_count"],
        "bindings": binding["bindings"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(document_binding_map, f, indent=2, sort_keys=True)

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[template_reference],
        output_references=[str(output_path)],
        metrics={
            "binding_count": len(binding["bindings"]),
            "review_required_bindings": review_required_count,
        },
        warnings=warnings,
        evidence={"bound_at": document_binding_map["bound_at"]},
    )
    manifest.record_stage(result)
    return result


def load_document_binding_map(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a document_binding_map.json file."""
    map_path = Path(path)
    if not map_path.is_file():
        raise DocumentBindingMapLoadError(f"document_binding_map.json not found: {map_path}")

    try:
        with map_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DocumentBindingMapLoadError(f"document_binding_map.json is not valid JSON: {map_path}") from exc

    if not isinstance(data, dict):
        raise DocumentBindingMapLoadError(f"document_binding_map.json did not parse to an object: {map_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DocumentBindingMapLoadError(f"document_binding_map.json missing required keys: {sorted(missing)}")

    return data
