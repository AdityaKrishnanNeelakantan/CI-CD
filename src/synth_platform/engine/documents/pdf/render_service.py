"""Checkpoint 9f orchestration: turns a compiled template + binding map +
generated synthetic values into a rendered synthetic PDF twin plus its
ground-truth JSON (runs/<run_id>/documents/<doc_id>/rendered.pdf and
document_ground_truth.json) - the Render stage of the PDF track.

Reads only already-source-disconnected artifacts (template, binding map,
synthetic values - all already redacted/generated, never a raw value) and
writes a brand-new PDF from scratch via src/documents/pdf_renderer.py.
Never touches the source PDF.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_SUCCESS, StageResult
from synth_platform.engine.documents.pdf.pdf_renderer import render_document_pdf

STAGE_NAME = "document_rendering"
RENDERED_PDF_FILENAME = "rendered.pdf"
GROUND_TRUTH_FILENAME = "document_ground_truth.json"

_REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "rendered_at", "page_count", "regions"}


class DocumentGroundTruthLoadError(Exception):
    """Raised when document_ground_truth.json is missing, corrupted, or malformed."""


def run_document_rendering(
    template: dict[str, Any],
    binding_map: dict[str, Any],
    synthetic_values: dict[str, Any],
    doc_id: str,
    manifest: RunManifest,
    synthetic_values_reference: str,
) -> StageResult:
    pdf_output_path = manifest.output_path(f"documents/{doc_id}/{RENDERED_PDF_FILENAME}")
    ground_truth_output_path = manifest.output_path(f"documents/{doc_id}/{GROUND_TRUTH_FILENAME}")
    if pdf_output_path.exists() or ground_truth_output_path.exists():
        raise RuntimeError(
            f"rendered output already exists for document {doc_id!r} in run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    render_result = render_document_pdf(template, binding_map, synthetic_values, str(pdf_output_path))

    ground_truth = {
        "doc_id": doc_id,
        "source_reference": synthetic_values_reference,
        "rendered_at": datetime.now(UTC).isoformat(),
        "page_count": render_result["page_count"],
        "regions": render_result["regions"],
    }
    with ground_truth_output_path.open("w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, sort_keys=True)

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[synthetic_values_reference],
        output_references=[str(pdf_output_path), str(ground_truth_output_path)],
        metrics={
            "page_count": render_result["page_count"],
            "region_count": len(render_result["regions"]),
        },
        evidence={"rendered_at": ground_truth["rendered_at"]},
    )
    manifest.record_stage(result)
    return result


def load_document_ground_truth(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a document_ground_truth.json file."""
    gt_path = Path(path)
    if not gt_path.is_file():
        raise DocumentGroundTruthLoadError(f"document_ground_truth.json not found: {gt_path}")

    try:
        with gt_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DocumentGroundTruthLoadError(f"document_ground_truth.json is not valid JSON: {gt_path}") from exc

    if not isinstance(data, dict):
        raise DocumentGroundTruthLoadError(f"document_ground_truth.json did not parse to an object: {gt_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DocumentGroundTruthLoadError(f"document_ground_truth.json missing required keys: {sorted(missing)}")

    return data
