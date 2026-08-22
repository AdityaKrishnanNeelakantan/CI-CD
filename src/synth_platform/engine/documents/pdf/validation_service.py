"""Checkpoint 9g orchestration: turns a rendered PDF twin + its ground
truth into a persisted document_validation_report.json - the Validate
stage of the PDF track, closing the loop the spec calls Render -> Validate.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.documents.pdf.errors import DocumentAdapterError
from synth_platform.engine.documents.pdf.render_validator import validate_rendered_document

STAGE_NAME = "document_validation"
VALIDATION_REPORT_FILENAME = "document_validation_report.json"

_REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "validated_at", "hard_checks_passed", "report"}


class DocumentValidationReportLoadError(Exception):
    """Raised when document_validation_report.json is missing, corrupted, or malformed."""


def run_document_validation(
    pdf_path: str | Path,
    ground_truth: dict[str, Any],
    doc_id: str,
    manifest: RunManifest,
    ground_truth_reference: str,
) -> StageResult:
    output_path = manifest.output_path(f"documents/{doc_id}/{VALIDATION_REPORT_FILENAME}")
    if output_path.exists():
        raise RuntimeError(
            f"{VALIDATION_REPORT_FILENAME} already exists for document {doc_id!r} in run "
            f"{manifest.run_id}; a stage output must never be overwritten. Start a new run instead."
        )

    try:
        report = validate_rendered_document(pdf_path, ground_truth)
    except DocumentAdapterError as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[ground_truth_reference],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    validation_report = {
        "doc_id": doc_id,
        "source_reference": ground_truth_reference,
        "validated_at": datetime.now(UTC).isoformat(),
        "hard_checks_passed": report["hard_checks_passed"],
        "report": report,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, sort_keys=True)

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[ground_truth_reference],
        output_references=[str(output_path)],
        metrics={
            "field_accuracy": report["field_accuracy"],
            "total_fields": report["total_fields"],
            "matched_fields": report["matched_fields"],
            "overflow_failure_count": report["overflow_failure_count"],
        },
        warnings=[] if report["hard_checks_passed"] else ["hard_checks_failed"],
        evidence={"validated_at": validation_report["validated_at"]},
    )
    manifest.record_stage(result)
    return result


def load_document_validation_report(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a document_validation_report.json file."""
    report_path = Path(path)
    if not report_path.is_file():
        raise DocumentValidationReportLoadError(f"document_validation_report.json not found: {report_path}")

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DocumentValidationReportLoadError(
            f"document_validation_report.json is not valid JSON: {report_path}"
        ) from exc

    if not isinstance(data, dict):
        raise DocumentValidationReportLoadError(
            f"document_validation_report.json did not parse to an object: {report_path}"
        )

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DocumentValidationReportLoadError(
            f"document_validation_report.json missing required keys: {sorted(missing)}"
        )

    return data
