"""Checkpoint 9c orchestration: turns extracted geometry into a persisted
DocumentTemplateSpec (runs/<run_id>/documents/<doc_id>/document_template.json).

Consumes the extraction_method already decided by
src/documents/service.py's document_profile.json (native, Docling, or OCR) and
resolves it to a LayoutBackend (src/documents/layout_backends/registry.py)
to re-extract spans - re-extracting geometry rather than threading spans
through run_document_profiling(), because that stage's job is detection
evidence and this stage's job is reusable template evidence. Two
independently-rerunnable stages, same relationship as profiling ->
cleaning in the structured pipeline.

This function never branches on which backend produced the spans -
adding another backend only requires registering it in
src/documents/layout_backends/registry.py.

Only structural evidence and masked value previews are ever persisted -
see src/documents/template_compiler.py for the redaction rules.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.documents.pdf.errors import DocumentAdapterError, OCRUnavailableError
from synth_platform.engine.documents.pdf.layout_backends.registry import resolve_layout_backend
from synth_platform.engine.documents.pdf.layout_engine import analyze_layout
from synth_platform.engine.documents.pdf.native_layout import extract_acroform_field_spans
from synth_platform.engine.documents.pdf.preflight import run_pdf_preflight
from synth_platform.engine.documents.pdf.template_compiler import compile_document_template

STAGE_NAME = "template_compilation"
DOCUMENT_TEMPLATE_FILENAME = "document_template.json"

_REQUIRED_TOP_LEVEL_KEYS = {
    "doc_id",
    "source_reference",
    "compiled_at",
    "extraction_method",
    "page_count",
    "pages",
}


class DocumentTemplateLoadError(Exception):
    """Raised when document_template.json is missing, corrupted, or malformed."""


def run_template_compilation(
    file_path: str | Path,
    doc_id: str,
    manifest: RunManifest,
    extraction_method: str,
) -> StageResult:
    output_path = manifest.output_path(f"documents/{doc_id}/{DOCUMENT_TEMPLATE_FILENAME}")
    if output_path.exists():
        raise RuntimeError(
            f"{DOCUMENT_TEMPLATE_FILENAME} already exists for document {doc_id!r} in run "
            f"{manifest.run_id}; a stage output must never be overwritten. Start a new run instead."
        )

    try:
        spans = resolve_layout_backend(extraction_method).extract_spans(file_path)
        preflight = run_pdf_preflight(file_path)
        if preflight.get("has_form_fields"):
            # AcroForm field values live outside the page content stream
            # (invisible to both native extract_text() and OCR of the
            # rendered page), so they are read independently and merged in
            # here regardless of which extraction path ran - a form can
            # have both static printed text and fillable field values, and
            # a fillable form's real data must never be silently absent
            # from the compiled template.
            spans = spans + extract_acroform_field_spans(file_path)
    except (DocumentAdapterError, OCRUnavailableError) as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[str(file_path)],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    layout = analyze_layout(spans, preflight["page_count"], page_dimensions=preflight.get("page_dimensions"))
    template = compile_document_template(layout)

    field_count = sum(
        1 for page in template["pages"] for region in page["regions"] if region["region_type"] == "field"
    )
    table_count = sum(
        1 for page in template["pages"] for region in page["regions"] if region["region_type"] == "table"
    )

    warnings: list[str] = []
    if field_count == 0 and table_count == 0:
        warnings.append("no_fields_or_tables_detected")

    document_template = {
        "doc_id": doc_id,
        "source_reference": str(file_path),
        "compiled_at": datetime.now(UTC).isoformat(),
        "extraction_method": extraction_method,
        "page_count": template["page_count"],
        "pages": template["pages"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(document_template, f, indent=2, sort_keys=True)

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[str(file_path)],
        output_references=[str(output_path)],
        metrics={
            "page_count": template["page_count"],
            "field_count": field_count,
            "table_count": table_count,
        },
        warnings=warnings,
        evidence={"compiled_at": document_template["compiled_at"]},
    )
    manifest.record_stage(result)
    return result


def load_document_template(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a document_template.json file."""
    template_path = Path(path)
    if not template_path.is_file():
        raise DocumentTemplateLoadError(f"document_template.json not found: {template_path}")

    try:
        with template_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DocumentTemplateLoadError(f"document_template.json is not valid JSON: {template_path}") from exc

    if not isinstance(data, dict):
        raise DocumentTemplateLoadError(f"document_template.json did not parse to an object: {template_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DocumentTemplateLoadError(f"document_template.json missing required keys: {sorted(missing)}")

    return data
