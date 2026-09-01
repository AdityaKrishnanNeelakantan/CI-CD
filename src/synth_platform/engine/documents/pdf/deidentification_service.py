"""Mode 2 orchestration: turns a compiled template alone into a
de-identified/redacted PDF - runs/<run_id>/documents/<doc_id>/redacted.pdf
and document_deidentification_report.json.

Distinct from the Render stage (document_rendering, Mode 3/Synthetic
Twin): that stage fabricates brand-new synthetic values via
generation_service.py's Faker-backed generator. This stage fabricates
nothing - it draws each field/cell's own already-masked_preview (and each
heading/paragraph's own already-redacted text) straight from
document_template.json, through the exact same source-disconnected
renderer (src/documents/pdf_renderer.py), so the output is a
structurally faithful but fully de-identified copy of the original
document. This project supports three independent PII workflows: Mode 1
(detection only, document_profile.json's pii_findings/entities - see
src/documents/service.py), Mode 2 (this module), and Mode 3 (synthetic
twin - template_service.py/binding_service.py/generation_service.py/
render_service.py). They are deliberately not collapsed into one path.

Never touches the source PDF, never receives a binding map or synthetic
values - template_compilation's own redaction (template_compiler.py) is
the only PII-handling logic this stage depends on. Because it draws a
brand-new PDF from scratch (never edits the original content stream),
this sidesteps the classic "black box drawn over still-selectable text"
redaction flaw: the original text is never given to the renderer at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_SUCCESS, StageResult
from synth_platform.engine.documents.pdf.pdf_renderer import render_document_pdf

STAGE_NAME = "document_deidentification"
REDACTED_PDF_FILENAME = "redacted.pdf"
DEIDENTIFICATION_REPORT_FILENAME = "document_deidentification_report.json"

_REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "redacted_at", "page_count", "regions"}

# No bindings are needed to redact: src/documents/pdf_renderer.py's field
# drawing already tolerates a missing binding (falls back to {region_id}),
# so Mode 2 never depends on Mode 3's semantic_binding stage.
_EMPTY_BINDING_MAP: dict[str, Any] = {"bindings": []}


class DocumentDeidentificationReportLoadError(Exception):
    """Raised when document_deidentification_report.json is missing, corrupted, or malformed."""


def _masked_field_value(region: dict[str, Any]) -> dict[str, Any]:
    return {"label": region.get("label"), "value": region.get("masked_preview", "")}


def _masked_table_rows(region: dict[str, Any]) -> list[list[dict[str, Any]]]:
    rows_out = []
    for row in region.get("rows", []):
        row_out = []
        for cell in row:
            if cell.get("value_type") == "static_label":
                row_out.append({"value": cell.get("text", "")})
            else:
                row_out.append({"value": cell.get("masked_preview", "")})
        rows_out.append(row_out)
    return rows_out


def _build_masked_values(template: dict[str, Any]) -> dict[str, Any]:
    """Shaped exactly like generation_service.py's document_synthetic_values
    (fields/tables/inline_spans) so it can be handed to the same renderer
    unchanged - except every value here is the template's own already-
    redacted masked_preview/text, never a fabricated replacement. Heading/
    paragraph text needs no inline splicing: template_compiler.py already
    redacted it in place, so an empty inline_spans map leaves it as-is.
    """
    fields: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    for page in template["pages"]:
        for region in page["regions"]:
            if region["region_type"] == "field":
                fields[region["region_id"]] = _masked_field_value(region)
            elif region["region_type"] == "table":
                tables[region["region_id"]] = _masked_table_rows(region)
    return {"fields": fields, "tables": tables, "inline_spans": {}}


def run_document_deidentification(
    template: dict[str, Any],
    doc_id: str,
    manifest: RunManifest,
    template_reference: str,
) -> StageResult:
    output_path = manifest.output_path(f"documents/{doc_id}/{REDACTED_PDF_FILENAME}")
    report_output_path = manifest.output_path(f"documents/{doc_id}/{DEIDENTIFICATION_REPORT_FILENAME}")
    if output_path.exists() or report_output_path.exists():
        raise RuntimeError(
            f"redacted output already exists for document {doc_id!r} in run {manifest.run_id}; "
            "a stage output must never be overwritten. Start a new run instead."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked_values = _build_masked_values(template)
    render_result = render_document_pdf(template, _EMPTY_BINDING_MAP, masked_values, str(output_path))

    report = {
        "doc_id": doc_id,
        "source_reference": template_reference,
        "redacted_at": datetime.now(UTC).isoformat(),
        "page_count": render_result["page_count"],
        "regions": render_result["regions"],
    }
    with report_output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[template_reference],
        output_references=[str(output_path), str(report_output_path)],
        metrics={
            "page_count": render_result["page_count"],
            "region_count": len(render_result["regions"]),
        },
        evidence={"redacted_at": report["redacted_at"]},
    )
    manifest.record_stage(result)
    return result


def load_document_deidentification_report(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a document_deidentification_report.json file."""
    report_path = Path(path)
    if not report_path.is_file():
        raise DocumentDeidentificationReportLoadError(
            f"document_deidentification_report.json not found: {report_path}"
        )

    try:
        with report_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DocumentDeidentificationReportLoadError(
            f"document_deidentification_report.json is not valid JSON: {report_path}"
        ) from exc

    if not isinstance(data, dict):
        raise DocumentDeidentificationReportLoadError(
            f"document_deidentification_report.json did not parse to an object: {report_path}"
        )

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DocumentDeidentificationReportLoadError(
            f"document_deidentification_report.json missing required keys: {sorted(missing)}"
        )

    return data
