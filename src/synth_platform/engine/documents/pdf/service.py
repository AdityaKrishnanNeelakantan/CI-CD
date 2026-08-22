"""Checkpoint 9 orchestration: turns one document into recorded profiling evidence.

Mirrors the structured-pipeline services (discovery/profiling/inference):
reads the file through a DocumentAdapter, runs every detector, and writes
only derived evidence - never the raw extracted text - into
runs/<run_id>/documents/<doc_id>/document_profile.json. The document hash
index (for duplicate detection) is the one durable, cross-run store here,
same role as metadata/dataset_contract.json in the structured pipeline.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult
from synth_platform.engine.documents.pdf.archetypes.registry import resolve_archetype
from synth_platform.engine.documents.pdf.base import DocumentAdapter
from synth_platform.engine.documents.pdf.classifier import (
    STATUS_REVIEW_REQUIRED as CLASSIFICATION_REVIEW_REQUIRED,
)
from synth_platform.engine.documents.pdf.classifier import classify_document
from synth_platform.engine.documents.pdf.duplicates import (
    check_duplicate,
    load_document_index,
    save_document_index,
)
from synth_platform.engine.documents.pdf.entities import extract_entities
from synth_platform.engine.documents.pdf.errors import DocumentAdapterError
from synth_platform.engine.documents.pdf.extraction_router import extract_with_fallback
from synth_platform.engine.documents.pdf.language import STATUS_REVIEW_REQUIRED as LANGUAGE_REVIEW_REQUIRED
from synth_platform.engine.documents.pdf.language import detect_language
from synth_platform.engine.documents.pdf.pii import detect_pii
from synth_platform.engine.documents.pdf.text_stats import compute_text_statistics

STAGE_NAME = "document_profiling"
DOCUMENT_PROFILE_FILENAME = "document_profile.json"

_REQUIRED_TOP_LEVEL_KEYS = {
    "doc_id",
    "source_reference",
    "source_format",
    "profiled_at",
    "text_hash",
    "file_size_bytes",
    "pages",
    "extraction_warnings",
    "statistics",
    "language",
    "pii_findings",
    "entities",
    "classification",
    "archetype",
    "duplicate",
    "extraction_method",
    "pdf_classification",
    "ocr",
}


class DocumentProfileLoadError(Exception):
    """Raised when document_profile.json is missing, corrupted, or malformed."""


def run_document_profiling(
    adapter: DocumentAdapter,
    file_path: str | Path,
    doc_id: str,
    manifest: RunManifest,
    metadata_dir: str | Path,
    preferred_extraction_method: str | None = None,
) -> StageResult:
    output_path = manifest.output_path(f"documents/{doc_id}/{DOCUMENT_PROFILE_FILENAME}")
    if output_path.exists():
        raise RuntimeError(
            f"{DOCUMENT_PROFILE_FILENAME} already exists for document {doc_id!r} in run "
            f"{manifest.run_id}; a stage output must never be overwritten. Start a new run instead."
        )

    try:
        normalised = extract_with_fallback(
            adapter, file_path, preferred_method=preferred_extraction_method
        )
    except DocumentAdapterError as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[str(file_path)],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    text = normalised["text"]
    statistics = compute_text_statistics(text)
    language = detect_language(text)
    pii_findings = detect_pii(text)
    entity_findings = extract_entities(text)
    classification = classify_document(text)
    archetype = resolve_archetype(text, classification)

    try:
        index = load_document_index(metadata_dir)
        duplicate = check_duplicate(normalised["text_hash"], index)
        if not duplicate["is_duplicate"]:
            index[normalised["text_hash"]] = {
                "source_reference": normalised["source_reference"],
                "first_seen_run_id": manifest.run_id,
            }
            save_document_index(index, metadata_dir)
    except DocumentAdapterError as exc:
        result = StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_FAILED,
            input_references=[str(file_path)],
            output_references=[],
            errors=[str(exc)],
        )
        manifest.record_stage(result)
        return result

    warnings: list[str] = list(normalised["extraction_warnings"])
    if language["status"] == LANGUAGE_REVIEW_REQUIRED:
        warnings.append("language_review_required")
    if classification["status"] == CLASSIFICATION_REVIEW_REQUIRED:
        warnings.append("classification_review_required")
    if pii_findings:
        warnings.append(f"contains_pii_findings={len(pii_findings)}")
    if duplicate["is_duplicate"]:
        warnings.append(f"duplicate_of={duplicate['duplicate_of']}")

    document_profile = {
        "doc_id": doc_id,
        "source_reference": normalised["source_reference"],
        "source_format": normalised["source_format"],
        "profiled_at": datetime.now(UTC).isoformat(),
        "text_hash": normalised["text_hash"],
        "file_size_bytes": normalised["file_size_bytes"],
        "pages": normalised["pages"],
        "extraction_warnings": normalised["extraction_warnings"],
        "statistics": statistics,
        "language": language,
        "pii_findings": pii_findings,
        "entities": entity_findings,
        "classification": classification,
        "archetype": archetype,
        "duplicate": duplicate,
        "extraction_method": normalised["extraction_method"],
        "pdf_classification": normalised["pdf_classification"],
        "ocr": normalised["ocr"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(document_profile, f, indent=2, sort_keys=True)

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[str(file_path)],
        output_references=[str(output_path)],
        metrics={
            "char_count": statistics["char_count"],
            "word_count": statistics["word_count"],
            "pii_finding_count": len(pii_findings),
            "entity_count": len(entity_findings),
            "document_type": classification["document_type"],
            "language": language["language"],
            "extraction_method": normalised["extraction_method"],
        },
        warnings=warnings,
        evidence={"text_hash": normalised["text_hash"], "profiled_at": document_profile["profiled_at"]},
    )
    manifest.record_stage(result)
    return result


def load_document_profile(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a document_profile.json file."""
    profile_path = Path(path)
    if not profile_path.is_file():
        raise DocumentProfileLoadError(f"document_profile.json not found: {profile_path}")

    try:
        with profile_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DocumentProfileLoadError(f"document_profile.json is not valid JSON: {profile_path}") from exc

    if not isinstance(data, dict):
        raise DocumentProfileLoadError(f"document_profile.json did not parse to an object: {profile_path}")

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DocumentProfileLoadError(f"document_profile.json missing required keys: {sorted(missing)}")

    return data
