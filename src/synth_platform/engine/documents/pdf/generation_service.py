"""Checkpoint 9e orchestration: turns a document_template.json +
document_binding_map.json pair into persisted synthetic values
(runs/<run_id>/documents/<doc_id>/document_synthetic_values.json) - the
document-level analogue of the structured pipeline's Generate stage.

Reads only the already-redacted template and its binding map - no source
file, no raw value, ever touched here, which is exactly what makes this
stage able to run standalone, disconnected from the source PDF, on any
machine that has the two portable artifacts.

Seeded for the same reproducibility guarantee the structured pipeline's
generation stage has ("same seed reproduces output").
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.common.database.core.stage_result import STATUS_SUCCESS, StageResult
from synth_platform.engine.documents.pdf.value_generator import generate_document_values
from synth_platform.engine.generation.backends import GeneratorFactory

STAGE_NAME = "value_generation"
DOCUMENT_SYNTHETIC_VALUES_FILENAME = "document_synthetic_values.json"

_REQUIRED_TOP_LEVEL_KEYS = {"doc_id", "source_reference", "generated_at", "seed", "fields", "tables", "inline_spans"}


class DocumentSyntheticValuesLoadError(Exception):
    """Raised when document_synthetic_values.json is missing, corrupted, or malformed."""


def run_value_generation(
    template: dict[str, Any],
    binding_map: dict[str, Any],
    doc_id: str,
    manifest: RunManifest,
    binding_map_reference: str,
    seed: int | None = None,
    *,
    llm_text_enabled: bool = False,
    generation_backend: str = "current",
) -> StageResult:
    output_path = manifest.output_path(f"documents/{doc_id}/{DOCUMENT_SYNTHETIC_VALUES_FILENAME}")
    if output_path.exists():
        raise RuntimeError(
            f"{DOCUMENT_SYNTHETIC_VALUES_FILENAME} already exists for document {doc_id!r} in run "
            f"{manifest.run_id}; a stage output must never be overwritten. Start a new run instead."
        )

    if generation_backend == "current":
        generated = generate_document_values(
            template, binding_map, seed=seed, llm_text_enabled=bool(llm_text_enabled)
        )
    else:
        GeneratorFactory.create(generation_backend).generate_pdf(
            {
                "template": template,
                "binding_map": binding_map,
                "doc_id": doc_id,
                "manifest": manifest,
                "binding_map_reference": binding_map_reference,
            },
            seed=seed,
        )
        return StageResult(
            stage_name=STAGE_NAME,
            status=STATUS_SUCCESS,
            input_references=[binding_map_reference],
            output_references=[str(output_path)],
            metrics={"seed": seed, "generation_backend": generation_backend},
            evidence={"generated_by": generation_backend},
        )

    document_synthetic_values = {
        "doc_id": doc_id,
        "source_reference": binding_map_reference,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "fields": generated["fields"],
        "tables": generated["tables"],
        "inline_spans": generated["inline_spans"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(document_synthetic_values, f, indent=2, sort_keys=True)

    table_cell_count = sum(len(row) for rows in generated["tables"].values() for row in rows)

    result = StageResult(
        stage_name=STAGE_NAME,
        status=STATUS_SUCCESS,
        input_references=[binding_map_reference],
        output_references=[str(output_path)],
        metrics={
            "field_count": len(generated["fields"]),
            "table_count": len(generated["tables"]),
            "table_cell_count": table_cell_count,
            "seed": seed,
        },
        evidence={"generated_at": document_synthetic_values["generated_at"]},
    )
    manifest.record_stage(result)
    return result


def load_document_synthetic_values(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a document_synthetic_values.json file."""
    values_path = Path(path)
    if not values_path.is_file():
        raise DocumentSyntheticValuesLoadError(f"document_synthetic_values.json not found: {values_path}")

    try:
        with values_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DocumentSyntheticValuesLoadError(
            f"document_synthetic_values.json is not valid JSON: {values_path}"
        ) from exc

    if not isinstance(data, dict):
        raise DocumentSyntheticValuesLoadError(
            f"document_synthetic_values.json did not parse to an object: {values_path}"
        )

    missing = _REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise DocumentSyntheticValuesLoadError(
            f"document_synthetic_values.json missing required keys: {sorted(missing)}"
        )

    return data
