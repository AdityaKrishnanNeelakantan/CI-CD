"""Mandatory fail-closed quality gate for document-to-relational extraction."""
from __future__ import annotations

import pandas as pd

from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.domain.extraction.quality import ExtractionCheck, ExtractionQualityReport
from synth_platform.domain.extraction.target_schema import FieldType, TargetSchema
from synth_platform.errors import ExtractionQualityError


def validate_extraction(
    document,
    dataset: RelationalDataset,
    target_schema: TargetSchema,
    *,
    minimum_required_coverage: float = 0.80,
    maximum_empty_row_rate: float = 0.50,
    maximum_row_explosion_factor: float = 10.0,
    raise_on_failure: bool = False,
) -> ExtractionQualityReport:
    checks: list[ExtractionCheck] = []
    content_ok = document.meaningful_character_count > 0 or bool(document.tables)
    checks.append(ExtractionCheck(name="meaningful_document_content", passed=content_ok,
                                  detail="native/OCR text or tables must exist"))
    image_only_pages = [page for page in getattr(document, "pages", [])
                        if page.image_count > 0 and not page.text_blocks]
    has_ocr = any(getattr(line, "provenance", "native") == "ocr"
                  for line in getattr(document, "lines", []))
    checks.append(ExtractionCheck(
        name="image_content_handled",
        passed=not image_only_pages or has_ocr,
        critical=True,
        detail="image-only pages require successful OCR"))
    table_evidence_rows = sum(max(0, len(t.rows) - 1) for t in document.tables)
    text_evidence_rows = sum(1 for line in getattr(document, "lines", []) if line.text.strip())
    total_source_rows = table_evidence_rows + text_evidence_rows
    extracted_rows = sum(len(frame) for frame in dataset.tables.values())
    explosion_limit = max(10, int(maximum_row_explosion_factor * max(1, total_source_rows)))
    checks.append(ExtractionCheck(
        name="row_explosion", passed=extracted_rows <= explosion_limit,
        metric=float(extracted_rows), threshold=float(explosion_limit),
        detail="extracted rows must remain bounded by source evidence"))

    for entity in target_schema.entities:
        frame = dataset.tables.get(entity.name, pd.DataFrame())
        entity_present = len(frame) > 0
        checks.append(ExtractionCheck(name=f"entity_present[{entity.name}]",
                                      passed=entity_present,
                                      detail="required entity has at least one matched row"))
        if len(frame):
            data_cols = [field.name for field in entity.fields if field.name in frame.columns]
            if data_cols:
                empty_rate = float(frame[data_cols].isna().all(axis=1).mean())
                checks.append(ExtractionCheck(
                    name=f"nonempty_rows[{entity.name}]",
                    passed=empty_rate <= maximum_empty_row_rate,
                    metric=empty_rate, threshold=maximum_empty_row_rate))
                duplicate_rate = float(frame.duplicated(data_cols).mean())
                checks.append(ExtractionCheck(
                    name=f"duplicate_rows[{entity.name}]",
                    passed=duplicate_rate <= 0.20,
                    metric=duplicate_rate, threshold=0.20))
        for field in entity.fields:
            coverage = (float(frame[field.name].notna().mean())
                        if len(frame) and field.name in frame else 0.0)
            if field.required:
                checks.append(ExtractionCheck(
                    name=f"required_coverage[{entity.name}.{field.name}]",
                    passed=coverage >= minimum_required_coverage,
                    metric=coverage, threshold=minimum_required_coverage))
            if field.type in {FieldType.INTEGER, FieldType.NUMBER, FieldType.BOOLEAN, FieldType.DATE}:
                checks.append(ExtractionCheck(
                    name=f"type_coercion[{entity.name}.{field.name}]",
                    passed=coverage >= (minimum_required_coverage if field.required else 0.0),
                    critical=field.required, metric=coverage,
                    threshold=(minimum_required_coverage if field.required else 0.0)))
        if entity.parent and len(frame):
            parent = dataset.tables.get(entity.parent.entity, pd.DataFrame())
            parent_entity = target_schema.entity(entity.parent.entity)
            parent_pk = parent_entity.primary_key if parent_entity else None
            valid = bool(parent_pk and parent_pk in parent and
                         entity.parent.fk_field in frame and
                         frame[entity.parent.fk_field].isin(set(parent[parent_pk])).all())
            checks.append(ExtractionCheck(
                name=f"parent_references[{entity.name}]", passed=valid))

    blocking = [check.name for check in checks if check.critical and not check.passed]
    report = ExtractionQualityReport(checks=checks, passed=not blocking,
                                     blocking_reasons=blocking)
    if raise_on_failure and blocking:
        raise ExtractionQualityError(
            "extraction quality gate failed: " + ", ".join(blocking))
    return report
