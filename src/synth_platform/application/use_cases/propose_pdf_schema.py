"""Deterministic evidence-based TargetSchema proposal for user confirmation."""
from __future__ import annotations

import re

from synth_platform.domain.extraction.target_schema import (
    ExtractionStrategy, FieldType, TargetEntity, TargetField, TargetSchema,
)


def _name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or fallback


def propose_pdf_schema(document) -> TargetSchema:
    entities: list[TargetEntity] = []
    for index, table in enumerate(document.tables):
        if not table.rows:
            continue
        headers = table.rows[0]
        fields = []
        used = set()
        for col_index, header in enumerate(headers):
            name = _name(str(header), f"column_{col_index + 1}")
            while name in used:
                name = f"{name}_{col_index + 1}"
            used.add(name)
            fields.append(TargetField(name=name, type=FieldType.STRING,
                                      aliases=[str(header)] if str(header).strip() else []))
        entity_name = _name(table.title or f"table_{index + 1}", f"table_{index + 1}")
        entities.append(TargetEntity(
            name=entity_name, primary_key=f"{entity_name}_id",
            fields=fields, aliases=[table.title] if table.title else [],
            strategy=ExtractionStrategy.ROWS_PER_TABLE_ROW))
    labels = []
    for line in document.lines:
        if ":" in line.text:
            label = line.text.split(":", 1)[0].strip()
            if label and label not in labels:
                labels.append(label)
    if labels:
        fields = [TargetField(name=_name(label, f"field_{i + 1}"),
                              type=FieldType.STRING, aliases=[label])
                  for i, label in enumerate(labels[:100])]
        entities.insert(0, TargetEntity(
            name="document", primary_key="document_id", fields=fields,
            strategy=ExtractionStrategy.SINGLETON))
    return TargetSchema(entities=entities)
