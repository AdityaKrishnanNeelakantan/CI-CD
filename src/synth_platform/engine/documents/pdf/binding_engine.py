"""Binds each template field/table-column to a semantic role and a
generation strategy - the document-level analogue of the structured
pipeline's LearningPlanner (Plan stage): given what a field IS (semantic
role, from src/documents/field_semantics.py) and what SHAPE its value had
(never the value itself), decide how a later, source-disconnected runtime
should synthesize a brand-new one.

Operates purely on the already-persisted, already-redacted
document_template.json - this stage never touches the source file or any
raw value again, only value_type/label/shape_pattern that template
compilation already reduced the real value to.

A table binds one semantic role per COLUMN (majority value_type across
that column's cells), not per cell - a column is one repeated kind of
data, even though each cell's own shape_pattern (already in the template)
still varies row to row and is used as-is at generation time.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from synth_platform.engine.documents.pdf.field_semantics import infer_field_semantics

GENERATOR_STRATEGY_BY_ROLE: dict[str, str] = {
    "person_name": "fake_person_name",
    "email_address": "fake_email",
    "phone_number": "fake_phone_number",
    "street_address": "fake_street_address",
    "organization": "fake_organization",
    "account_number": "shaped_numeric_id",
    "reference_code": "shaped_alphanumeric_id",
    "currency_amount": "bounded_decimal",
    "date": "bounded_date",
    "biological_sex": "categorical_choice",
    "smoking_status": "categorical_choice",
    "alcohol_consumption": "categorical_choice",
    "diet_preference": "categorical_choice",
    "exercise_habits": "categorical_choice",
    "transaction_description": "fake_transaction_description",
    "medical_reason": "fake_medical_reason",
    "medication_name": "fake_medication_name",
    "dosage": "fake_dosage",
    "frequency": "fake_frequency",
    "narrative_text": "fake_narrative",
    "generic_numeric": "shaped_numeric_id",
    # Never shape-fill random letters for open text — that is the root of
    # gibberish labels/cells ("Udaxihh", "Yzfwnk"). Use Faker prose instead.
    "generic_text": "fake_generic_text",
}


def _header_label_for_column(region: dict[str, Any], col_index: int) -> str | None:
    """Prefer the last leading header row's cell text as the column label.

    Multi-row preambles often put a section title in row 0 and the real
    column headers ("Date", "Description") in a later header row — using
    the last header row matches what a human reads as the column name.
    """
    header_row_count = int(region.get("header_row_count") or 0)
    if header_row_count <= 0:
        return None
    header_row = region["rows"][header_row_count - 1]
    if col_index >= len(header_row):
        return None
    text = header_row[col_index].get("text")
    if text is None:
        return None
    cleaned = str(text).strip()
    return cleaned or None


def _bind_field(region: dict[str, Any]) -> dict[str, Any]:
    semantics = infer_field_semantics(region.get("label"), region["value_type"])
    return {
        "region_id": region["region_id"],
        "binding_type": "field",
        "label": region.get("label"),
        "semantic_role": semantics["semantic_role"],
        "semantic_status": semantics["status"],
        "semantic_confidence": semantics["confidence"],
        "generator_strategy": GENERATOR_STRATEGY_BY_ROLE[semantics["semantic_role"]],
    }


def _bind_table(region: dict[str, Any]) -> dict[str, Any]:
    header_row_count = region.get("header_row_count", 0)
    data_rows = region["rows"][header_row_count:]
    columns = []
    for col_index in range(region["column_count"]):
        column_value_types = [
            row[col_index]["value_type"] for row in data_rows if col_index < len(row)
        ]
        majority_value_type = (
            Counter(column_value_types).most_common(1)[0][0] if column_value_types else "text"
        )
        # Column headers are first-class semantic signal — without them,
        # Description/Reason columns fall through to value_type alone and
        # Title-Case phrases used to become person_name (Faker names).
        header_label = _header_label_for_column(region, col_index)
        semantics = infer_field_semantics(header_label, majority_value_type)
        columns.append(
            {
                "column_index": col_index,
                "label": header_label,
                "semantic_role": semantics["semantic_role"],
                "semantic_status": semantics["status"],
                "semantic_confidence": semantics["confidence"],
                "generator_strategy": GENERATOR_STRATEGY_BY_ROLE[semantics["semantic_role"]],
            }
        )
    return {
        "region_id": region["region_id"],
        "binding_type": "table",
        "row_count": region["row_count"],
        "column_count": region["column_count"],
        "columns": columns,
    }


def _bind_inline_spans(region: dict[str, Any]) -> dict[str, Any]:
    spans = []
    for span in region["inline_variable_spans"]:
        semantics = infer_field_semantics(None, span["value_type"])
        spans.append(
            {
                "start": span["start"],
                "end": span["end"],
                "semantic_role": semantics["semantic_role"],
                "semantic_status": semantics["status"],
                "semantic_confidence": semantics["confidence"],
                "generator_strategy": GENERATOR_STRATEGY_BY_ROLE[semantics["semantic_role"]],
            }
        )
    return {"region_id": region["region_id"], "binding_type": "inline_spans", "spans": spans}


def bind_document_template(template: dict[str, Any]) -> dict[str, Any]:
    bindings = []
    for page in template["pages"]:
        for region in page["regions"]:
            if region["region_type"] == "field":
                bindings.append(_bind_field(region))
            elif region["region_type"] == "table":
                bindings.append(_bind_table(region))
            elif region.get("inline_variable_spans"):
                # A heading/paragraph is static template wording overall,
                # but still carries redacted sensitive spans (see
                # src/documents/template_compiler.py) that need their own
                # binding so the render stage can splice in a fresh
                # generated value instead of the literal masked
                # placeholder that's sitting in region["text"].
                bindings.append(_bind_inline_spans(region))

    return {"page_count": template["page_count"], "bindings": bindings}
