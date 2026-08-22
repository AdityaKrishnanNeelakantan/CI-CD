from __future__ import annotations

import pytest

from synth_platform.engine.documents.pdf.binding_engine import GENERATOR_STRATEGY_BY_ROLE, bind_document_template

pytestmark = pytest.mark.unit


def _template(regions):
    return {"page_count": 1, "pages": [{"page_number": 1, "regions": regions}]}


def _field_region(label, value_type, region_id="p1_r0", shape_pattern="NNNNNNN"):
    return {
        "region_id": region_id,
        "region_type": "field",
        "label": label,
        "value_type": value_type,
        "confidence": 0.9,
        "masked_preview": "**",
        "shape_pattern": shape_pattern,
    }


def _table_region(rows, region_id="p1_r1"):
    return {
        "region_id": region_id,
        "region_type": "table",
        "row_count": len(rows),
        "column_count": max((len(r) for r in rows), default=0),
        "rows": rows,
    }


def test_field_binding_carries_semantic_role_and_strategy():
    template = _template([_field_region("Account Number", "numeric")])
    binding = bind_document_template(template)
    assert len(binding["bindings"]) == 1
    b = binding["bindings"][0]
    assert b["binding_type"] == "field"
    assert b["semantic_role"] == "account_number"
    assert b["generator_strategy"] == GENERATOR_STRATEGY_BY_ROLE["account_number"]


def test_every_semantic_role_has_a_registered_generator_strategy():
    from synth_platform.engine.documents.pdf.field_semantics import SEMANTIC_ROLES

    assert set(GENERATOR_STRATEGY_BY_ROLE) == SEMANTIC_ROLES


def test_table_binding_uses_header_labels_for_column_semantics():
    rows = [
        [
            {"value_type": "static_label", "text": "Date"},
            {"value_type": "static_label", "text": "Description"},
            {"value_type": "static_label", "text": "Amount"},
        ],
        [
            {"value_type": "date", "shape_pattern": "NN/NN/NNNN"},
            {"value_type": "text", "shape_pattern": "Ulllll Ullll"},
            {"value_type": "currency", "shape_pattern": "N.NN"},
        ],
        [
            {"value_type": "date", "shape_pattern": "NN/NN/NNNN"},
            {"value_type": "text", "shape_pattern": "Ullll Ulll"},
            {"value_type": "currency", "shape_pattern": "NN.NN"},
        ],
    ]
    region = _table_region(rows)
    region["header_row_count"] = 1
    template = _template([region])
    binding = bind_document_template(template)
    columns = binding["bindings"][0]["columns"]
    assert columns[0]["semantic_role"] == "date"
    assert columns[1]["semantic_role"] == "transaction_description"
    assert columns[1]["generator_strategy"] == GENERATOR_STRATEGY_BY_ROLE["transaction_description"]
    assert columns[2]["semantic_role"] == "currency_amount"


def test_table_binding_produces_one_column_entry_per_column():
    rows = [
        [{"value_type": "text", "shape_pattern": "Ulllll"}, {"value_type": "currency", "shape_pattern": "N.NN"}],
        [{"value_type": "text", "shape_pattern": "Ulllll"}, {"value_type": "currency", "shape_pattern": "NN.NN"}],
        [{"value_type": "text", "shape_pattern": "Ulllll"}, {"value_type": "currency", "shape_pattern": "N.NN"}],
    ]
    template = _template([_table_region(rows)])
    binding = bind_document_template(template)
    b = binding["bindings"][0]
    assert b["binding_type"] == "table"
    assert len(b["columns"]) == 2
    assert b["columns"][1]["semantic_role"] == "currency_amount"


def test_table_column_role_uses_majority_value_type_across_rows():
    rows = [
        [{"value_type": "currency", "shape_pattern": "N.NN"}],
        [{"value_type": "currency", "shape_pattern": "N.NN"}],
        [{"value_type": "text", "shape_pattern": "lll"}],  # one outlier, should not win majority
    ]
    template = _template([_table_region(rows)])
    binding = bind_document_template(template)
    assert binding["bindings"][0]["columns"][0]["semantic_role"] == "currency_amount"


def test_headings_and_paragraphs_produce_no_binding():
    template = _template(
        [
            {"region_id": "p1_r0", "region_type": "heading", "text": "Summary"},
            {"region_id": "p1_r1", "region_type": "paragraph", "text": "Some static wording."},
        ]
    )
    binding = bind_document_template(template)
    assert binding["bindings"] == []


def test_paragraph_with_inline_variable_spans_produces_a_span_binding():
    template = _template(
        [
            {
                "region_id": "p1_r0",
                "region_type": "paragraph",
                "text": "Beginning balance 69*96",
                "inline_variable_spans": [
                    {"start": 19, "end": 24, "value_type": "currency", "shape_pattern": "NN.NN"}
                ],
            }
        ]
    )
    binding = bind_document_template(template)
    assert len(binding["bindings"]) == 1
    b = binding["bindings"][0]
    assert b["binding_type"] == "inline_spans"
    assert b["spans"][0]["semantic_role"] == "currency_amount"
    assert b["spans"][0]["generator_strategy"] == GENERATOR_STRATEGY_BY_ROLE["currency_amount"]


def test_binding_never_carries_a_masked_preview_or_raw_value():
    template = _template([_field_region("Total Due", "currency")])
    binding = bind_document_template(template)
    serialized = str(binding)
    assert "masked_preview" not in serialized
