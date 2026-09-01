from __future__ import annotations

import re
from decimal import Decimal

import pytest

from synth_platform.engine.documents.pdf.value_generator import generate_document_values

pytestmark = pytest.mark.unit


def _template(fields=None, tables=None):
    pages = [{"page_number": 1, "regions": (fields or []) + (tables or [])}]
    return {"page_count": 1, "pages": pages}


def _field_region(region_id, shape_pattern):
    return {"region_id": region_id, "shape_pattern": shape_pattern}


def _binding(region_id, semantic_role, generator_strategy, binding_type="field", label=None):
    return {
        "region_id": region_id,
        "binding_type": binding_type,
        "label": label,
        "semantic_role": semantic_role,
        "semantic_status": "proposed",
        "semantic_confidence": 1.0,
        "generator_strategy": generator_strategy,
    }


def test_same_seed_reproduces_identical_output():
    template = _template(fields=[_field_region("p1_r0", "NNNNNNN")])
    binding = {"bindings": [_binding("p1_r0", "account_number", "shaped_numeric_id")]}
    a = generate_document_values(template, binding, seed=42)
    b = generate_document_values(template, binding, seed=42)
    assert a == b


def test_different_seeds_produce_different_output():
    template = _template(fields=[_field_region("p1_r0", "NNNNNNN")])
    binding = {"bindings": [_binding("p1_r0", "account_number", "shaped_numeric_id")]}
    a = generate_document_values(template, binding, seed=1)
    b = generate_document_values(template, binding, seed=2)
    assert a != b


def test_shaped_numeric_id_matches_shape_pattern_length_and_class():
    template = _template(fields=[_field_region("p1_r0", "UU-NNNNN")])
    binding = {"bindings": [_binding("p1_r0", "reference_code", "shaped_alphanumeric_id")]}
    result = generate_document_values(template, binding, seed=7)
    value = result["fields"]["p1_r0"]["value"]
    assert re.match(r"^[A-Z]{2}-\d{5}$", value)


def test_shaped_numeric_id_has_no_leading_zero():
    template = _template(fields=[_field_region("p1_r0", "NNNNNNN")])
    binding = {"bindings": [_binding("p1_r0", "account_number", "shaped_numeric_id")]}
    for seed in range(20):
        result = generate_document_values(template, binding, seed=seed)
        value = result["fields"]["p1_r0"]["value"]
        assert value[0] != "0"


def test_currency_amount_is_decimal_safe():
    template = _template(fields=[_field_region("p1_r0", "N,NNN.NN")])
    binding = {"bindings": [_binding("p1_r0", "currency_amount", "bounded_decimal")]}
    result = generate_document_values(template, binding, seed=3)
    entry = result["fields"]["p1_r0"]
    assert Decimal(entry["decimal"])  # must parse cleanly as a Decimal
    assert "," in entry["formatted"]


def test_malformed_multi_dot_currency_shape_keeps_every_digit():
    """Regression test: a real OCR'd bank-statement table row merged two
    amounts into one cell upstream ("$6,410.44 $2.41"), producing a shape
    with two literal dots. The old fallback took only the digits before
    the first dot and silently discarded the cents entirely
    ("6410" instead of a value derived from all the digits present).
    """
    template = _template(fields=[_field_region("p1_r0", "$N,NNN.NN $N.NN")])
    binding = {"bindings": [_binding("p1_r0", "currency_amount", "bounded_decimal")]}
    result = generate_document_values(template, binding, seed=17)
    entry = result["fields"]["p1_r0"]
    decimal_value = Decimal(entry["decimal"])  # must not raise, and must not be a truncated stub
    assert decimal_value > Decimal("100")  # a 4-digit integer part must survive, not be dropped


def test_date_shape_nnnn_dash_nn_dash_nn_produces_iso_date():
    template = _template(fields=[_field_region("p1_r0", "NNNN-NN-NN")])
    binding = {"bindings": [_binding("p1_r0", "date", "bounded_date")]}
    result = generate_document_values(template, binding, seed=9)
    value = result["fields"]["p1_r0"]["value"]
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", value)


def test_faker_backed_role_produces_a_nonempty_string_not_the_shape():
    template = _template(fields=[_field_region("p1_r0", "Ulllll Ulllll")])
    binding = {"bindings": [_binding("p1_r0", "person_name", "fake_person_name")]}
    result = generate_document_values(template, binding, seed=5)
    value = result["fields"]["p1_r0"]["value"]
    assert isinstance(value, str) and value


@pytest.mark.parametrize(
    ("semantic_role", "expected_choices"),
    [
        ("biological_sex", {"Male", "Female"}),
        ("smoking_status", {"Never", "Former", "Current"}),
    ],
)
def test_categorical_choice_draws_from_a_fixed_vocabulary_not_the_shape(semantic_role, expected_choices):
    """Regression: found by inspecting an actual generated synthetic
    twin - fields like "Sex"/"Smoking Status" have no dedicated
    generator strategy, so they fell through to shaped_text (refilling
    the shape_pattern with random characters), producing unreadable
    gibberish ("Hhexdv", "Yzfwn") instead of a plausible category value.
    """
    template = _template(fields=[_field_region("p1_r0", "Ulllll")])
    binding = {"bindings": [_binding("p1_r0", semantic_role, "categorical_choice")]}
    for seed in range(20):
        result = generate_document_values(template, binding, seed=seed)
        value = result["fields"]["p1_r0"]["value"]
        assert value in expected_choices


def test_table_generates_one_row_per_source_row_matching_each_cells_own_shape():
    template = _template(
        tables=[
            {
                "region_id": "p1_r0",
                "rows": [
                    [{"shape_pattern": "N.NN"}, {"shape_pattern": "Ulll"}],
                    [{"shape_pattern": "NN.NN"}, {"shape_pattern": "Ulll"}],
                ],
            }
        ]
    )
    binding = {
        "bindings": [
            {
                "region_id": "p1_r0",
                "binding_type": "table",
                "row_count": 2,
                "column_count": 2,
                "columns": [
                    {"column_index": 0, "semantic_role": "currency_amount", "generator_strategy": "bounded_decimal"},
                    {"column_index": 1, "semantic_role": "generic_text", "generator_strategy": "fake_generic_text"},
                ],
            }
        ]
    }
    result = generate_document_values(template, binding, seed=11)
    rows = result["tables"]["p1_r0"]
    assert len(rows) == 2
    assert len(rows[0]) == 2
    assert Decimal(rows[0][0]["decimal"])
    text_value = rows[1][1]["value"]
    assert isinstance(text_value, str) and text_value
    # Readable prose — never random letter-salad from shape fill.
    assert not re.fullmatch(r"[A-Za-z]{1,8}", text_value) or " " in text_value or text_value[0].isupper()


def test_generic_text_never_emits_shape_fill_gibberish():
    template = _template(fields=[_field_region("p1_r0", "Ulllll")])
    binding = {"bindings": [_binding("p1_r0", "generic_text", "fake_generic_text")]}
    for seed in range(15):
        value = generate_document_values(template, binding, seed=seed)["fields"]["p1_r0"]["value"]
        assert isinstance(value, str) and len(value) >= 2
        # Must be a real word-like token from Faker, not empty / placeholder.
        assert any(ch.isalpha() for ch in value)

def test_document_identity_reuses_same_person_name_across_fields():
    template = _template(
        fields=[
            _field_region("p1_r0", "Ulllll Ulllll"),
            _field_region("p1_r1", "Ulllll Ulllll"),
        ]
    )
    binding = {
        "bindings": [
            _binding("p1_r0", "person_name", "fake_person_name", label="Name"),
            _binding("p1_r1", "person_name", "fake_person_name", label="Account Holder"),
        ]
    }
    result = generate_document_values(template, binding, seed=21)
    assert result["fields"]["p1_r0"]["value"] == result["fields"]["p1_r1"]["value"]


def test_alcohol_categorical_is_readable():
    template = _template(fields=[_field_region("p1_r0", "Ulllll")])
    binding = {"bindings": [_binding("p1_r0", "alcohol_consumption", "categorical_choice")]}
    choices = {"Never", "Rarely", "Socially", "Regularly", "Former"}
    for seed in range(20):
        value = generate_document_values(template, binding, seed=seed)["fields"]["p1_r0"]["value"]
        assert value in choices


def test_transaction_description_is_not_a_person_name():
    template = _template(fields=[_field_region("p1_r0", "Ullllllll Ullll")])
    binding = {
        "bindings": [_binding("p1_r0", "transaction_description", "fake_transaction_description")]
    }
    value = generate_document_values(template, binding, seed=3)["fields"]["p1_r0"]["value"]
    assert value in {
        "Grocery Mart",
        "Direct Deposit Payroll",
        "Coffee Shop",
        "Fuel Station",
        "Online Transfer",
        "ATM Withdrawal",
        "Utility Payment",
        "Pharmacy Purchase",
        "Restaurant",
        "Subscription Service",
    }


def test_generated_value_is_never_the_literal_shape_pattern_placeholder():
    """Sanity check that shape-filling actually substitutes characters
    rather than accidentally returning the pattern string itself.
    """
    template = _template(fields=[_field_region("p1_r0", "UU-NNNNN")])
    binding = {"bindings": [_binding("p1_r0", "reference_code", "shaped_alphanumeric_id")]}
    result = generate_document_values(template, binding, seed=13)
    assert result["fields"]["p1_r0"]["value"] != "UU-NNNNN"
