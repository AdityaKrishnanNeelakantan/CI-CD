"""Tests for schema-generic platform behavior."""

from __future__ import annotations

import pandas as pd

from synth_platform.engine.generation.schema.demos.banking import build_banking_demo_schema
from synth_platform.engine.validation.schema.format_checks import invalid_typed_mask
from synth_platform.engine.generation.schema.pii_columns import infer_value_semantic, resolve_column_semantic
from synth_platform.engine.profiling.schema.profiler import infer_relationships
from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table
from synth_platform.engine.validation.schema.validation import DataValidator


def test_typed_format_validation_uses_schema_type_not_column_name():
    """A generically named column with schema type ssn is validated as SSN."""
    df = pd.DataFrame({"notes": ["123-45-6789", "000-00-0000", "111-22-3333"]})
    schema = SchemaConfig(
        name="generic-ssn-test",
        tables=[Table(name="people", row_count=3)],
        columns={"people": [Column(name="notes", type="ssn")]},
    )
    report = DataValidator({"people": df}, schema).validate_all()
    typed_issues = [i for i in report.issues if "typed-format" in i.message]
    assert typed_issues
    assert any(i.affected_rows >= 1 for i in typed_issues)


def test_name_only_account_column_is_not_validated_without_schema_type():
    values = pd.Series(["hello", "world", "foo"])
    assert invalid_typed_mask("notes", values) is None
    assert invalid_typed_mask("account_notes", values) is None


def test_infer_relationships_from_id_columns():
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["a", "b", "c"]})
    orders = pd.DataFrame({"order_id": [10, 11], "customer_id": [1, 2], "amount": [5.0, 6.0]})
    rels = infer_relationships([("customers", customers), ("orders", orders)])
    assert len(rels) == 1
    assert rels[0].parent_table == "customers"
    assert rels[0].child_table == "orders"
    assert rels[0].parent_key == "customer_id"
    assert rels[0].child_key == "customer_id"


def test_pii_semantics_include_cpf_and_nin():
    assert resolve_column_semantic("cpf") == "cpf"
    assert resolve_column_semantic("nin") == "nin"
    assert resolve_column_semantic("national_id") == "national_id"
    cpf_series = pd.Series(["123.456.789-01", "987.654.321-00"])
    assert infer_value_semantic("identifier", cpf_series) == "cpf"


def test_shared_banking_demo_schema_is_reusable():
    schema = build_banking_demo_schema(transactions=100, seed=7, country="US", llm_notes=False)
    assert schema.domain == "banking"
    assert len(schema.tables) == 3
    assert any(col.name == "case_note" for col in schema.get_columns("transactions"))
