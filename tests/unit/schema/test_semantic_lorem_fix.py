"""Regression: bare text schemas must not emit lorem ipsum IDs/cities."""

from __future__ import annotations

from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table
from synth_platform.engine.inference.schema.semantic import enrich_schema_semantics, infer_relationships_from_column_names
from synth_platform.engine.generation.schema.simulator import DataSimulator


def _banking_like_schema() -> SchemaConfig:
    return SchemaConfig(
        name="schema",
        tables=[
            Table(name="customers", row_count=20),
            Table(name="accounts", row_count=20),
            Table(name="cards", row_count=20),
        ],
        columns={
            "customers": [
                Column(name="customer_id", type="text", unique=True, nullable=False),
                Column(name="first_name", type="text", nullable=True),
                Column(name="last_name", type="text", nullable=True),
                Column(name="email", type="text", nullable=True),
                Column(name="city", type="text", nullable=True),
                Column(name="credit_score", type="int", nullable=True),
            ],
            "accounts": [
                Column(name="account_id", type="text", unique=True, nullable=False),
                Column(name="customer_id", type="text", nullable=False),
            ],
            "cards": [
                Column(name="card_id", type="text", unique=True, nullable=False),
                Column(name="account_id", type="text", nullable=False),
                Column(name="card_type", type="text", nullable=True),
            ],
        },
        relationships=[],
    )


def _collect(sim: DataSimulator) -> dict:
    data = {}
    for table_name, batch_df in sim.generate_all():
        if table_name not in data:
            data[table_name] = batch_df
        else:
            import pandas as pd

            data[table_name] = pd.concat([data[table_name], batch_df], ignore_index=True)
    return data


def test_enrich_infers_relationships_and_semantics():
    enriched = enrich_schema_semantics(_banking_like_schema())
    assert any(
        r.parent_table == "customers" and r.child_table == "accounts" and r.child_key == "customer_id"
        for r in enriched.relationships
    )
    assert any(
        r.parent_table == "accounts" and r.child_table == "cards" and r.child_key == "account_id"
        for r in enriched.relationships
    )

    by_name = {c.name: c for c in enriched.columns["customers"]}
    assert by_name["city"].distribution_params.get("text_type") == "city"
    assert by_name["email"].distribution_params.get("text_type") == "email"
    assert by_name["customer_id"].distribution_params.get("text_type") == "uuid"
    assert by_name["credit_score"].type == "int"
    assert by_name["credit_score"].distribution_params.get("min") == 300

    accounts = {c.name: c for c in enriched.columns["accounts"]}
    assert accounts["customer_id"].type == "foreign_key"
    cards = {c.name: c for c in enriched.columns["cards"]}
    assert cards["account_id"].type == "foreign_key"
    assert cards["card_type"].type == "categorical"


def test_bare_text_schema_does_not_emit_lorem_ids_or_cities():
    schema = _banking_like_schema()
    schema.seed = 11
    data = _collect(DataSimulator(schema))

    cities = data["customers"]["city"].astype(str)
    assert not cities.str.contains(r"\b(?:lorem|ipsum|commodo|eiusmod|ullamco)\b", case=False, regex=True).any()
    assert cities.str.len().max() < 60

    for table, pk in (("customers", "customer_id"), ("accounts", "account_id"), ("cards", "card_id")):
        values = data[table][pk].astype(str)
        assert values.is_unique
        assert not values.str.contains(r"\b(?:lorem|ipsum|commodo|eiusmod)\b", case=False, regex=True).any()
        assert values.str.len().median() < 80

    assert set(data["accounts"]["customer_id"]).issubset(set(data["customers"]["customer_id"]))
    assert set(data["cards"]["account_id"]).issubset(set(data["accounts"]["account_id"]))

    scores = data["customers"]["credit_score"].dropna()
    assert scores.min() >= 300
    assert scores.max() <= 850


def test_infer_relationships_from_column_names_plural_tables():
    schema = _banking_like_schema()
    rels = infer_relationships_from_column_names(schema.tables, schema.columns)
    assert {(r.parent_table, r.child_table, r.child_key) for r in rels} >= {
        ("customers", "accounts", "customer_id"),
        ("accounts", "cards", "account_id"),
    }
