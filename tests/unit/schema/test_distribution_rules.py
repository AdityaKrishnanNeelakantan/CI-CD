"""Distribution rule chunk aggregation tests."""

from __future__ import annotations

import pandas as pd

from synth_platform.engine.inference.schema.distribution_rules import (
    GlobalRuleBudget,
    aggregate_rule_evidence,
    apply_distribution_rules_to_chunk,
    evaluate_distribution_rules,
    parse_distribution_rules,
)
from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table


def _items_schema(rows: int = 1000) -> SchemaConfig:
    return SchemaConfig(
        name="items",
        seed=1,
        tables=[Table(name="items", row_count=rows)],
        columns={
            "items": [
                Column(name="item_id", type="int", unique=True, min=1, max=100000),
                Column(name="status", type="categorical", distribution_params={"choices": ["open", "closed", "pending"]}),
            ]
        },
    )


def test_aggregate_rule_evidence_sums_rows():
    rules = parse_distribution_rules("40% of items.status = open")
    chunk_records = [
        evaluate_distribution_rules({"items": pd.DataFrame({"status": ["open"] * 20 + ["closed"] * 80})}, rules),
        evaluate_distribution_rules({"items": pd.DataFrame({"status": ["open"] * 30 + ["closed"] * 70})}, rules),
    ]
    aggregated = aggregate_rule_evidence(chunk_records, rules=rules, table_totals={"items": 200})
    assert aggregated[0]["actual_rows"] == 50
    assert aggregated[0]["target_rows"] == 80


def test_chunk_rule_budget_allocates_targets():
    rules = parse_distribution_rules("50% of items.status = open")
    budget = GlobalRuleBudget.from_rules(rules, {"items": 1000})
    first = budget.chunk_target(rules[0], rows_before=0, chunk_size=200)
    second = budget.chunk_target(rules[0], rows_before=200, chunk_size=200)
    assert first + second == 200


def test_apply_distribution_rules_to_chunk_validates_only_current_table():
    schema = SchemaConfig(
        name="multi",
        seed=1,
        tables=[Table(name="users", row_count=100), Table(name="orders", row_count=200)],
        columns={
            "users": [
                Column(name="user_id", type="int", unique=True, min=1, max=100),
                Column(name="segment", type="categorical", distribution_params={"choices": ["a", "b"]}),
            ],
            "orders": [
                Column(name="order_id", type="int", unique=True, min=1, max=200),
                Column(name="status", type="categorical", distribution_params={"choices": ["open", "priority"]}),
            ],
        },
    )
    rules = parse_distribution_rules(
        "30% of users.segment = premium\n40% of orders.status = priority"
    )
    budget = GlobalRuleBudget.from_rules(rules, {"users": 100, "orders": 200})
    users_chunk = pd.DataFrame({"user_id": range(50), "segment": ["a"] * 50})
    updated = apply_distribution_rules_to_chunk(
        users_chunk,
        schema,
        rules,
        budget,
        table_name="users",
        rows_before=0,
        seed=1,
    )
    assert len(updated) == 50


def test_apply_distribution_rules_to_chunk_is_table_scoped():
    schema = _items_schema(400)
    rules = parse_distribution_rules("25% of items.status = open")
    budget = GlobalRuleBudget.from_rules(rules, {"items": 400})
    chunk = pd.DataFrame({"item_id": range(100), "status": ["closed"] * 100})
    updated = apply_distribution_rules_to_chunk(
        chunk,
        schema,
        rules,
        budget,
        table_name="items",
        rows_before=0,
        seed=1,
    )
    assert (updated["status"] == "open").sum() >= 1
