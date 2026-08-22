"""Trust and correctness tests for validation contract, IDs, and rules."""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.inference.schema.distribution_rules import (
    CrossTableRule,
    DistributionRuleError,
    WeightedRule,
    apply_distribution_rules,
    evaluate_distribution_rules,
    parse_distribution_rules,
    resolve_relationship_path,
)
from synth_platform.engine.validation.schema.export_validation import validate_single_table_export
from synth_platform.engine.generation.schema.pii_columns import generate_fresh_column_batch, is_text_heavy_column
from synth_platform.application.orchestration.schema.config import PipelineConfig
from synth_platform.application.orchestration.schema.schema_driven import run_schema_pipeline
from synth_platform.engine.inference.schema.schema import Column, Relationship, SchemaConfig, Table
from synth_platform.engine.validation.schema.validation_contract import ValidationStatus, build_validation_contract

pytest.importorskip("sdv")

from synth_platform.application.orchestration.schema.source_driven import run_source_pipeline


def _users_orders_schema(*, user_rows: int = 200, order_rows: int = 400) -> SchemaConfig:
    return SchemaConfig(
        name="commerce",
        seed=3,
        tables=[
            Table(name="users", row_count=user_rows),
            Table(name="orders", row_count=order_rows),
        ],
        columns={
            "users": [
                Column(name="user_id", type="int", unique=True, min=1, max=100000),
                Column(
                    name="segment",
                    type="categorical",
                    distribution_params={"choices": ["standard", "premium", "trial"]},
                ),
            ],
            "orders": [
                Column(name="order_id", type="int", unique=True, min=1, max=200000),
                Column(name="user_id", type="foreign_key", distribution_params={}),
                Column(
                    name="status",
                    type="categorical",
                    distribution_params={"choices": ["open", "priority", "closed"]},
                ),
            ],
        },
        relationships=[
            Relationship(
                parent_table="users",
                parent_key="user_id",
                child_table="orders",
                child_key="user_id",
            )
        ],
    )


def test_id_like_unique_across_chunks_uses_row_offset():
    source = pd.Series([1, 2, 3, 4, 5] * 10000)
    first = generate_fresh_column_batch("record_id", source, 25000, seed=1, unique=True, row_offset=0)
    second = generate_fresh_column_batch("record_id", source, 25000, seed=2, unique=True, row_offset=25000)
    combined = pd.concat([first, second], ignore_index=True)
    assert combined.nunique() == 50000


def test_text_heavy_column_detection():
    notes = pd.Series([f"Long narrative note number {i} with extra detail." for i in range(50)])
    assert is_text_heavy_column("notes", notes)


def test_cross_table_rule_parses_and_applies():
    raw = (
        "For orders linked to users where segment = premium, "
        "make 40% of orders.status = priority"
    )
    rules = parse_distribution_rules(raw)
    assert len(rules) == 1
    assert isinstance(rules[0], CrossTableRule)

    schema = _users_orders_schema(user_rows=100, order_rows=200)
    users = pd.DataFrame(
        {
            "user_id": range(100),
            "segment": ["premium"] * 40 + ["standard"] * 60,
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": range(200),
            "user_id": [i % 100 for i in range(200)],
            "status": ["open"] * 200,
        }
    )
    tables = apply_distribution_rules({"users": users, "orders": orders}, schema, rules, seed=1)
    evidence = evaluate_distribution_rules(tables, rules, schema=schema)
    assert evidence[0]["rule_type"] == "cross_table"
    assert evidence[0]["child_scope_rows"] > 0


def test_ambiguous_cross_table_path_rejected():
    schema = SchemaConfig(
        name="orphan",
        tables=[Table(name="a", row_count=10), Table(name="b", row_count=10)],
        columns={"a": [Column(name="a_id", type="int", unique=True)], "b": [Column(name="b_id", type="int", unique=True)]},
    )
    with pytest.raises(DistributionRuleError):
        resolve_relationship_path(schema, "a", "b")


def test_weighted_rule_parses_with_multiplier_evidence():
    raw = "Rows with amount >= 1000 should be 2x more likely to have flag = true"
    rules = parse_distribution_rules(raw)
    assert len(rules) == 1
    assert isinstance(rules[0], WeightedRule)
    df = pd.DataFrame({"amount": [500, 1500, 2000, 800], "flag": [False, False, False, False]})
    schema = SchemaConfig(
        name="weighted",
        tables=[Table(name="source_table", row_count=4)],
        columns={"source_table": [Column(name="amount", type="float"), Column(name="flag", type="boolean")]},
    )
    tables = apply_distribution_rules({"source_table": df}, schema, rules, seed=9)
    evidence = evaluate_distribution_rules(tables, rules, schema=schema)
    assert evidence[0]["rule_type"] == "weighted"
    assert evidence[0]["target_percent"] == "N/A"
    assert evidence[0]["multiplier_requested"] == 2.0


def test_sampled_advisory_does_not_force_hard_fail():
    contract = build_validation_contract(
        hard_checks={"has_errors": False, "row_count_passed": True, "fk_passed": False},
        advisory_fidelity={"passed": False, "note": "sampled"},
        export_validation={"passed": True, "pk_passed": True, "fk_passed": True},
        validation_scope="full_export",
        generation_mode="schema_driven",
    )
    assert contract.hard_validation_status == ValidationStatus.PASS
    assert contract.export_ready is True


def test_skipped_check_not_pass():
    contract = build_validation_contract(
        hard_checks={"has_errors": False, "row_count_passed": True},
        validation_scope="sampled",
        generation_mode="schema_driven",
    )
    assert contract.advisory_fidelity_status == ValidationStatus.NOT_AVAILABLE


def test_sampled_preview_without_export_checks_is_export_ready():
    contract = build_validation_contract(
        hard_checks={
            "has_errors": False,
            "row_count_passed": True,
            "pk_passed": None,
            "fk_passed": None,
        },
        validation_scope="sampled",
        generation_mode="schema_driven",
    )
    assert contract.hard_validation_status == ValidationStatus.PASS
    assert contract.export_ready is True


def test_schema_export_path_validation_overrides_sample_fk_fail(tmp_path):
    schema = _users_orders_schema(user_rows=80, order_rows=160)
    config = PipelineConfig(
        generation_mode="schema_driven",
        preview_only=False,
        preview_rows=30,
        full_rows=240,
        chunk_size=80,
        export_format="parquet",
        output_dir=tmp_path / "export_val",
        seed=5,
    )
    result = run_schema_pipeline(schema, config)
    assert result.export_validation.get("passed") is True
    assert result.validation_contract.get("hard_validation_status") == "PASS"


def test_source_driven_id_unique_at_scale(tmp_path):
    rows = 5000
    source = pd.DataFrame(
        {
            "record_id": list(range(rows)),
            "amount": [float(i) for i in range(rows)],
            "segment": (["a", "b", "c"] * (rows // 3 + 1))[:rows],
        }
    )
    config = PipelineConfig(
        generation_mode="source_driven",
        preview_only=False,
        preview_rows=100,
        full_rows=rows,
        chunk_size=1000,
        export_format="parquet",
        output_dir=tmp_path / "source_ids",
        seed=4,
        table_name="records",
    )
    result = run_source_pipeline(source, config)
    path = result.export_paths["records"]
    pk_check = validate_single_table_export(path, expected_rows=rows, pk_column="record_id")
    assert pk_check["pk_passed"] is True
    assert pk_check["distinct_pk"] == rows
