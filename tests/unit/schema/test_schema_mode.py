"""Integration tests for the Schema Mode application facade."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.application.workflows.schema_twin import (
    SchemaModeResult,
    generate_from_schema,
    load_schema_bytes,
    package_download,
    summarize_schema,
    validation_highlights,
)
from synth_platform.application.orchestration.schema.result import PipelineResult
from synth_platform.engine.generation.schema.vocab_seeds import CITIES_BY_COUNTRY
from synth_platform.engine.inference.schema.schema import (
    Column,
    RealismConfig,
    Relationship,
    SchemaConfig,
    Table,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "schema" / "schema_twin_minimal.json"


def _minimal_schema(name: str) -> SchemaConfig:
    return SchemaConfig(
        name=name,
        tables=[Table(name="records", row_count=1)],
        columns={"records": [Column(name="id", type="int")]},
    )


def test_load_schema_bytes_from_flexible_json():
    schema = load_schema_bytes(FIXTURE.read_bytes(), FIXTURE.name, seed=7)
    assert schema.name == "schema_twin_minimal"
    assert schema.seed == 7
    assert {t.name for t in schema.tables} == {"users", "orders"}
    assert len(schema.relationships) == 1


def test_summarize_schema_counts_tables_and_columns():
    schema = load_schema_bytes(FIXTURE.read_bytes(), FIXTURE.name)
    summary = summarize_schema(schema)
    assert summary.table_count == 2
    assert summary.column_count == 6
    assert summary.relationship_count == 1


def test_generate_from_schema_reuses_pipeline_and_exports(tmp_path: Path):
    schema = load_schema_bytes(FIXTURE.read_bytes(), FIXTURE.name)
    result = generate_from_schema(
        schema,
        row_count=25,
        seed=11,
        locale="en_US",
        output_dir=tmp_path / "out",
        export_format="csv",
        preview_rows=10,
    )

    assert set(result.preview_tables) == {"users", "orders"}
    assert result.row_counts["users"] == 25
    assert result.row_counts["orders"] > result.row_counts["users"]
    assert result.export_paths["users"].exists()
    assert result.export_paths["orders"].exists()

    users = pd.read_csv(result.export_paths["users"])
    orders = pd.read_csv(result.export_paths["orders"])
    assert set(orders["user_id"]).issubset(set(users["user_id"]))

    highlights = validation_highlights(result)
    assert highlights["hard_checks_passed"] is True
    assert highlights["export_count"] == 2

    zip_bytes = package_download(result)
    with zipfile.ZipFile(__import__("io").BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    assert "users.csv" in names
    assert "orders.csv" in names
    assert "validation_report.json" in names


def test_schema_twin_generates_only_valid_city_country_tuples(tmp_path: Path):
    """Regression: independently declared City/Country categoricals stay linked."""
    schema = SchemaConfig(
        name="branch-geography",
        seed=17,
        tables=[Table(name="branches", row_count=1)],
        columns={
            "branches": [
                Column(
                    name="BranchId",
                    type="int",
                    unique=True,
                    distribution_params={"min": 1, "max": 10_000},
                ),
                Column(name="branch_name", type="text", distribution_params={}),
                Column(
                    name="City",
                    type="categorical",
                    distribution_params={
                        "choices": ["Fort Worth", "Dallas", "Austin", "Houston", "Los Angeles", "Chicago"]
                    },
                ),
                Column(
                    name="Country",
                    type="categorical",
                    distribution_params={"choices": ["India", "Canada", "France", "Germany", "Australia"]},
                ),
            ]
        },
    )

    result = generate_from_schema(
        schema,
        row_count=80,
        seed=17,
        output_dir=tmp_path / "geo",
        preview_rows=80,
    )
    branches = result.preview_tables["branches"]

    assert len(branches) == 80
    assert all(
        city in CITIES_BY_COUNTRY[country]
        for city, country in zip(branches["City"], branches["Country"])
    )
    assert all(
        str(branch_name).startswith(f"{city} ")
        for branch_name, city in zip(branches["branch_name"], branches["City"])
    )


def test_schema_twin_scales_child_rows_from_foreign_keys(tmp_path: Path):
    """Regression: the global setting is a base, not a forced flat count."""
    schema = SchemaConfig(
        name="parent-child-counts",
        seed=5,
        tables=[Table(name="parents", row_count=1), Table(name="children", row_count=1)],
        columns={
            "parents": [Column(name="parent_id", type="int", unique=True, distribution_params={})],
            "children": [
                Column(name="child_id", type="int", unique=True, distribution_params={}),
                Column(name="parent_id", type="foreign_key", distribution_params={}),
            ],
        },
        relationships=[
            Relationship(
                parent_table="parents",
                parent_key="parent_id",
                child_table="children",
                child_key="parent_id",
            )
        ],
        realism=RealismConfig(relationship_multipliers={"parents->children": 3.0}),
    )

    result = generate_from_schema(
        schema,
        row_count=12,
        seed=5,
        output_dir=tmp_path / "counts",
        preview_rows=12,
    )

    assert result.row_counts == {"parents": 12, "children": 36}
    assert set(result.preview_tables["children"]["parent_id"]).issubset(
        set(result.preview_tables["parents"]["parent_id"])
    )


def test_schema_twin_infers_fintech_defaults_for_banking_constellation(tmp_path: Path):
    schema = SchemaConfig(
        name="banking-defaults",
        seed=21,
        tables=[
            Table(name="customers", row_count=1),
            Table(name="accounts", row_count=1),
            Table(name="loans", row_count=1),
            Table(name="transactions", row_count=1),
        ],
        columns={
            "customers": [Column(name="customer_id", type="text", unique=True)],
            "accounts": [
                Column(name="account_id", type="text", unique=True),
                Column(name="account_type", type="text"),
                Column(name="balance_usd", type="float"),
            ],
            "loans": [
                Column(name="loan_id", type="text", unique=True),
                Column(name="loan_amount", type="float"),
                Column(name="interest_rate", type="float"),
            ],
            "transactions": [
                Column(name="transaction_id", type="text", unique=True),
                Column(name="amount_usd", type="float"),
            ],
        },
    )

    result = generate_from_schema(
        schema,
        row_count=50,
        seed=21,
        output_dir=tmp_path / "banking",
        preview_rows=50,
    )

    accounts = result.preview_tables["accounts"]
    loans = result.preview_tables["loans"]
    assert result.schema.domain == "fintech"
    assert set(accounts["account_type"]) <= {
        "checking",
        "savings",
        "money_market",
        "certificate_of_deposit",
    }
    assert loans["interest_rate"].between(0.01, 0.36).all()
    assert loans["loan_amount"].ge(1_000).all()


@pytest.mark.parametrize(
    "validation_report",
    [
        {},
        {"summary": "validation output unavailable"},
        {"status": "not_run"},
        {"status": "unknown"},
        ["unexpected"],  # type: ignore[list-item]
    ],
)
def test_schema_mode_hard_checks_fail_closed_for_missing_or_malformed_validation(validation_report):
    result = SchemaModeResult(
        schema=_minimal_schema("empty_validation"),
        pipeline=PipelineResult(generation_mode="schema_driven", validation_report=validation_report),
    )

    assert result.hard_checks_passed is False
    assert validation_highlights(result)["hard_checks_passed"] is False


@pytest.mark.parametrize(
    "validation_report",
    [
        {"hard_checks_passed": True},
        {"passed": True},
        {"status": "passed"},
    ],
)
def test_schema_mode_hard_checks_still_accept_explicit_pass_signals(validation_report):
    result = SchemaModeResult(
        schema=_minimal_schema("valid_validation"),
        pipeline=PipelineResult(generation_mode="schema_driven", validation_report=validation_report),
    )

    assert result.hard_checks_passed is True


def test_load_yaml_fixture_via_facade():
    yaml_path = Path(__file__).resolve().parents[2] / "fixtures" / "schema" / "minimal_company.yaml"
    schema = load_schema_bytes(yaml_path.read_bytes(), yaml_path.name, seed=3)
    assert schema.name == "minimal_company"
    assert schema.seed == 3
    assert schema.tables[0].name == "users"


def test_load_sql_ddl_bytes_via_facade():
    """Generic SQL DDL ingest — CREATE TABLE, not format-specific filenames."""
    ddl = b"""
    CREATE TABLE parent (
      parent_id VARCHAR(20) PRIMARY KEY,
      label VARCHAR(50)
    );
    CREATE TABLE child (
      child_id VARCHAR(20) PRIMARY KEY,
      parent_id VARCHAR(20) NOT NULL,
      amount DECIMAL(10,2),
      FOREIGN KEY (parent_id) REFERENCES parent(parent_id)
    );
    """
    schema = load_schema_bytes(ddl, "customer_schema.sql", seed=19)
    assert schema.seed == 19
    assert schema.name == "customer_schema"
    assert {t.name for t in schema.tables} == {"parent", "child"}
    assert any(
        r.parent_table == "parent" and r.child_table == "child" for r in schema.relationships
    )


def test_generate_from_schema_caps_chunk_size_for_streaming(tmp_path: Path, monkeypatch):
    """Large row_count must not set chunk_size >= total rows (that disabled streaming)."""
    from synth_platform.application.orchestration.schema import schema_driven as schema_driven_mod

    captured: dict = {}

    original = schema_driven_mod.run_schema_pipeline

    def _capture(schema, config):
        captured["chunk_size"] = config.chunk_size
        captured["full_rows"] = config.full_rows
        return original(schema, config)

    monkeypatch.setattr(schema_driven_mod, "run_schema_pipeline", _capture)
    # schema_mode imports run_schema_pipeline into its namespace — patch there too
    import synth_platform.application.workflows.schema_twin as schema_mode_mod

    monkeypatch.setattr(schema_mode_mod, "run_schema_pipeline", _capture)

    schema = load_schema_bytes(FIXTURE.read_bytes(), FIXTURE.name)
    generate_from_schema(
        schema,
        row_count=50_000,
        seed=1,
        output_dir=tmp_path / "out",
        export_format="csv",
        preview_rows=5,
    )
    assert captured["chunk_size"] == 10_000
    assert captured["chunk_size"] < captured["full_rows"]


def test_facade_rejects_unsupported_schema_extension():
    with pytest.raises(ValueError, match=r"\.sql"):
        load_schema_bytes(b"not a schema", "schema.txt")


def test_facade_does_not_accept_non_mapping_json_root():
    with pytest.raises(ValueError, match="JSON object"):
        load_schema_bytes(json.dumps([1, 2, 3]).encode("utf-8"), "bad.json")
