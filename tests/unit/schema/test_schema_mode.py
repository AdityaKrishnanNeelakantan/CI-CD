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
from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table

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
    assert result.row_counts["orders"] == 25
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
