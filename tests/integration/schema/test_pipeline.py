"""Pipeline orchestration tests for schema-first and source-driven modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synth_platform.application.orchestration.schema.config import PipelineConfig
from synth_platform.application.orchestration.schema.schema_driven import run_schema_pipeline
from synth_platform.application.orchestration.schema.session import clear_large_session_objects, store_preview_sample_only
from synth_platform.engine.inference.schema.schema import Column, Relationship, SchemaConfig, Table
from synth_platform.engine.generation.schema.simulator import DataSimulator
from synth_platform.engine.inference.schema.yaml_schema import load_yaml_schema


pytest.importorskip("sdv")

from synth_platform.application.orchestration.schema.source_driven import fit_or_reuse_source_generator, run_source_pipeline
from synth_platform.engine.validation.schema.performance.timing import PipelineStageTimings


def _parent_child_schema(*, parent_rows: int = 200, child_rows: int = 400) -> SchemaConfig:
    return SchemaConfig(
        name="parent_child",
        seed=7,
        tables=[
            Table(name="parents", row_count=parent_rows),
            Table(name="children", row_count=child_rows),
        ],
        columns={
            "parents": [
                Column(name="parent_id", type="int", unique=True, min=1, max=100000),
                Column(name="segment", type="categorical", distribution_params={"choices": ["a", "b", "c"]}),
            ],
            "children": [
                Column(name="child_id", type="int", unique=True, min=1, max=200000),
                Column(name="parent_id", type="foreign_key", distribution_params={}),
                Column(name="score", type="float", min=0, max=100),
            ],
        },
        relationships=[
            Relationship(parent_table="parents", parent_key="parent_id", child_table="children", child_key="parent_id"),
        ],
    )


def _bench_source(rows: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(rows),
            "amount": np.round(rng.normal(100, 25, rows), 2),
            "segment": rng.choice(["a", "b", "c"], rows),
            "active": rng.choice([True, False], rows),
        }
    )


class _FakeSession(dict):
    def pop(self, key, default=None):
        return super().pop(key, default)


def test_schema_preview_and_full_rows_independent(tmp_path: Path):
    schema = _parent_child_schema(parent_rows=50, child_rows=100)
    preview_cfg = PipelineConfig(
        generation_mode="schema_driven",
        preview_only=True,
        preview_rows=20,
        seed=7,
        output_dir=tmp_path / "preview",
    )
    preview = run_schema_pipeline(schema, preview_cfg)
    preview_total = sum(preview.row_counts.values())

    full_schema = _parent_child_schema(parent_rows=500, child_rows=1000)
    full_cfg = PipelineConfig(
        generation_mode="schema_driven",
        preview_only=False,
        preview_rows=20,
        full_rows=600,
        chunk_size=250,
        export_format="parquet",
        output_dir=tmp_path / "full",
        seed=7,
    )
    full = run_schema_pipeline(_parent_child_schema(parent_rows=500, child_rows=1000), full_cfg)
    assert sum(full.row_counts.values()) >= 600
    assert preview_total < sum(full.row_counts.values())
    assert full.performance is not None
    assert full.performance.rows_generated >= 600


def test_schema_streaming_does_not_store_full_tables_in_session_helper():
    session = _FakeSession({"full_tables": {"t": pd.DataFrame({"x": range(1000)})}, "sample_tables": {"t": pd.DataFrame({"x": [1]})}})
    store_preview_sample_only(session, key="full_preview_tables", tables={"t": pd.DataFrame({"x": range(5000)})}, max_rows=25)
    clear_large_session_objects(session)
    assert "full_tables" not in session
    assert len(session["full_preview_tables"]["t"]) == 25


def test_schema_chunked_generation_preserves_fk_integrity(tmp_path: Path):
    schema = _parent_child_schema(parent_rows=300, child_rows=600)
    config = PipelineConfig(
        generation_mode="schema_driven",
        preview_only=False,
        preview_rows=50,
        full_rows=900,
        chunk_size=150,
        export_format="parquet",
        output_dir=tmp_path / "fk",
        seed=11,
    )
    result = run_schema_pipeline(schema, config)
    parents = pd.read_parquet(tmp_path / "fk" / "parents.parquet")
    children = pd.read_parquet(tmp_path / "fk" / "children.parquet")
    assert set(children["parent_id"]).issubset(set(parents["parent_id"]))


def test_schema_rule_evidence_aggregates_across_chunks(tmp_path: Path):
    schema = SchemaConfig(
        name="rules",
        seed=3,
        tables=[Table(name="items", row_count=1000)],
        columns={
            "items": [
                Column(name="item_id", type="int", unique=True, min=1, max=100000),
                Column(name="status", type="categorical", distribution_params={"choices": ["open", "closed"]}),
            ]
        },
    )
    config = PipelineConfig(
        generation_mode="schema_driven",
        preview_only=False,
        preview_rows=50,
        full_rows=1000,
        chunk_size=200,
        rules_text="30% of items.status = open",
        export_format="parquet",
        output_dir=tmp_path / "rules",
        seed=3,
    )
    result = run_schema_pipeline(schema, config)
    assert result.rule_evidence
    assert any(item.get("aggregated") for item in result.rule_evidence)


def test_simulator_uses_compact_context_columns():
    schema = _parent_child_schema(parent_rows=20, child_rows=40)
    simulator = DataSimulator(schema, is_preview=True)
    list(simulator.generate_all())
    for table_name, ctx in simulator.context.items():
        assert "parent_id" in ctx.columns or "child_id" in ctx.columns or "segment" in ctx.columns
        assert len(ctx.columns) <= 4


def test_source_preview_and_full_independent(tmp_path: Path):
    source = _bench_source(100, seed=1)
    preview_cfg = PipelineConfig(
        generation_mode="source_driven",
        preview_only=True,
        preview_rows=40,
        full_rows=500,
        output_dir=tmp_path / "preview",
        seed=1,
        table_name="bench",
    )
    preview = run_source_pipeline(source, preview_cfg)
    assert preview.row_counts["bench"] <= 40

    full_cfg = PipelineConfig(
        generation_mode="source_driven",
        preview_only=False,
        preview_rows=40,
        full_rows=500,
        chunk_size=150,
        output_dir=tmp_path / "full",
        reuse_artifacts_dir=preview.artifact_dir,
        seed=1,
        table_name="bench",
    )
    full = run_source_pipeline(source, full_cfg)
    assert full.row_counts["bench"] == 500
    assert full.cache_evidence.get("model_reused") is True


def test_source_profile_cache_hit(tmp_path: Path, monkeypatch):
    source = _bench_source(80, seed=2)
    monkeypatch.setenv("HOME", str(tmp_path))
    stages = PipelineStageTimings()
    config = PipelineConfig(
        generation_mode="source_driven",
        preview_rows=30,
        seed=2,
        table_name="bench",
        output_dir=tmp_path / "a",
    )
    _, ev1 = fit_or_reuse_source_generator(source, config, stages=stages, artifact_dir=tmp_path / "a" / "preview")
    assert ev1["refit_performed"] is True
    _, ev2 = fit_or_reuse_source_generator(source, config, stages=stages, artifact_dir=tmp_path / "b" / "preview")
    assert ev2.get("profile_cache_hit") is True


def test_source_chunked_row_count(tmp_path: Path):
    source = _bench_source(120, seed=4)
    config = PipelineConfig(
        generation_mode="source_driven",
        preview_only=False,
        preview_rows=50,
        full_rows=750,
        chunk_size=200,
        export_format="parquet",
        output_dir=tmp_path / "chunk",
        seed=4,
        table_name="bench",
    )
    result = run_source_pipeline(source, config)
    assert result.row_counts["bench"] == 750
    assert (tmp_path / "chunk" / "full" / "bench.parquet").exists()
    assert len(pd.read_parquet(tmp_path / "chunk" / "full" / "bench.parquet")) == 750


def test_performance_metrics_in_final_report(tmp_path: Path):
    schema = load_yaml_schema(Path("tests/fixtures/schema/minimal_company.yaml"), seed=42)
    config = PipelineConfig(
        generation_mode="schema_driven",
        preview_only=True,
        preview_rows=10,
        output_dir=tmp_path / "metrics",
        seed=42,
    )
    result = run_schema_pipeline(schema, config)
    assert result.performance is not None
    payload = result.performance.to_dict()
    assert payload["generation_time_seconds"] >= 0
    assert payload["total_time_seconds"] >= 0
    assert "stages" in payload


def test_export_uses_final_not_preview(tmp_path: Path):
    source = _bench_source(60, seed=5)
    preview_cfg = PipelineConfig(
        generation_mode="source_driven",
        preview_only=True,
        preview_rows=20,
        full_rows=200,
        output_dir=tmp_path / "prev",
        seed=5,
        table_name="bench",
    )
    preview = run_source_pipeline(source, preview_cfg)
    full_cfg = PipelineConfig(
        generation_mode="source_driven",
        preview_only=False,
        preview_rows=20,
        full_rows=200,
        chunk_size=100,
        output_dir=tmp_path / "final",
        reuse_artifacts_dir=preview.artifact_dir,
        seed=5,
        table_name="bench",
    )
    full = run_source_pipeline(source, full_cfg)
    assert full.export_paths
    export_path = next(iter(full.export_paths.values()))
    assert len(pd.read_parquet(export_path)) == 200
    assert len(next(iter(preview.preview_tables.values()))) <= 20
