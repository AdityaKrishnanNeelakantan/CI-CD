"""Tests for optimized source overlay and sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sdv")

from synth_platform.application.orchestration.schema.config import PipelineConfig
from synth_platform.application.orchestration.schema.source_driven import DEFAULT_SOURCE_CHUNK_SIZE, resolve_source_chunk_size, run_source_pipeline
from synth_platform.engine.inference.schema.source_driven import SourceDrivenGenerator, pii_replay_column_names
from synth_platform.engine.inference.schema.source_overlay import ProtectedValueCache, build_overlay_plan, apply_overlay_plan


def _pii_source(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(rows)],
            "name": [f"User {i}" for i in range(rows)],
            "email": [f"user{i}@example.com" for i in range(rows)],
            "age": rng.integers(22, 75, rows),
            "segment": rng.choice(["a", "b"], rows),
        }
    )


def test_overlay_plan_cached_and_replay_zero():
    df = _pii_source(80)
    gen = SourceDrivenGenerator()
    gen.fit(df, model_type="gaussian_copula", seed=3)
    assert gen._overlay_plan is not None
    synthetic = gen.sample(50, seed=3)
    report = gen._overlay_plan.replay_report(synthetic)
    assert report["passed"] is True
    assert report["total_replay_values"] == 0


def test_overlay_replay_zero_on_original_like_pii_columns():
    """Regression: common Faker names/emails must not replay source PII columns."""
    rng = np.random.default_rng(99)
    rows = 500
    df = pd.DataFrame(
        {
            "Full Name": [f"Person {i}" for i in range(rows)],
            "Username": [f"user{i}" for i in range(rows)],
            "Email": [f"person{i}@example.com" for i in range(rows)],
            "Age": rng.integers(25, 75, rows),
            "Gender": rng.choice(["Male", "Female"], rows),
            "Account Type": rng.choice(["Checking", "Savings"], rows),
            "Country": rng.choice(["US", "UK", "CA"], rows),
            "Balance": rng.uniform(100, 50000, rows).round(2),
        }
    )
    gen = SourceDrivenGenerator()
    gen.fit(df, model_type="gaussian_copula", seed=42)
    synthetic = gen.sample(100, seed=42)
    report = gen._overlay_plan.replay_report(synthetic)
    assert report["passed"] is True, report
    assert report["total_replay_values"] == 0
    assert "Balance" not in pii_replay_column_names(gen.profile)


def test_protected_value_cache_detects_replay():
    df = _pii_source(20)
    cache = ProtectedValueCache.from_source(df, ["email"])
    bad = pd.DataFrame({"email": [df["email"].iloc[0], "fresh@example.com"]})
    assert cache.total_replay(bad, ["email"]) == 1


def test_resolve_source_chunk_size_defaults_to_25k():
    cfg = PipelineConfig(generation_mode="source_driven", chunk_size=10_000)
    assert resolve_source_chunk_size(cfg, 100_000) == DEFAULT_SOURCE_CHUNK_SIZE


def test_resolve_source_chunk_size_honors_explicit_override():
    cfg = PipelineConfig(generation_mode="source_driven", source_chunk_size=50_000)
    assert resolve_source_chunk_size(cfg, 100_000) == 50_000


def test_chunked_pipeline_writes_expected_row_count_without_full_frame_in_result(tmp_path):
    df = _pii_source(100)
    result = run_source_pipeline(
        df,
        PipelineConfig(
            generation_mode="source_driven",
            preview_only=False,
            preview_rows=20,
            full_rows=500,
            source_chunk_size=200,
            output_dir=tmp_path / "stream",
            seed=4,
            table_name="customers",
        ),
    )
    assert result.row_counts["customers"] == 500
    preview = result.preview_tables["customers"]
    assert len(preview) <= 20
    export_path = next(iter(result.export_paths.values()))
    assert export_path.exists()
    assert len(pd.read_parquet(export_path)) == 500


def test_performance_report_includes_sdv_and_pii_breakdown(tmp_path):
    df = _pii_source(80)
    result = run_source_pipeline(
        df,
        PipelineConfig(
            generation_mode="source_driven",
            preview_only=False,
            preview_rows=20,
            full_rows=200,
            source_chunk_size=100,
            output_dir=tmp_path / "perf",
            seed=5,
            table_name="customers",
        ),
    )
    assert result.performance is not None
    stages = result.performance.stages.to_dict()
    assert "pii_overlay_seconds" in stages
    assert stages.get("pii_overlay_seconds", 0) >= 0
