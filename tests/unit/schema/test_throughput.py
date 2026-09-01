"""Throughput and chunked export tests for source-driven generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sdv")

from synth_platform.engine.generation.schema.chunked_export import export_csv_chunks, export_parquet_chunks
from synth_platform.engine.inference.schema.source_driven import (
    SourceDataProfiler,
    SourceDrivenGenerator,
    run_source_driven_pipeline,
)


def _bench_source(rows: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(rows),
            "amount": np.round(rng.normal(100, 25, rows), 2),
            "segment": rng.choice(["a", "b", "c"], rows),
            "active": rng.choice([True, False], rows),
        }
    )


def test_profile_includes_correlations_and_missingness():
    df = _bench_source(200)
    df.loc[0, "amount"] = np.nan
    profile = SourceDataProfiler().profile(df, table_name="t")
    assert profile.missingness.get("columns_with_nulls", 0) >= 1
    assert isinstance(profile.numeric_correlations, list)


def test_sample_chunks_yields_expected_row_count():
    df = _bench_source(80)
    gen = SourceDrivenGenerator()
    gen.fit(df, model_type="gaussian_copula", seed=1)
    chunks = list(gen.sample_chunks(95, chunk_size=30, seed=1))
    assert sum(len(c) for c in chunks) == 95
    assert len(chunks) == 4


def test_chunked_parquet_export(tmp_path: Path):
    df = _bench_source(50)
    gen = SourceDrivenGenerator()
    gen.fit(df, model_type="gaussian_copula", seed=2)
    path = tmp_path / "out.parquet"
    rows = export_parquet_chunks(gen.sample_chunks(120, 40, seed=2), path)
    assert rows == 120
    loaded = pd.read_parquet(path)
    assert len(loaded) == 120


def test_chunked_csv_export(tmp_path: Path):
    frames = [pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [3, 4, 5]})]
    path = tmp_path / "out.csv"
    rows = export_csv_chunks(iter(frames), path)
    assert rows == 5
    loaded = pd.read_csv(path)
    assert len(loaded) == 5


def test_run_source_driven_pipeline_writes_reports(tmp_path: Path):
    source = _bench_source(300)
    result = run_source_driven_pipeline(
        source,
        rows=500,
        output_dir=tmp_path,
        seed=7,
        table_name="bench",
        model_type="gaussian_copula",
        chunk_size=200,
        export_format="parquet",
    )
    assert (tmp_path / "performance_report.json").exists()
    assert (tmp_path / "fidelity_report.json").exists()
    assert len(result["preview"]) <= 500
    perf = result["performance"]
    assert perf["rows_per_second"] >= 0
    assert "peak_memory_mb" in perf
    assert perf["stages"]["profile_seconds"] >= 0
    assert perf["stages"]["fit_seconds"] >= 0


def test_benchmark_100k_smoke(tmp_path: Path):
    """100k-row smoke benchmark with chunked Parquet export."""
    source = _bench_source(1000, seed=42)
    out = tmp_path / "bench_100k"
    result = run_source_driven_pipeline(
        source,
        rows=100_000,
        output_dir=out,
        seed=42,
        table_name="benchmark",
        model_type="gaussian_copula",
        chunk_size=25_000,
        export_format="parquet",
    )
    perf = result["performance"]
    assert perf["rows_generated"] == 100_000
    assert perf["memory_bounded"] is True
    parquet_path = out / "benchmark.parquet"
    assert parquet_path.exists()
    assert len(pd.read_parquet(parquet_path)) == 100_000
    payload = json.loads((out / "performance_report.json").read_text(encoding="utf-8"))
    assert payload["stages"]["export_seconds"] >= 0
