"""Tests for PM-style source vs synthetic comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sdv")

from synth_platform.engine.validation.schema.pm_compare import compare_source_synthetic, write_pm_compare_report
from synth_platform.engine.inference.schema.source_driven import SourceDrivenGenerator


def _banking_analytics_df(rows: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"Customer {i}" for i in range(rows)]
    return pd.DataFrame(
        {
            "Customer ID": rng.integers(1000, 9999, rows),
            "Customer Name": names,
            "Age": rng.integers(22, 78, rows),
            "Gender": rng.choice(["Male", "Female", "Other"], rows),
            "Country": rng.choice(["USA", "UK", "India", "Germany"], rows),
            "Account Type": rng.choice(["Checking", "Savings", "Premium"], rows),
            "Balance": np.round(rng.normal(8500, 3200, rows), 2),
            "Credit Score": rng.integers(520, 820, rows),
            "Account Creation Date": pd.date_range("2018-01-01", periods=rows, freq="D").astype(str),
            "Last Active Date": pd.date_range("2024-01-01", periods=rows, freq="D").astype(str),
            "Is Active": rng.choice([True, False], rows, p=[0.85, 0.15]),
            "Customer Segment": rng.choice(["Retail", "Premium", "Business"], rows),
        }
    )


def test_compare_source_synthetic_passes_on_generated_twin(tmp_path: Path):
    source = _banking_analytics_df(rows=180, seed=1)
    generator = SourceDrivenGenerator()
    generator.fit(source, table_name="Global Banking Customer Analytics Dataset", seed=11)
    synthetic = generator.sample(len(source), seed=11)

    report = compare_source_synthetic(
        source,
        synthetic,
        expected_rows=len(source),
        table_name="Global Banking Customer Analytics Dataset",
    )
    check_names = {c["name"] for c in report["checks"]}
    assert "schema_parity" in check_names
    assert "genericness" in check_names
    assert "privacy_and_uniqueness" in check_names
    assert report["checks"][0]["passed"] is True
    assert report["pm_readiness"] is True
    assert "Customer Name" in report["pii_columns_detected"]

    out = write_pm_compare_report(report, tmp_path / "pm_report.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pm_summary"]


def test_compare_detects_schema_mismatch():
    source = _banking_analytics_df(rows=50, seed=2)
    synthetic = source.drop(columns=["Customer Name"]).copy()
    report = compare_source_synthetic(source, synthetic, table_name="bank")
    schema = next(c for c in report["checks"] if c["name"] == "schema_parity")
    assert schema["passed"] is False
    assert report["pm_readiness"] is False
