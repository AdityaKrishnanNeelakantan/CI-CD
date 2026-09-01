"""Unit tests for the export-safe companion fields added to StructuredProfiler's
output (generation_lower_bound/upper_bound, safe_category_frequencies) -
computed alongside the exact minimum/maximum/category_frequencies, never
replacing them, so in-pipeline consumers (inference, cleaning) keep the
exact evidence they already depend on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.common.database.privacy.rare_category import RARE_CATEGORY_LABEL
from synth_platform.engine.profiling.database.profiler import ProfilerConfig, StructuredProfiler

pytestmark = pytest.mark.unit


def test_numeric_bounds_are_within_the_exact_min_max():
    df = pd.DataFrame({"amount": list(range(1, 101))})  # 1..100
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["amount"]
    assert col["minimum"] == 1.0
    assert col["maximum"] == 100.0
    assert col["generation_lower_bound"] > col["minimum"]
    assert col["generation_upper_bound"] < col["maximum"]


def test_numeric_bounds_never_leak_a_true_outlier():
    """A single unusual value (987654.32 among ordinary ~1000-2000 values)
    must not appear as the exported upper bound - only the exact
    'maximum' field may equal it, the safe companion must not.
    """
    values = [1000.0 + i for i in range(99)] + [987654.32]
    df = pd.DataFrame({"balance": values})
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["balance"]
    assert col["maximum"] == 987654.32
    assert col["generation_upper_bound"] != 987654.32
    assert col["generation_upper_bound"] < 987654.32


def test_datetime_bounds_are_present_and_within_range():
    df = pd.DataFrame({"signup": pd.to_datetime([f"2024-01-{d:02d}" for d in range(1, 29)])})
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["signup"]
    assert col["generation_lower_bound"] is not None
    assert col["generation_upper_bound"] is not None
    assert col["minimum"] <= col["generation_lower_bound"]
    assert col["generation_upper_bound"] <= col["maximum"]


def test_safe_category_frequencies_suppresses_rare_values():
    df = pd.DataFrame({"status": ["active"] * 30 + ["rare_one_off"]})
    config = ProfilerConfig(rare_category_minimum_support=20)
    report = StructuredProfiler(config).profile_table(df, "t")
    col = report["columns"]["status"]
    assert col["category_frequencies"] == {"active": 30, "rare_one_off": 1}
    assert col["safe_category_frequencies"] == {"active": 30, RARE_CATEGORY_LABEL: 1}
    assert "rare_one_off" not in col["safe_category_frequencies"]


def test_boolean_columns_skip_rare_category_suppression():
    df = pd.DataFrame({"flag": [True] * 25 + [False] * 2})
    config = ProfilerConfig(rare_category_minimum_support=20)
    report = StructuredProfiler(config).profile_table(df, "t")
    col = report["columns"]["flag"]
    assert col["safe_category_frequencies"] == col["category_frequencies"]


def test_empty_or_unprofiled_columns_have_none_safe_fields():
    df = pd.DataFrame({"free_text_col": ["a long unique sentence " + str(i) for i in range(60)]})
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["free_text_col"]
    assert col["generation_lower_bound"] is None
    assert col["generation_upper_bound"] is None
    assert col["safe_category_frequencies"] is None
