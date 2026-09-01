"""Unit tests for StructuredProfiler.profile_table_chunked - the memory-bounded
counterpart to profile_table(), wired to src/core/scaling.py's chunk merge.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.common.database.core.scaling import iter_dataframe_chunks
from synth_platform.engine.profiling.database.profiler import StructuredProfiler

pytestmark = pytest.mark.unit


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric_col": [10, 20, 30, 40, 50, 60, 70],
            "category_col": ["red", "blue", "red", "green", "red", "blue", "red"],
        }
    )


def test_profile_table_chunked_reports_same_row_count_as_unchunked(df):
    profiler = StructuredProfiler()
    chunked = profiler.profile_table_chunked(iter_dataframe_chunks(df, chunk_size=3), "t")
    assert chunked["row_count"] == len(df)
    assert chunked["execution_metadata"]["chunk_count"] == 3


def test_profile_table_chunked_matches_unchunked_column_stats(df):
    profiler = StructuredProfiler()
    whole = profiler.profile_table(df, "t")
    chunked = profiler.profile_table_chunked(iter_dataframe_chunks(df, chunk_size=3), "t")

    for col_name in ("numeric_col", "category_col"):
        assert chunked["columns"][col_name]["null_count"] == whole["columns"][col_name]["null_count"]
        assert chunked["columns"][col_name]["distinct_count"] == whole["columns"][col_name]["distinct_count"]

    assert chunked["columns"]["numeric_col"]["minimum"] == whole["columns"]["numeric_col"]["minimum"]
    assert chunked["columns"]["numeric_col"]["maximum"] == whole["columns"]["numeric_col"]["maximum"]
    assert chunked["columns"]["numeric_col"]["mean"] == pytest.approx(
        whole["columns"]["numeric_col"]["mean"]
    )
    assert chunked["columns"]["category_col"]["category_frequencies"] == (
        whole["columns"]["category_col"]["category_frequencies"]
    )


def test_profile_table_chunked_omits_quantiles_with_warning(df):
    profiler = StructuredProfiler()
    chunked = profiler.profile_table_chunked(iter_dataframe_chunks(df, chunk_size=3), "t")
    numeric_col = chunked["columns"]["numeric_col"]
    assert numeric_col["median"] is None
    assert numeric_col["quantiles"] is None
    assert "numeric_col: quantile_stats_skipped_in_chunked_mode" in chunked["warnings"]


def test_profile_table_chunked_omits_correlations_with_explanatory_warning(df):
    profiler = StructuredProfiler()
    chunked = profiler.profile_table_chunked(iter_dataframe_chunks(df, chunk_size=3), "t")
    assert chunked["correlations"] == []
    assert "correlations_skipped_in_chunked_mode" in chunked["warnings"]


def test_profile_table_chunked_recomputes_safe_category_frequencies(df):
    profiler = StructuredProfiler()
    chunked = profiler.profile_table_chunked(iter_dataframe_chunks(df, chunk_size=3), "t")
    assert chunked["columns"]["category_col"]["safe_category_frequencies"] is not None


def test_profile_table_chunked_handles_empty_iterator():
    profiler = StructuredProfiler()
    chunked = profiler.profile_table_chunked(iter([]), "t")
    assert chunked["row_count"] == 0
    assert chunked["columns"] == {}
    assert "empty_table" in chunked["warnings"]
