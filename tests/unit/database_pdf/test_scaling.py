"""Unit tests for src/core/scaling.py's chunking and profile-merge utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.common.database.core.scaling import (
    estimate_chunk_count,
    iter_dataframe_chunks,
    merge_chunk_profiles,
    process_in_chunks,
)
from synth_platform.engine.profiling.database.profiler import StructuredProfiler

pytestmark = pytest.mark.unit


def test_iter_dataframe_chunks_splits_by_size():
    df = pd.DataFrame({"x": range(10)})
    chunks = list(iter_dataframe_chunks(df, chunk_size=3))
    assert [len(c) for c in chunks] == [3, 3, 3, 1]
    assert pd.concat(chunks)["x"].tolist() == list(range(10))


def test_iter_dataframe_chunks_rejects_non_positive_size():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        list(iter_dataframe_chunks(df, chunk_size=0))


def test_estimate_chunk_count():
    assert estimate_chunk_count(0) == 0
    assert estimate_chunk_count(10, chunk_size=3) == 4
    assert estimate_chunk_count(9, chunk_size=3) == 3


def test_process_in_chunks_without_aggregator_returns_list():
    df = pd.DataFrame({"x": range(5)})
    results = process_in_chunks(df, lambda chunk: len(chunk), chunk_size=2)
    assert results == [2, 2, 1]


def test_process_in_chunks_with_aggregator_combines_results():
    df = pd.DataFrame({"x": range(5)})
    total = process_in_chunks(df, lambda chunk: len(chunk), chunk_size=2, aggregator=sum)
    assert total == 5


def test_merge_chunk_profiles_empty_list_returns_empty_dict():
    assert merge_chunk_profiles([]) == {}


def test_merge_chunk_profiles_matches_unchunked_profile_for_bounded_cardinality_column():
    df = pd.DataFrame({"category_col": ["a", "b", "a", "c", "a", "b"]})
    profiler = StructuredProfiler()

    whole = profiler.profile_table(df, "t")
    chunk_profiles = [
        profiler.profile_table(chunk, "t") for chunk in iter_dataframe_chunks(df, chunk_size=2)
    ]
    merged = merge_chunk_profiles(chunk_profiles)

    col = merged["columns"]["category_col"]
    whole_col = whole["columns"]["category_col"]
    assert col["distinct_count"] == whole_col["distinct_count"]
    assert col["null_count"] == whole_col["null_count"]
    assert col["category_frequencies"] == whole_col["category_frequencies"]
    assert "approximate_distinct_count_chunked" not in col["warnings"]


def test_merge_chunk_profiles_flags_approximate_distinct_count_for_high_cardinality_column():
    # Each chunk must exceed ProfilerConfig.max_category_values (50) on its
    # own so category_frequencies is None per-chunk - otherwise the merge
    # recovers an exact distinct_count from the merged frequency table.
    df = pd.DataFrame({"identifier_col": [f"id-{i}" for i in range(120)]})
    profiler = StructuredProfiler()

    chunk_profiles = [
        profiler.profile_table(chunk, "t") for chunk in iter_dataframe_chunks(df, chunk_size=60)
    ]
    merged = merge_chunk_profiles(chunk_profiles)

    col = merged["columns"]["identifier_col"]
    assert col["category_frequencies"] is None
    assert "approximate_distinct_count_chunked" in col["warnings"]
    assert col["distinct_count"] <= merged["row_count"]


def test_merge_chunk_profiles_combines_numeric_min_max():
    df = pd.DataFrame({"n": [5, 1, 9, 3, 7, 2]})
    profiler = StructuredProfiler()

    chunk_profiles = [
        profiler.profile_table(chunk, "t") for chunk in iter_dataframe_chunks(df, chunk_size=2)
    ]
    merged = merge_chunk_profiles(chunk_profiles)

    col = merged["columns"]["n"]
    assert col["minimum"] == 1
    assert col["maximum"] == 9
    assert col["row_count"] == 6


def test_merge_chunk_profiles_reconstructs_exact_mean_via_weighted_average():
    df = pd.DataFrame({"n": [10, 20, 30, 40, 50, 60, 70]})
    profiler = StructuredProfiler()

    whole = profiler.profile_table(df, "t")
    chunk_profiles = [
        profiler.profile_table(chunk, "t") for chunk in iter_dataframe_chunks(df, chunk_size=3)
    ]
    merged = merge_chunk_profiles(chunk_profiles)

    assert merged["columns"]["n"]["mean"] == pytest.approx(whole["columns"]["n"]["mean"])


def test_merge_chunk_profiles_omits_quantiles_with_warning_instead_of_first_chunk_only():
    df = pd.DataFrame({"n": [10, 20, 30, 40, 50, 60, 70]})
    profiler = StructuredProfiler()

    chunk_profiles = [
        profiler.profile_table(chunk, "t") for chunk in iter_dataframe_chunks(df, chunk_size=3)
    ]
    merged = merge_chunk_profiles(chunk_profiles)

    col = merged["columns"]["n"]
    assert col["median"] is None
    assert col["quantiles"] is None
    assert col["generation_lower_bound"] is None
    assert col["generation_upper_bound"] is None
    assert "quantile_stats_skipped_in_chunked_mode" in col["warnings"]


def test_merge_chunk_profiles_reconstructs_exact_string_length_mean():
    df = pd.DataFrame({"s": ["a", "bb", "ccc", "dddd", "e", "ff", "ggggggg"]})
    profiler = StructuredProfiler()

    whole = profiler.profile_table(df, "t")
    chunk_profiles = [
        profiler.profile_table(chunk, "t") for chunk in iter_dataframe_chunks(df, chunk_size=3)
    ]
    merged = merge_chunk_profiles(chunk_profiles)

    col = merged["columns"]["s"]
    whole_col = whole["columns"]["s"]
    assert col["string_lengths"]["min"] == whole_col["string_lengths"]["min"]
    assert col["string_lengths"]["max"] == whole_col["string_lengths"]["max"]
    assert col["string_lengths"]["mean"] == pytest.approx(whole_col["string_lengths"]["mean"])
