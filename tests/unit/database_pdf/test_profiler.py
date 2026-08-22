"""Unit tests: StructuredProfiler statistics against independently hand-calculated values.

Covers the fixture categories the spec calls out explicitly: numeric,
categorical, Boolean, datetime, identifier, constant, mostly-null, empty,
mixed-type and very-high-cardinality columns.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.profiling.database.profiler import StructuredProfiler

pytestmark = pytest.mark.unit


@pytest.fixture
def mixed_shape_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric_col": [10, 20, 30, 40, 50],
            "float_col": [1.5, 2.5, None, 4.5, 5.5],
            "categorical_col": ["red", "blue", "red", "green", "red"],
            "boolean_col": [True, False, True, True, False],
            "datetime_col": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "identifier_col": ["id-1", "id-2", "id-3", "id-4", "id-5"],
            "constant_col": ["X", "X", "X", "X", "X"],
            "mostly_null_col": [None, None, None, None, "only-one"],
        }
    )


@pytest.fixture
def report(mixed_shape_df: pd.DataFrame) -> dict:
    return StructuredProfiler().profile_table(mixed_shape_df, "fixture_table")


def test_row_count_and_table_name(report):
    assert report["table_name"] == "fixture_table"
    assert report["row_count"] == 5


def test_numeric_column_stats(report):
    col = report["columns"]["numeric_col"]
    assert col["inferred_dtype"] == "integer"
    assert col["null_count"] == 0
    assert col["null_percentage"] == 0.0
    assert col["distinct_count"] == 5
    assert col["distinct_ratio"] == 1.0
    assert col["minimum"] == 10
    assert col["maximum"] == 50
    assert col["mean"] == pytest.approx(30.0)
    assert col["median"] == pytest.approx(30.0)
    assert col["quantiles"] == pytest.approx({"p25": 20.0, "p50": 30.0, "p75": 40.0})
    assert "near_unique_column" in col["warnings"]


def test_float_column_with_null_stats(report):
    col = report["columns"]["float_col"]
    assert col["inferred_dtype"] == "float"
    assert col["null_count"] == 1
    assert col["null_percentage"] == pytest.approx(20.0)
    assert col["distinct_count"] == 4
    assert col["distinct_ratio"] == pytest.approx(0.8)
    assert col["minimum"] == pytest.approx(1.5)
    assert col["maximum"] == pytest.approx(5.5)
    assert col["mean"] == pytest.approx(3.5)
    assert col["median"] == pytest.approx(3.5)
    assert col["quantiles"] == pytest.approx({"p25": 2.25, "p50": 3.5, "p75": 4.75})
    assert col["warnings"] == []


def test_categorical_column_stats(report):
    col = report["columns"]["categorical_col"]
    assert col["inferred_dtype"] == "string"
    assert col["distinct_count"] == 3
    assert col["distinct_ratio"] == pytest.approx(0.6)
    assert col["category_frequencies"] == {"red": 3, "blue": 1, "green": 1}
    assert col["string_lengths"] == {"min": 3, "max": 5, "mean": pytest.approx(3.6)}
    assert col["warnings"] == []


def test_boolean_column_stats(report):
    col = report["columns"]["boolean_col"]
    assert col["inferred_dtype"] == "boolean"
    assert col["distinct_count"] == 2
    assert col["category_frequencies"] == {"True": 3, "False": 2}


def test_datetime_column_stats(report):
    col = report["columns"]["datetime_col"]
    assert col["inferred_dtype"] == "datetime"
    assert col["minimum"] == "2024-01-01T00:00:00"
    assert col["maximum"] == "2024-01-05T00:00:00"
    assert col["distinct_count"] == 5
    assert "near_unique_column" in col["warnings"]


def test_identifier_like_column_stats(report):
    col = report["columns"]["identifier_col"]
    assert col["distinct_count"] == 5
    assert col["distinct_ratio"] == 1.0
    assert "near_unique_column" in col["warnings"]
    assert col["string_lengths"] == {"min": 4, "max": 4, "mean": 4.0}


def test_constant_column_stats(report):
    col = report["columns"]["constant_col"]
    assert col["distinct_count"] == 1
    assert col["category_frequencies"] == {"X": 5}
    assert "constant_column" in col["warnings"]
    # A trivially constant column must not also be reported as "imbalanced" -
    # that warning is reserved for >1-category columns with a dominant value.
    assert "severe_category_imbalance" not in col["warnings"]


def test_mostly_null_column_stats(report):
    col = report["columns"]["mostly_null_col"]
    assert col["null_count"] == 4
    assert col["null_percentage"] == pytest.approx(80.0)
    assert col["distinct_count"] == 1
    assert "extreme_missingness" in col["warnings"]
    assert col["string_lengths"] == {"min": 8, "max": 8, "mean": 8.0}


def test_representative_examples_are_masked(report):
    col = report["columns"]["identifier_col"]
    assert len(col["representative_examples"]) <= 3
    for example in col["representative_examples"]:
        assert example not in {"id-1", "id-2", "id-3", "id-4", "id-5"}


def test_dataset_level_warnings_aggregate_column_warnings(report):
    assert any(w.startswith("constant_col:") for w in report["warnings"])
    assert any(w.startswith("mostly_null_col:") for w in report["warnings"])


def test_empty_table_produces_zeroed_stats_and_warning():
    df = pd.DataFrame({"col_a": pd.Series([], dtype="object"), "col_b": pd.Series([], dtype="int64")})
    report = StructuredProfiler().profile_table(df, "empty_table")

    assert report["row_count"] == 0
    assert "empty_table" in report["warnings"]
    for col in report["columns"].values():
        assert col["row_count"] == 0
        assert col["null_count"] == 0
        assert col["null_percentage"] == 0.0
        assert col["distinct_count"] == 0
        assert col["distinct_ratio"] == 0.0
        assert col["inferred_dtype"] == "unknown"


def test_all_null_column_reports_extreme_missingness_without_being_constant():
    df = pd.DataFrame({"col": [None, None, None]})
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["col"]
    assert col["null_percentage"] == 100.0
    assert col["distinct_count"] == 0
    assert "extreme_missingness" in col["warnings"]
    assert "constant_column" not in col["warnings"]


def test_mixed_type_column_falls_back_to_string():
    df = pd.DataFrame({"mixed_col": [1, "two", 3.0, None, "five"]})
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["mixed_col"]
    assert col["inferred_dtype"] == "string"
    assert col["null_count"] == 1
    assert col["distinct_count"] == 4
    assert col["string_lengths"] == {"min": 1, "max": 4, "mean": pytest.approx(2.75)}


def test_integer_column_with_nulls_is_still_classified_integer():
    df = pd.DataFrame({"age": [18, 25, None, 40]})
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["age"]
    assert col["inferred_dtype"] == "integer"
    assert col["null_count"] == 1
    assert col["minimum"] == pytest.approx(18.0)
    assert col["maximum"] == pytest.approx(40.0)


def test_very_high_cardinality_column_skips_category_frequencies():
    df = pd.DataFrame({"uuid_col": [f"row-{i}" for i in range(200)]})
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["uuid_col"]
    assert col["distinct_count"] == 200
    assert col["distinct_ratio"] == 1.0
    assert col["category_frequencies"] is None
    assert "near_unique_column" in col["warnings"]


def test_invalid_dates_warning_for_partially_parseable_string_column():
    df = pd.DataFrame({"maybe_date": ["2024-01-01", "2024-02-01", "not-a-date", "2024-03-01"]})
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["maybe_date"]
    assert col["date_parse_success_rate"] == pytest.approx(0.75)
    assert "invalid_dates" in col["warnings"]


def test_fully_parseable_date_column_has_no_invalid_dates_warning():
    df = pd.DataFrame({"date_str": ["2024-01-01", "2024-02-01", "2024-03-01"]})
    report = StructuredProfiler().profile_table(df, "t")
    col = report["columns"]["date_str"]
    assert col["date_parse_success_rate"] == pytest.approx(1.0)
    assert "invalid_dates" not in col["warnings"]


def test_correlation_excludes_constant_column_and_includes_perfect_correlation():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 3, 4, 1, 2],
            "const": [7, 7, 7, 7, 7],
        }
    )
    report = StructuredProfiler().profile_table(df, "t")
    pairs = {(c["column_a"], c["column_b"]): c["correlation"] for c in report["correlations"]}

    assert ("a", "b") in pairs
    assert pairs[("a", "b")] == pytest.approx(1.0)
    assert all("const" not in pair for pair in pairs)
    assert len(pairs) == 3  # a-b, a-c, b-c; all three "const" pairs excluded


def test_primary_key_cross_check_flags_duplicates_and_nulls():
    df = pd.DataFrame({"pk": [1, 1, None, 3]})
    schema = {"primary_key": ["pk"]}
    report = StructuredProfiler().profile_table(df, "t", schema)
    col = report["columns"]["pk"]
    assert "primary_key_has_nulls" in col["warnings"]
    assert "primary_key_duplicates_in_sample" in col["warnings"]


def test_primary_key_cross_check_passes_for_clean_key():
    df = pd.DataFrame({"pk": [1, 2, 3, 4]})
    schema = {"primary_key": ["pk"]}
    report = StructuredProfiler().profile_table(df, "t", schema)
    col = report["columns"]["pk"]
    assert "primary_key_has_nulls" not in col["warnings"]
    assert "primary_key_duplicates_in_sample" not in col["warnings"]


def test_execution_metadata_present():
    df = pd.DataFrame({"x": [1, 2, 3]})
    report = StructuredProfiler().profile_table(df, "t")
    meta = report["execution_metadata"]
    assert meta["sample_row_count"] == 3
    assert meta["duration_ms"] >= 0
    assert meta["profiler_version"] == "1.0"
