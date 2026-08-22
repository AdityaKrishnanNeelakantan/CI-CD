"""Contract: profile_table() always returns the same shape, regardless of
what the data looks like. Downstream stages (semantic inference) must be
able to rely on this without checking "did this table have numeric columns"
before indexing into the report.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.profiling.database.profiler import StructuredProfiler

pytestmark = pytest.mark.contract

REQUIRED_REPORT_KEYS = {
    "table_name",
    "profiled_at",
    "row_count",
    "columns",
    "correlations",
    "warnings",
    "execution_metadata",
}
REQUIRED_COLUMN_KEYS = {
    "name",
    "row_count",
    "null_count",
    "null_percentage",
    "distinct_count",
    "distinct_ratio",
    "inferred_dtype",
    "minimum",
    "maximum",
    "mean",
    "median",
    "quantiles",
    "string_lengths",
    "category_frequencies",
    "representative_examples",
    "warnings",
}
REQUIRED_EXECUTION_METADATA_KEYS = {"duration_ms", "profiler_version", "sample_row_count"}

FIXTURE_FRAMES = {
    "all_numeric": pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]}),
    "all_string": pd.DataFrame({"s": ["x", "y", "z"]}),
    "empty": pd.DataFrame({"a": pd.Series([], dtype="int64")}),
    "single_row": pd.DataFrame({"a": [1]}),
    "all_null": pd.DataFrame({"a": [None, None]}),
}


@pytest.mark.parametrize("df", FIXTURE_FRAMES.values(), ids=FIXTURE_FRAMES.keys())
def test_report_shape_is_invariant(df: pd.DataFrame):
    report = StructuredProfiler().profile_table(df, "t")

    assert set(report) == REQUIRED_REPORT_KEYS
    assert isinstance(report["columns"], dict)
    assert isinstance(report["correlations"], list)
    assert isinstance(report["warnings"], list)
    assert set(report["execution_metadata"]) == REQUIRED_EXECUTION_METADATA_KEYS

    for column in report["columns"].values():
        assert REQUIRED_COLUMN_KEYS.issubset(column)
        assert isinstance(column["warnings"], list)
        assert isinstance(column["representative_examples"], list)


def test_report_is_json_serialisable():
    import json

    df = pd.DataFrame(
        {
            "n": [1, 2, None],
            "s": ["a", "b", "c"],
            "d": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    report = StructuredProfiler().profile_table(df, "t")
    json.dumps(report)  # must not raise
