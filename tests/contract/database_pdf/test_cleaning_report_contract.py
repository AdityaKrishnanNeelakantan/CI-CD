"""Contract: clean_table() always returns the same report shape, and the
cleaned DataFrame only ever contains approved columns, regardless of what
semantic types or data shapes are involved.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.profiling.database.cleaning.cleaner import DataCleaner

pytestmark = pytest.mark.contract

REQUIRED_REPORT_KEYS = {
    "table_name",
    "cleaned_at",
    "row_count",
    "duplicate_row_count",
    "excluded_columns",
    "columns",
    "warnings",
}
REQUIRED_COLUMN_ENTRY_KEYS = {
    "semantic_type",
    "action",
    "nulls_before",
    "nulls_after",
    "nulls_imputed",
    "outliers_flagged",
    "duplicate_value_count",
    "coercion_failures",
}

FIXTURES = {
    "numerical": (
        pd.DataFrame({"n": [1, 2, None]}),
        {"columns": {"n": {"semantic_type": "numerical", "inference_status": "approved"}}},
    ),
    "category": (
        pd.DataFrame({"c": ["a", "b", None]}),
        {"columns": {"c": {"semantic_type": "category", "inference_status": "approved"}}},
    ),
    "identifier": (
        pd.DataFrame({"i": ["x1", "x2"]}),
        {"columns": {"i": {"semantic_type": "identifier", "inference_status": "approved"}}},
    ),
    "datetime": (
        pd.DataFrame({"d": ["2024-01-01", "bad"]}),
        {"columns": {"d": {"semantic_type": "datetime", "inference_status": "approved"}}},
    ),
    "not_approved": (
        pd.DataFrame({"r": [1, 2]}),
        {"columns": {"r": {"semantic_type": "numerical", "inference_status": "review_required"}}},
    ),
    "empty": (
        pd.DataFrame({"n": pd.Series([], dtype="float64")}),
        {"columns": {"n": {"semantic_type": "numerical", "inference_status": "approved"}}},
    ),
}


@pytest.mark.parametrize("df,table_contract", FIXTURES.values(), ids=FIXTURES.keys())
def test_report_shape_is_invariant(df: pd.DataFrame, table_contract: dict):
    cleaned_df, report = DataCleaner().clean_table(df, "t", table_contract)

    assert set(report) == REQUIRED_REPORT_KEYS
    assert isinstance(report["excluded_columns"], list)
    assert isinstance(report["warnings"], list)
    for column_entry in report["columns"].values():
        assert set(column_entry) == REQUIRED_COLUMN_ENTRY_KEYS

    # The cleaned DataFrame must only ever contain approved columns.
    approved_columns = {
        name
        for name, col in table_contract["columns"].items()
        if col["inference_status"] == "approved"
    }
    assert set(cleaned_df.columns) == approved_columns


def test_report_is_json_serialisable():
    import json

    df = pd.DataFrame({"n": [1, 2, None], "c": ["a", "b", None]})
    table_contract = {
        "columns": {
            "n": {"semantic_type": "numerical", "inference_status": "approved"},
            "c": {"semantic_type": "category", "inference_status": "approved"},
        }
    }
    _, report = DataCleaner().clean_table(df, "t", table_contract)
    json.dumps(report)  # must not raise
