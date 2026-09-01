"""Unit tests: DataCleaner against hand-calculated expected values, one
semantic type at a time.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.profiling.database.cleaning.cleaner import CleaningConfig, DataCleaner

pytestmark = pytest.mark.unit


def approved(semantic_type: str) -> dict:
    return {"semantic_type": semantic_type, "inference_status": "approved"}


def contract(**columns: dict) -> dict:
    return {"columns": columns}


def test_numerical_column_imputed_with_median_and_outliers_flagged():
    df = pd.DataFrame({"amount": [10.0, 12.0, 11.0, 13.0, None, 1000.0]})
    table_contract = contract(amount=approved("numerical"))

    cleaned_df, report = DataCleaner().clean_table(df, "t", table_contract)

    col = report["columns"]["amount"]
    # non-null values [10,12,11,13,1000], median = 12.0
    assert col["action"] == "imputed_median"
    assert col["nulls_before"] == 1
    assert col["nulls_imputed"] == 1
    assert col["nulls_after"] == 0
    assert cleaned_df["amount"].iloc[4] == pytest.approx(12.0)  # imputed slot
    assert col["outliers_flagged"] == 1  # 1000.0 is a clear IQR outlier
    assert not cleaned_df["amount"].isna().any()


def test_numerical_column_can_use_mean_strategy():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, None]})
    table_contract = contract(x=approved("numerical"))
    cleaner = DataCleaner(CleaningConfig(numerical_imputation_strategy="mean"))

    cleaned_df, report = cleaner.clean_table(df, "t", table_contract)
    assert report["columns"]["x"]["action"] == "imputed_mean"
    assert cleaned_df["x"].iloc[3] == pytest.approx(2.0)  # mean of [1,2,3]


def test_numerical_column_tracks_coercion_failures():
    df = pd.DataFrame({"x": ["1", "2", "not-a-number", "4"]})
    table_contract = contract(x=approved("numerical"))
    cleaned_df, report = DataCleaner().clean_table(df, "t", table_contract)
    col = report["columns"]["x"]
    assert col["coercion_failures"] == 1
    assert col["nulls_imputed"] == 1  # the unparseable value got imputed too
    assert not cleaned_df["x"].isna().any()


def test_category_column_imputed_with_mode():
    df = pd.DataFrame({"status": ["active", "active", "inactive", None]})
    table_contract = contract(status=approved("category"))
    cleaned_df, report = DataCleaner().clean_table(df, "t", table_contract)
    col = report["columns"]["status"]
    assert col["action"] == "imputed_mode"
    assert cleaned_df["status"].iloc[3] == "active"
    assert col["nulls_imputed"] == 1


def test_category_column_can_use_sentinel_strategy():
    df = pd.DataFrame({"status": ["active", "inactive", None]})
    table_contract = contract(status=approved("category"))
    cleaner = DataCleaner(CleaningConfig(category_imputation_strategy="sentinel"))
    cleaned_df, report = cleaner.clean_table(df, "t", table_contract)
    assert cleaned_df["status"].iloc[2] == "MISSING"
    assert report["columns"]["status"]["action"] == "imputed_sentinel"


def test_category_column_strips_whitespace():
    df = pd.DataFrame({"status": [" active ", "inactive"]})
    table_contract = contract(status=approved("category"))
    cleaned_df, _ = DataCleaner().clean_table(df, "t", table_contract)
    assert cleaned_df["status"].iloc[0] == "active"


def test_boolean_column_imputed_with_mode():
    df = pd.DataFrame({"is_active": [True, True, False, None]})
    table_contract = contract(is_active=approved("boolean"))
    cleaned_df, report = DataCleaner().clean_table(df, "t", table_contract)
    assert cleaned_df["is_active"].iloc[3] == True  # noqa: E712
    assert report["columns"]["is_active"]["nulls_imputed"] == 1


def test_datetime_column_is_coerced_and_failures_tracked():
    df = pd.DataFrame({"signup": ["2024-01-01", "not-a-date", None]})
    table_contract = contract(signup=approved("datetime"))
    cleaned_df, report = DataCleaner().clean_table(df, "t", table_contract)
    col = report["columns"]["signup"]
    assert col["action"] == "coerced_datetime"
    assert col["nulls_before"] == 1
    assert col["coercion_failures"] == 1  # "not-a-date" failed to parse
    assert col["nulls_after"] == 2
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["signup"])


def test_identifier_column_is_never_imputed_only_reported():
    df = pd.DataFrame({"customer_id": ["c1", "c1", "c2", None]})
    table_contract = contract(customer_id=approved("identifier"))
    cleaned_df, report = DataCleaner().clean_table(df, "t", table_contract)
    col = report["columns"]["customer_id"]
    assert col["action"] == "no_modification"
    assert col["duplicate_value_count"] == 1  # "c1" appears twice
    assert col["nulls_imputed"] == 0
    assert cleaned_df["customer_id"].isna().sum() == 1  # null left untouched


def test_email_and_free_text_are_only_whitespace_stripped_never_imputed():
    df = pd.DataFrame({"email": [" a@x.com ", None]})
    table_contract = contract(email=approved("email"))
    cleaned_df, report = DataCleaner().clean_table(df, "t", table_contract)
    col = report["columns"]["email"]
    assert col["action"] == "whitespace_stripped"
    assert col["nulls_imputed"] == 0
    assert cleaned_df["email"].iloc[0] == "a@x.com"
    assert pd.isna(cleaned_df["email"].iloc[1])


def test_columns_not_approved_are_excluded_not_guessed():
    df = pd.DataFrame(
        {
            "approved_col": [1, 2, 3],
            "review_col": ["a", "b", "c"],
            "unknown_col": [1, 2, 3],
        }
    )
    table_contract = contract(
        approved_col=approved("numerical"),
        review_col={"semantic_type": "category", "inference_status": "review_required"},
        # unknown_col intentionally absent from the contract entirely
    )
    cleaned_df, report = DataCleaner().clean_table(df, "t", table_contract)

    assert list(cleaned_df.columns) == ["approved_col"]
    reasons = {e["column"]: e["reason"] for e in report["excluded_columns"]}
    assert reasons["review_col"] == "inference_status=review_required"
    assert reasons["unknown_col"] == "not_in_contract"


def test_duplicate_row_count_is_reported():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    table_contract = contract(a=approved("numerical"), b=approved("category"))
    _, report = DataCleaner().clean_table(df, "t", table_contract)
    assert report["duplicate_row_count"] == 1


def test_empty_table_produces_empty_table_warning():
    df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
    table_contract = contract(a=approved("numerical"))
    _, report = DataCleaner().clean_table(df, "empty_t", table_contract)
    assert report["row_count"] == 0
    assert "empty_table" in report["warnings"]
