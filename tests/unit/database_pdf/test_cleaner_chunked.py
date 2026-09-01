"""Unit tests for DataCleaner.clean_table_chunked - the memory-bounded
counterpart to clean_table(), which streams chunks once and accumulates
report statistics without holding the whole table in memory (only a
numerical column's own non-null values, when present, are briefly
concatenated to compute an exact global median/IQR).
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.profiling.database.cleaning.cleaner import CleaningConfig, DataCleaner
from synth_platform.engine.common.database.core.scaling import iter_dataframe_chunks

pytestmark = pytest.mark.unit


def approved(semantic_type: str) -> dict:
    return {"semantic_type": semantic_type, "inference_status": "approved"}


def contract(**columns: dict) -> dict:
    return {"columns": columns}


def test_clean_table_chunked_matches_unchunked_numerical_report():
    df = pd.DataFrame({"amount": [10.0, 12.0, 11.0, 13.0, None, 1000.0]})
    table_contract = contract(amount=approved("numerical"))
    cleaner = DataCleaner()

    _cleaned_df, whole = cleaner.clean_table(df, "t", table_contract)
    chunked = cleaner.clean_table_chunked(iter_dataframe_chunks(df, chunk_size=2), "t", table_contract)

    col_whole = whole["columns"]["amount"]
    col_chunked = chunked["columns"]["amount"]
    assert col_chunked["action"] == col_whole["action"] == "imputed_median"
    assert col_chunked["nulls_before"] == col_whole["nulls_before"] == 1
    assert col_chunked["nulls_imputed"] == col_whole["nulls_imputed"] == 1
    assert col_chunked["nulls_after"] == col_whole["nulls_after"] == 0
    assert col_chunked["outliers_flagged"] == col_whole["outliers_flagged"] == 1
    assert chunked["row_count"] == whole["row_count"] == 6


def test_clean_table_chunked_matches_unchunked_mean_strategy():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, None]})
    table_contract = contract(x=approved("numerical"))
    cleaner = DataCleaner(CleaningConfig(numerical_imputation_strategy="mean"))

    _cleaned_df, whole = cleaner.clean_table(df, "t", table_contract)
    chunked = cleaner.clean_table_chunked(iter_dataframe_chunks(df, chunk_size=2), "t", table_contract)

    assert chunked["columns"]["x"]["action"] == whole["columns"]["x"]["action"] == "imputed_mean"
    assert chunked["columns"]["x"]["nulls_imputed"] == whole["columns"]["x"]["nulls_imputed"]


def test_clean_table_chunked_tracks_coercion_failures_across_chunk_boundary():
    # "not-a-number" and "4" fall in different chunks (chunk_size=2).
    df = pd.DataFrame({"x": ["1", "2", "not-a-number", "4"]})
    table_contract = contract(x=approved("numerical"))
    cleaner = DataCleaner()

    _cleaned_df, whole = cleaner.clean_table(df, "t", table_contract)
    chunked = cleaner.clean_table_chunked(iter_dataframe_chunks(df, chunk_size=2), "t", table_contract)

    assert chunked["columns"]["x"]["coercion_failures"] == whole["columns"]["x"]["coercion_failures"] == 1
    assert chunked["columns"]["x"]["nulls_imputed"] == whole["columns"]["x"]["nulls_imputed"] == 1


def test_clean_table_chunked_matches_unchunked_category_mode_across_chunks():
    df = pd.DataFrame({"status": ["active", "active", "inactive", None]})
    table_contract = contract(status=approved("category"))
    cleaner = DataCleaner()

    _cleaned_df, whole = cleaner.clean_table(df, "t", table_contract)
    chunked = cleaner.clean_table_chunked(iter_dataframe_chunks(df, chunk_size=2), "t", table_contract)

    assert chunked["columns"]["status"]["action"] == whole["columns"]["status"]["action"] == "imputed_mode"
    assert chunked["columns"]["status"]["nulls_imputed"] == whole["columns"]["status"]["nulls_imputed"] == 1


def test_clean_table_chunked_category_sentinel_strategy():
    df = pd.DataFrame({"status": ["active", "inactive", None]})
    table_contract = contract(status=approved("category"))
    cleaner = DataCleaner(CleaningConfig(category_imputation_strategy="sentinel"))

    chunked = cleaner.clean_table_chunked(iter_dataframe_chunks(df, chunk_size=2), "t", table_contract)
    assert chunked["columns"]["status"]["action"] == "imputed_sentinel"
    assert chunked["columns"]["status"]["nulls_imputed"] == 1


def test_clean_table_chunked_identifier_duplicate_detection_spans_chunks():
    # "A1" appears in chunk 1 (rows 0-1) and again in chunk 2 (rows 2-3) -
    # a naive per-chunk-independent duplicate count would miss this.
    df = pd.DataFrame({"id": ["A1", "A2", "A1", "A3"]})
    table_contract = contract(id=approved("identifier"))
    cleaner = DataCleaner()

    _cleaned_df, whole = cleaner.clean_table(df, "t", table_contract)
    chunked = cleaner.clean_table_chunked(iter_dataframe_chunks(df, chunk_size=2), "t", table_contract)

    assert chunked["columns"]["id"]["duplicate_value_count"] == 1
    assert chunked["columns"]["id"]["duplicate_value_count"] == whole["columns"]["id"]["duplicate_value_count"]


def test_clean_table_chunked_whole_row_duplicate_detection_spans_chunks():
    # The row (1, "a") appears in chunk 1 and again in chunk 2.
    df = pd.DataFrame({"n": [1, 2, 1, 3], "s": ["a", "b", "a", "c"]})
    table_contract = contract(n=approved("numerical"), s=approved("category"))
    cleaner = DataCleaner()

    _cleaned_df, whole = cleaner.clean_table(df, "t", table_contract)
    chunked = cleaner.clean_table_chunked(iter_dataframe_chunks(df, chunk_size=2), "t", table_contract)

    assert chunked["duplicate_row_count"] == whole["duplicate_row_count"] == 1


def test_clean_table_chunked_datetime_coercion_matches_unchunked():
    df = pd.DataFrame({"d": ["2023-01-01", "not-a-date", "2023-02-01", None]})
    table_contract = contract(d=approved("datetime"))
    cleaner = DataCleaner()

    _cleaned_df, whole = cleaner.clean_table(df, "t", table_contract)
    chunked = cleaner.clean_table_chunked(iter_dataframe_chunks(df, chunk_size=2), "t", table_contract)

    assert chunked["columns"]["d"]["nulls_after"] == whole["columns"]["d"]["nulls_after"]
    assert chunked["columns"]["d"]["coercion_failures"] == whole["columns"]["d"]["coercion_failures"] == 1


def test_clean_table_chunked_excludes_unapproved_and_unknown_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    table_contract = contract(
        a=approved("numerical"),
        b={"semantic_type": "numerical", "inference_status": "review_required"},
    )
    cleaner = DataCleaner()
    chunked = cleaner.clean_table_chunked(iter_dataframe_chunks(df, chunk_size=1), "t", table_contract)

    reasons = {entry["column"]: entry["reason"] for entry in chunked["excluded_columns"]}
    assert reasons["b"] == "inference_status=review_required"
    assert reasons["c"] == "not_in_contract"
    assert list(chunked["columns"].keys()) == ["a"]


def test_clean_table_chunked_reports_chunked_cleaning_mode_warning():
    df = pd.DataFrame({"a": [1, 2, 3]})
    table_contract = contract(a=approved("numerical"))
    chunked = DataCleaner().clean_table_chunked(iter_dataframe_chunks(df, chunk_size=2), "t", table_contract)
    assert "chunked_cleaning_mode" in chunked["warnings"]


def test_clean_table_chunked_handles_empty_iterator():
    table_contract = contract(a=approved("numerical"))
    chunked = DataCleaner().clean_table_chunked(iter([]), "t", table_contract)
    assert chunked["row_count"] == 0
    assert chunked["duplicate_row_count"] == 0
    assert chunked["columns"] == {}
    assert "empty_table" in chunked["warnings"]
