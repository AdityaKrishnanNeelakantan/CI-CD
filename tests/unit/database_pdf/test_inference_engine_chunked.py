"""Unit tests for SemanticInferenceEngine.infer_table_chunked - the
memory-bounded counterpart to infer_table(). Unlike cleaning's chunked
report, inference makes a classification decision that depends on exact
distinct_count/distinct_ratio across the WHOLE column, so these tests
specifically exercise cases that would break under a naive
sum-per-chunk-distinct-count approximation (e.g. a low-cardinality
category repeated across many chunks).
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.common.database.core.scaling import iter_dataframe_chunks
from synth_platform.engine.inference.database.engine import STATUS_PROPOSED, SemanticInferenceEngine

pytestmark = pytest.mark.unit


def chunked_candidate(name: str, values: list, chunk_size: int) -> dict:
    df = pd.DataFrame({name: values})
    result = SemanticInferenceEngine().infer_table_chunked(
        iter_dataframe_chunks(df, chunk_size=chunk_size), "t"
    )
    return result[name]


def whole_candidate(name: str, values: list) -> dict:
    df = pd.DataFrame({name: values})
    return SemanticInferenceEngine().infer_table(df, "t")[name]


def test_low_cardinality_category_survives_chunking_across_many_chunks():
    # 6 distinct statuses repeated 20x = 120 rows; chunk_size=6 means 20
    # chunks, each individually containing all 6 distinct values. A naive
    # per-chunk-distinct-sum approximation would report ~120 (looks
    # high-cardinality) instead of the true global distinct_count of 6.
    values = ["active", "inactive", "pending", "closed", "archived", "draft"] * 20
    whole = whole_candidate("status", values)
    chunked = chunked_candidate("status", values, chunk_size=6)

    assert whole["semantic_type"] == "category"
    assert chunked["semantic_type"] == "category"
    assert chunked["status"] == STATUS_PROPOSED


def test_identifier_column_stays_identifier_across_chunks():
    values = [f"CUST-{i:04d}" for i in range(20)]
    whole = whole_candidate("customer_id", values)
    chunked = chunked_candidate("customer_id", values, chunk_size=4)

    assert whole["semantic_type"] == chunked["semantic_type"] == "identifier"
    assert chunked["status"] == STATUS_PROPOSED


def test_email_pattern_match_rate_matches_unchunked_with_duplicates():
    # Duplicated values across chunk boundaries - a distinct-only
    # accounting (rather than frequency-preserving) would corrupt the
    # match rate here.
    values = ["ada@example.com", "ada@example.com", "not-an-email", "not-an-email"] * 5
    whole = whole_candidate("email", values)
    chunked = chunked_candidate("email", values, chunk_size=3)

    assert whole["confidence"] == pytest.approx(chunked["confidence"])
    assert whole["semantic_type"] == chunked["semantic_type"]


def test_numeric_column_classification_matches_unchunked():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    whole = whole_candidate("amount", values)
    chunked = chunked_candidate("amount", values, chunk_size=3)
    assert whole["semantic_type"] == chunked["semantic_type"] == "numerical"


def test_float_vs_integer_distinction_resolved_only_after_seeing_all_chunks():
    # First chunk is all-round (looks like "integer"); a later chunk
    # introduces a genuine fractional value - the dtype decision must
    # depend on ALL chunks, not just the first one seen.
    values = [1.0, 2.0, 3.0, 4.5, 5.0, 6.0]
    whole = whole_candidate("x", values)
    chunked = chunked_candidate("x", values, chunk_size=3)
    assert whole["semantic_type"] == chunked["semantic_type"]


def test_primary_key_hint_applied_in_chunked_mode():
    df = pd.DataFrame({"id": [f"ROW-{i}" for i in range(10)]})
    result = SemanticInferenceEngine().infer_table_chunked(
        iter_dataframe_chunks(df, chunk_size=4), "t", discovery_table={"primary_key": ["id"]}
    )
    assert "primary_key=true" in result["id"]["evidence"]


def test_profile_warnings_applied_in_chunked_mode():
    df = pd.DataFrame({"status": ["active"] * 10})
    profile_table = {"columns": {"status": {"warnings": ["constant_column"]}}}
    result = SemanticInferenceEngine().infer_table_chunked(
        iter_dataframe_chunks(df, chunk_size=3), "t", profile_table=profile_table
    )
    assert any("constant_column" in e for e in result["status"]["evidence"])


def test_infer_table_chunked_handles_empty_iterator():
    result = SemanticInferenceEngine().infer_table_chunked(iter([]), "t")
    assert result == {}
