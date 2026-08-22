from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.common.database.core.fingerprint import compute_dataframe_fingerprint, compute_fingerprint

pytestmark = pytest.mark.unit


def test_fingerprint_is_deterministic_regardless_of_key_order():
    a = compute_fingerprint({"tables": {"x": 1, "y": 2}, "source_type": "sqlite"})
    b = compute_fingerprint({"source_type": "sqlite", "tables": {"y": 2, "x": 1}})
    assert a == b


def test_fingerprint_changes_with_content():
    a = compute_fingerprint({"tables": {"x": 1}})
    b = compute_fingerprint({"tables": {"x": 2}})
    assert a != b


def test_fingerprint_has_sha256_prefix():
    result = compute_fingerprint({"a": 1})
    assert result.startswith("sha256:")
    assert len(result) == len("sha256:") + 64


def test_dataframe_fingerprint_is_deterministic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert compute_dataframe_fingerprint(df) == compute_dataframe_fingerprint(df.copy())


def test_dataframe_fingerprint_changes_with_content():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 4]})
    assert compute_dataframe_fingerprint(df1) != compute_dataframe_fingerprint(df2)


def test_dataframe_fingerprint_changes_with_row_order():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [3, 2, 1]})
    assert compute_dataframe_fingerprint(df1) != compute_dataframe_fingerprint(df2)


def test_empty_dataframe_fingerprint_is_stable_and_column_sensitive():
    empty_a = pd.DataFrame({"a": pd.Series([], dtype="int64")})
    empty_b = pd.DataFrame({"b": pd.Series([], dtype="int64")})
    assert compute_dataframe_fingerprint(empty_a) == compute_dataframe_fingerprint(empty_a.copy())
    assert compute_dataframe_fingerprint(empty_a) != compute_dataframe_fingerprint(empty_b)
