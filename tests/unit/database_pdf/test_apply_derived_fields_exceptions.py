from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.generation.database.constraint_engine import apply_derived_fields

pytestmark = pytest.mark.unit


def test_apply_derived_fields_bad_formula_leaves_nan_not_crash():
    df = pd.DataFrame({"credit_limit": [1000], "balance": [0]})
    result = apply_derived_fields(
        df, [{"target_column": "ratio", "formula": "credit_limit / balance"}]
    )
    assert pd.isna(result["ratio"].iloc[0])


def test_apply_derived_fields_missing_column_leaves_nan():
    df = pd.DataFrame({"a": [1]})
    result = apply_derived_fields(
        df, [{"target_column": "b", "formula": "missing_col + 1"}]
    )
    assert pd.isna(result["b"].iloc[0])
