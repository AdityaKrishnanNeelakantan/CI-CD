"""Contract: infer_table() always returns the same candidate shape per
column, regardless of what the data looks like - downstream code (contract
approval, and eventually a Streamlit review page) must be able to rely on
this without checking "did this column get a category_frequencies-style
optional field" first.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.inference.database.engine import SemanticInferenceEngine

pytestmark = pytest.mark.contract

REQUIRED_CANDIDATE_KEYS = {"column", "semantic_type", "status", "confidence", "evidence", "alternatives"}

FIXTURE_FRAMES = {
    "identifier": pd.DataFrame({"customer_id": ["a1", "a2", "a3", "a4"]}),
    "numeric": pd.DataFrame({"amount": [1.5, 2.5, 3.5]}),
    "empty": pd.DataFrame({"col": pd.Series([], dtype="object")}),
    "all_null": pd.DataFrame({"col": [None, None]}),
    "mixed": pd.DataFrame({"col": [1, "two", None]}),
}


@pytest.mark.parametrize("df", FIXTURE_FRAMES.values(), ids=FIXTURE_FRAMES.keys())
def test_candidate_shape_is_invariant(df: pd.DataFrame):
    result = SemanticInferenceEngine().infer_table(df, "t")
    for column_name, candidate in result.items():
        assert set(candidate) == REQUIRED_CANDIDATE_KEYS
        assert candidate["column"] == column_name
        assert candidate["status"] in {"proposed", "REVIEW_REQUIRED"}
        assert 0.0 <= candidate["confidence"] <= 1.0
        assert isinstance(candidate["evidence"], list)
        assert isinstance(candidate["alternatives"], list)
        for alt in candidate["alternatives"]:
            assert set(alt) == {"semantic_type", "confidence"}


def test_candidate_output_is_json_serialisable():
    import json

    df = pd.DataFrame({"customer_id": ["a1", "a2"], "amount": [1.5, 2.5]})
    result = SemanticInferenceEngine().infer_table(df, "t")
    json.dumps(result)  # must not raise
