"""Proves the exact scenario this capability exists for: a source with a
male:female ratio of 4:3 trains a model once; the user can then request
a DIFFERENT ratio (e.g. 2:3) at generation time, repeatedly, without ever
calling fit() again - the same fitted model object, same file on disk,
just a different sample() call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.training.database.adapters.dp_copula_adapter import DPCopulaSynthesizerAdapter
from synth_platform.engine.training.database.adapters.safe_copula_adapter import SafeCopulaSynthesizerAdapter

pytestmark = pytest.mark.unit


def approved(semantic_type: str, physical_type: str = "TEXT") -> dict:
    return {"semantic_type": semantic_type, "inference_status": "approved", "physical_type": physical_type}


def _gendered_df(n: int = 2000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    male_count = round(n * 4 / 7)  # exactly 4:3
    genders = ["male"] * male_count + ["female"] * (n - male_count)
    rng.shuffle(genders)
    return pd.DataFrame(
        {"person_id": [f"P{i:05d}" for i in range(n)], "gender": genders, "age": rng.integers(18, 80, n)}
    )


def _contract() -> dict:
    return {
        "primary_key": ["person_id"],
        "columns": {"person_id": approved("identifier"), "gender": approved("category"), "age": approved("numerical")},
    }


def _male_ratio(series: pd.Series) -> float:
    return float(series.value_counts(normalize=True).get("male", 0.0))


def _value_count(series: pd.Series, value: str) -> int:
    return int((series == value).sum())


def test_safe_adapter_generates_the_source_ratio_by_default_then_2_to_3_on_request():
    df = _gendered_df()
    adapter = SafeCopulaSynthesizerAdapter(category_minimum_support=5)
    adapter.fit(df, "people", _contract(), seed=1)

    # Learned distribution is introspectable before generating anything.
    learned = adapter.get_category_distribution("gender")
    assert learned["male"] == pytest.approx(4 / 7, abs=0.02)

    default_sample = adapter.sample(5000, seed=2)
    assert _male_ratio(default_sample["gender"]) == pytest.approx(4 / 7, abs=0.03)

    # Same fitted adapter, no re-fit call anywhere - just a different
    # sample() invocation with the desired target ratio.
    tweaked_sample = adapter.sample(5000, seed=2, category_overrides={"gender": {"male": 2, "female": 3}})
    assert _value_count(tweaked_sample["gender"], "male") == 2000
    assert _value_count(tweaked_sample["gender"], "female") == 3000

    # And the model itself never changed - a third call with no override
    # still reproduces the ORIGINAL learned ratio.
    reverted_sample = adapter.sample(5000, seed=2)
    assert _male_ratio(reverted_sample["gender"]) == pytest.approx(4 / 7, abs=0.03)


def test_dp_adapter_supports_the_same_tweak_without_spending_additional_privacy_budget():
    df = _gendered_df()
    adapter = DPCopulaSynthesizerAdapter(
        epsilon_budget=5.0, column_bounds={"age": (18, 80)}, category_minimum_support=5
    )
    evidence = adapter.fit(df, "people", _contract(), seed=1)
    spent_before = evidence["privacy_summary"]["total_epsilon_spent"]

    tweaked_sample = adapter.sample(5000, seed=2, category_overrides={"gender": {"male": 2, "female": 3}})
    assert _value_count(tweaked_sample["gender"], "male") == 2000
    assert _value_count(tweaked_sample["gender"], "female") == 3000

    # Rebalancing must never trigger a new DP query - the accountant's
    # recorded total spend from fit() is exactly what it was, unchanged.
    assert adapter._privacy_summary["total_epsilon_spent"] == pytest.approx(spent_before)


def test_tweaking_survives_a_save_and_load_round_trip(tmp_path: Path):
    """The whole point of "no retraining": this must still work after
    the model has been persisted to disk and reloaded in a fresh process,
    not just on the original in-memory adapter object.
    """
    df = _gendered_df()
    adapter = SafeCopulaSynthesizerAdapter(category_minimum_support=5)
    adapter.fit(df, "people", _contract(), seed=1)
    model_path = tmp_path / "model.json"
    adapter.save(model_path)

    loaded = SafeCopulaSynthesizerAdapter.load(model_path)
    tweaked = loaded.sample(5000, seed=2, category_overrides={"gender": {"male": 1, "female": 4}})
    assert _value_count(tweaked["gender"], "male") == 1000
    assert _value_count(tweaked["gender"], "female") == 4000


def test_override_for_an_unknown_column_raises():
    df = _gendered_df()
    adapter = SafeCopulaSynthesizerAdapter(category_minimum_support=5)
    adapter.fit(df, "people", _contract(), seed=1)
    with pytest.raises(Exception):
        adapter.sample(10, category_overrides={"not_a_real_column": {"x": 1}})
