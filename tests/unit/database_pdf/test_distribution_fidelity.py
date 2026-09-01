"""Verifies the actual claim behind every synthesizer adapter in this
project: generated categorical proportions should replicate the source's
real proportions (e.g. a 4:3 male:female ratio in the source should stay
close to 4:3 in generated output), not just "some distribution or other".

This is deliberately a statistical property test, not a hand-derived
expectation - it fits each adapter on a real DataFrame with a KNOWN,
constructed ratio and checks the generated ratio lands close to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.training.database.adapters.dp_copula_adapter import DPCopulaSynthesizerAdapter
from synth_platform.engine.training.database.adapters.safe_copula_adapter import SafeCopulaSynthesizerAdapter
from synth_platform.engine.training.database.adapters.sdv_adapter import SDVSynthesizerAdapter

pytestmark = pytest.mark.unit


def approved(semantic_type: str, physical_type: str = "TEXT") -> dict:
    return {"semantic_type": semantic_type, "inference_status": "approved", "physical_type": physical_type}


def _gendered_df(n: int = 2000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # male:female = 4:3 exactly, by construction - not an approximation
    # of some other split.
    male_count = round(n * 4 / 7)
    genders = ["male"] * male_count + ["female"] * (n - male_count)
    rng.shuffle(genders)
    return pd.DataFrame(
        {
            "person_id": [f"P{i:05d}" for i in range(n)],
            "gender": genders,
            "age": rng.integers(18, 80, n),
        }
    )


def _contract() -> dict:
    return {
        "primary_key": ["person_id"],
        "columns": {
            "person_id": approved("identifier"),
            "gender": approved("category"),
            "age": approved("numerical"),
        },
    }


def _male_ratio(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True)
    return float(counts.get("male", 0.0))


def test_safe_copula_adapter_replicates_a_4_to_3_ratio():
    df = _gendered_df()
    true_ratio = _male_ratio(df["gender"])  # ~0.5714
    assert true_ratio == pytest.approx(4 / 7, abs=0.01)

    adapter = SafeCopulaSynthesizerAdapter(category_minimum_support=5)
    adapter.fit(df, "people", _contract(), seed=1)
    generated = adapter.sample(5000, seed=2)

    generated_ratio = _male_ratio(generated["gender"])
    assert abs(generated_ratio - true_ratio) < 0.03


def test_sdv_adapter_replicates_a_4_to_3_ratio():
    df = _gendered_df()
    true_ratio = _male_ratio(df["gender"])

    adapter = SDVSynthesizerAdapter()
    adapter.fit(df, "people", _contract(), seed=1)
    generated = adapter.sample(5000, seed=2)

    generated_ratio = _male_ratio(generated["gender"])
    assert abs(generated_ratio - true_ratio) < 0.03


def test_dp_copula_adapter_roughly_replicates_a_4_to_3_ratio_at_reasonable_epsilon():
    df = _gendered_df()
    true_ratio = _male_ratio(df["gender"])

    adapter = DPCopulaSynthesizerAdapter(
        epsilon_budget=5.0, column_bounds={"age": (18, 80)}, category_minimum_support=5
    )
    adapter.fit(df, "people", _contract(), seed=1)
    generated = adapter.sample(5000, seed=2)

    generated_ratio = _male_ratio(generated["gender"])
    # Generous tolerance - DP noise is expected to shift this somewhat;
    # the property under test is "roughly preserved", not "exact".
    assert abs(generated_ratio - true_ratio) < 0.08


def test_replication_holds_for_a_three_way_category_split():
    rng = np.random.default_rng(1)
    n = 3000
    # active:inactive:pending = 6:3:1
    categories = ["active"] * 1800 + ["inactive"] * 900 + ["pending"] * 300
    rng.shuffle(categories)
    df = pd.DataFrame({"id": [f"P{i}" for i in range(n)], "status": categories, "age": rng.integers(18, 80, n)})
    contract = {
        "primary_key": ["id"],
        "columns": {"id": approved("identifier"), "status": approved("category"), "age": approved("numerical")},
    }

    adapter = SafeCopulaSynthesizerAdapter(category_minimum_support=5)
    adapter.fit(df, "t", contract, seed=1)
    generated = adapter.sample(5000, seed=2)

    true_props = pd.Series(categories).value_counts(normalize=True)
    generated_props = generated["status"].value_counts(normalize=True)
    for category in true_props.index:
        assert abs(generated_props.get(category, 0.0) - true_props[category]) < 0.03
