from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.training.database.adapters.category_rebalancing import (
    CategoryOverrideError,
    build_rebalanced_encoder,
    normalize_category_weights,
    rebalance_column,
)
from synth_platform.engine.training.database.adapters.copula_encoders import CategoricalEncoder

pytestmark = pytest.mark.unit


def test_normalize_category_weights_sums_to_one():
    normalized = normalize_category_weights({"male": 2, "female": 3})
    assert sum(normalized.values()) == pytest.approx(1.0)
    assert normalized["male"] == pytest.approx(0.4)
    assert normalized["female"] == pytest.approx(0.6)


def test_normalize_rejects_non_positive_total():
    with pytest.raises(CategoryOverrideError):
        normalize_category_weights({"a": 0, "b": 0})


def test_build_rebalanced_encoder_requires_every_known_category():
    original = CategoricalEncoder.from_frequencies({"male": 400, "female": 300})
    with pytest.raises(CategoryOverrideError):
        build_rebalanced_encoder(original, {"male": 0.5})  # missing "female"


def test_build_rebalanced_encoder_rejects_unknown_categories():
    original = CategoricalEncoder.from_frequencies({"male": 400, "female": 300})
    with pytest.raises(CategoryOverrideError):
        build_rebalanced_encoder(original, {"male": 0.5, "female": 0.3, "nonbinary": 0.2})


def test_build_rebalanced_encoder_produces_the_requested_proportions():
    original = CategoricalEncoder.from_frequencies({"male": 400, "female": 300})  # 4:3
    rebalanced = build_rebalanced_encoder(original, {"male": 2, "female": 3})  # -> 2:3
    proportions = rebalanced.get_proportions()
    assert proportions["male"] == pytest.approx(0.4, abs=1e-6)
    assert proportions["female"] == pytest.approx(0.6, abs=1e-6)


def test_rebalance_column_shifts_the_observed_ratio_from_4_3_to_2_3():
    # Simulate an already-sampled numeric-space column under the ORIGINAL
    # 4:3 encoding: values uniformly spread across [0, 1].
    import numpy as np

    rng = np.random.default_rng(1)
    sampled_encoded = pd.Series(rng.uniform(0, 1, 10_000))

    original = CategoricalEncoder.from_frequencies({"male": 4, "female": 3})
    original_decoded = original.decode(sampled_encoded)
    original_ratio = (original_decoded == "male").mean()
    assert original_ratio == pytest.approx(4 / 7, abs=0.02)

    rebalanced_decoded = rebalance_column(sampled_encoded, original, {"male": 2, "female": 3})
    rebalanced_ratio = (rebalanced_decoded == "male").mean()
    assert rebalanced_ratio == pytest.approx(2 / 5, abs=0.02)


def test_rebalance_never_requires_touching_the_original_model_state():
    """The whole point: rebalancing is a pure function of the already-
    sampled values and the override weights - it never mutates the
    original encoder (no re-fitting, no state change).
    """
    original = CategoricalEncoder.from_frequencies({"male": 4, "female": 3})
    original_dict_before = original.to_dict()
    rebalance_column(pd.Series([0.1, 0.9]), original, {"male": 2, "female": 3})
    assert original.to_dict() == original_dict_before
