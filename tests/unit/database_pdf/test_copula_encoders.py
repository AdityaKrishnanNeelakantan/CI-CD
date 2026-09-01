from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.common.database.privacy.rare_category import RARE_CATEGORY_LABEL
from synth_platform.engine.training.database.adapters.copula_encoders import CategoricalEncoder

pytestmark = pytest.mark.unit


def test_from_frequencies_builds_a_working_encoder():
    encoder = CategoricalEncoder.from_frequencies({"active": 90, "inactive": 10})
    assert encoder.to_dict()["categories"] == ["active", "inactive"]
    decoded = encoder.decode(pd.Series([0.01, 0.95]))
    assert list(decoded) == ["active", "inactive"]


def test_from_frequencies_matches_fit_on_the_same_underlying_counts():
    series = pd.Series(["active"] * 90 + ["inactive"] * 10)
    fitted = CategoricalEncoder(minimum_support=1)
    fitted.fit(series)

    from_freq = CategoricalEncoder.from_frequencies({"active": 90, "inactive": 10}, minimum_support=1)
    assert fitted.to_dict()["categories"] == from_freq.to_dict()["categories"]
    assert fitted.to_dict()["boundaries"] == pytest.approx(from_freq.to_dict()["boundaries"])


def test_from_frequencies_handles_empty_input():
    # An empty histogram (e.g. every DP-noised count fell below the
    # suppression threshold) must still collapse to a single valid
    # bucket, not a degenerate one-boundary encoder.
    encoder = CategoricalEncoder.from_frequencies({})
    assert encoder.to_dict()["categories"] == [RARE_CATEGORY_LABEL]
    assert encoder.to_dict()["boundaries"] == [0.0, 1.0]


def test_from_frequencies_empty_input_can_encode_and_decode_real_values_without_crashing():
    # Regression: previously _boundaries=[0.0] (length 1) made encode()
    # raise IndexError on self._boundaries[index + 1] for any real value.
    encoder = CategoricalEncoder.from_frequencies({})
    rng = np.random.default_rng(0)
    series = pd.Series(["male", "female", "male"])

    encoded = encoder.encode(series, rng)
    assert len(encoded) == 3

    decoded = encoder.decode(encoded)
    assert list(decoded) == [RARE_CATEGORY_LABEL] * 3


def test_get_proportions_matches_the_original_frequencies():
    encoder = CategoricalEncoder.from_frequencies({"male": 400, "female": 300})
    proportions = encoder.get_proportions()
    assert proportions["male"] == pytest.approx(4 / 7, abs=1e-6)
    assert proportions["female"] == pytest.approx(3 / 7, abs=1e-6)
    assert sum(proportions.values()) == pytest.approx(1.0)
