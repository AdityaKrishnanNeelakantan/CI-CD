from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.common.database.privacy.dp_statistics import dp_category_histogram, dp_correlation_matrix, dp_mean_and_variance
from synth_platform.engine.common.database.privacy.privacy_accountant import PrivacyAccountant

pytestmark = pytest.mark.unit


def test_dp_mean_and_variance_are_reasonably_close_to_true_values_for_generous_epsilon():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(loc=50.0, scale=5.0, size=2000))
    accountant = PrivacyAccountant(epsilon_budget=100.0)

    dp_mean, dp_variance = dp_mean_and_variance(
        series, lower=0.0, upper=100.0, epsilon=10.0, accountant=accountant, rng=rng, purpose="amount"
    )
    assert abs(dp_mean - 50.0) < 2.0
    assert abs(dp_variance - 25.0) < 15.0


def test_dp_mean_and_variance_spends_exactly_epsilon_split_evenly():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(50, 5, 500))
    accountant = PrivacyAccountant(epsilon_budget=1.0)
    dp_mean_and_variance(series, 0, 100, epsilon=1.0, accountant=accountant, rng=rng, purpose="amount")
    assert accountant.total_epsilon() == pytest.approx(1.0)
    purposes = {e.purpose for e in accountant._expenditures}
    assert purposes == {"amount:mean", "amount:variance"}


def test_dp_mean_and_variance_never_produces_negative_variance():
    rng = np.random.default_rng(1)
    # Tiny epsilon -> huge noise, which could easily drive the noisy sum
    # of squared deviations negative before clamping.
    series = pd.Series([50.0] * 20)
    for _ in range(50):
        _, variance = dp_mean_and_variance(
            series, 0, 100, epsilon=0.001, accountant=PrivacyAccountant(epsilon_budget=10.0), rng=rng, purpose="x"
        )
        assert variance >= 0.0


def test_dp_mean_noise_shrinks_as_epsilon_grows():
    series = pd.Series(np.full(500, 50.0))
    tight_means = [
        dp_mean_and_variance(series, 0, 100, epsilon=0.05, accountant=PrivacyAccountant(10.0), rng=np.random.default_rng(seed), purpose="x")[0]
        for seed in range(30)
    ]
    loose_means = [
        dp_mean_and_variance(series, 0, 100, epsilon=5.0, accountant=PrivacyAccountant(10.0), rng=np.random.default_rng(seed), purpose="x")[0]
        for seed in range(30)
    ]
    assert np.std(tight_means) > np.std(loose_means)


def test_dp_category_histogram_recovers_dominant_categories_for_generous_epsilon():
    series = pd.Series(["active"] * 900 + ["inactive"] * 100)
    accountant = PrivacyAccountant(epsilon_budget=10.0)
    rng = np.random.default_rng(1)
    histogram = dp_category_histogram(series, epsilon=5.0, accountant=accountant, rng=rng, purpose="status")
    assert "active" in histogram
    assert abs(histogram["active"] - 900) < 20


def test_dp_category_histogram_drops_categories_below_noisy_threshold():
    series = pd.Series(["common"] * 500 + ["exceedingly_rare_value"] * 1)
    accountant = PrivacyAccountant(epsilon_budget=10.0)
    rng = np.random.default_rng(1)
    histogram = dp_category_histogram(
        series, epsilon=0.5, accountant=accountant, rng=rng, purpose="status", minimum_noisy_count=20
    )
    assert "exceedingly_rare_value" not in histogram


def test_dp_category_histogram_spends_epsilon_once_not_per_category():
    series = pd.Series(["a", "b", "c", "a", "b"])
    accountant = PrivacyAccountant(epsilon_budget=1.0)
    dp_category_histogram(series, epsilon=1.0, accountant=accountant, rng=np.random.default_rng(1), purpose="x")
    assert accountant.total_epsilon() == pytest.approx(1.0)


def test_dp_correlation_matrix_is_symmetric_unit_diagonal_and_psd():
    rng = np.random.default_rng(1)
    base = rng.normal(size=1000)
    df = pd.DataFrame({"a": base, "b": base + rng.normal(scale=0.1, size=1000), "c": rng.normal(size=1000)})
    standardized = (df - df.mean()) / df.std()

    accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-4)
    correlation = dp_correlation_matrix(
        standardized, epsilon=0.5, delta=1e-5, accountant=accountant, rng=rng, purpose="corr"
    )

    assert np.allclose(correlation, correlation.T)
    assert np.allclose(np.diag(correlation), 1.0)
    eigenvalues = np.linalg.eigvalsh(correlation)
    assert (eigenvalues >= -1e-8).all()


def test_dp_correlation_matrix_roughly_recovers_a_strong_correlation():
    rng = np.random.default_rng(1)
    base = rng.normal(size=5000)
    df = pd.DataFrame({"a": base, "b": base + rng.normal(scale=0.05, size=5000)})
    standardized = (df - df.mean()) / df.std()

    accountant = PrivacyAccountant(epsilon_budget=5.0, delta_budget=1e-3)
    correlation = dp_correlation_matrix(
        standardized, epsilon=0.5, delta=1e-4, accountant=accountant, rng=rng, purpose="corr"
    )
    assert correlation[0, 1] > 0.7  # true correlation is ~0.999; generous noise tolerance


def test_dp_correlation_matrix_spends_epsilon_and_delta():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-4)
    dp_correlation_matrix(df, epsilon=0.5, delta=1e-5, accountant=accountant, rng=np.random.default_rng(1), purpose="corr")
    assert accountant.total_epsilon() == pytest.approx(0.5)
    assert accountant.total_delta() == pytest.approx(1e-5)


def test_empty_dataframe_returns_empty_correlation_matrix():
    accountant = PrivacyAccountant(epsilon_budget=1.0)
    result = dp_correlation_matrix(
        pd.DataFrame(index=range(5)), epsilon=0.5, delta=1e-5, accountant=accountant, rng=np.random.default_rng(1), purpose="corr"
    )
    assert result.shape == (0, 0)
