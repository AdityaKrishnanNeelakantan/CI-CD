from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.common.database.privacy.dp_primitives import (
    InvalidPrivacyParameterError,
    clip_series,
    dp_count,
    gaussian_mechanism,
    gaussian_mechanism_sigma,
    laplace_mechanism,
)

pytestmark = pytest.mark.unit


def test_clip_series_bounds_values():
    series = pd.Series([-10, 0, 5, 100])
    clipped = clip_series(series, lower=0, upper=10)
    assert list(clipped) == [0, 0, 5, 10]


def test_clip_series_rejects_inverted_bounds():
    with pytest.raises(InvalidPrivacyParameterError):
        clip_series(pd.Series([1, 2]), lower=10, upper=0)


def test_laplace_mechanism_is_unbiased_in_expectation():
    """Laplace noise has mean 0 - averaged over many draws, the noisy
    value should converge to the true value.
    """
    rng = np.random.default_rng(1)
    true_value = 100.0
    samples = [laplace_mechanism(true_value, sensitivity=1.0, epsilon=1.0, rng=rng) for _ in range(20_000)]
    assert abs(np.mean(samples) - true_value) < 1.0


def test_laplace_mechanism_noise_scale_increases_as_epsilon_shrinks():
    """Smaller epsilon = stronger privacy = more noise. This is the whole
    point of the mechanism, and must be independently verified, not just
    assumed from the formula.
    """
    rng_tight = np.random.default_rng(1)
    rng_loose = np.random.default_rng(1)
    tight_privacy_samples = [laplace_mechanism(0.0, sensitivity=1.0, epsilon=0.01, rng=rng_tight) for _ in range(5000)]
    loose_privacy_samples = [laplace_mechanism(0.0, sensitivity=1.0, epsilon=10.0, rng=rng_loose) for _ in range(5000)]
    assert np.std(tight_privacy_samples) > np.std(loose_privacy_samples) * 10


def test_laplace_mechanism_rejects_non_positive_epsilon():
    with pytest.raises(InvalidPrivacyParameterError):
        laplace_mechanism(1.0, sensitivity=1.0, epsilon=0.0, rng=np.random.default_rng())


def test_gaussian_mechanism_sigma_matches_the_classical_formula():
    import math

    sigma = gaussian_mechanism_sigma(sensitivity=2.0, epsilon=0.5, delta=1e-5)
    expected = 2.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 0.5
    assert sigma == pytest.approx(expected)


def test_gaussian_mechanism_sigma_rejects_epsilon_outside_proof_range():
    with pytest.raises(InvalidPrivacyParameterError):
        gaussian_mechanism_sigma(sensitivity=1.0, epsilon=1.5, delta=1e-5)
    with pytest.raises(InvalidPrivacyParameterError):
        gaussian_mechanism_sigma(sensitivity=1.0, epsilon=0.0, delta=1e-5)


def test_gaussian_mechanism_rejects_delta_outside_range():
    with pytest.raises(InvalidPrivacyParameterError):
        gaussian_mechanism_sigma(sensitivity=1.0, epsilon=0.5, delta=0.0)
    with pytest.raises(InvalidPrivacyParameterError):
        gaussian_mechanism_sigma(sensitivity=1.0, epsilon=0.5, delta=1.0)


def test_gaussian_mechanism_adds_elementwise_noise_to_a_matrix():
    rng = np.random.default_rng(1)
    matrix = np.eye(3)
    noisy = gaussian_mechanism(matrix, sensitivity=1.0, epsilon=0.5, delta=1e-5, rng=rng)
    assert noisy.shape == (3, 3)
    assert not np.allclose(noisy, matrix)  # noise was actually added


def test_dp_count_has_sensitivity_one_laplace_noise():
    rng = np.random.default_rng(1)
    samples = [dp_count(1000, epsilon=1.0, rng=rng) for _ in range(5000)]
    assert abs(np.mean(samples) - 1000) < 5
    # scale should be sensitivity(1)/epsilon(1) = 1.0 -> std of Laplace(0,1) is sqrt(2)
    assert 1.0 < np.std(samples) < 2.0


def test_same_seed_reproduces_identical_noise():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = laplace_mechanism(50.0, sensitivity=1.0, epsilon=1.0, rng=rng_a)
    b = laplace_mechanism(50.0, sensitivity=1.0, epsilon=1.0, rng=rng_b)
    assert a == b
