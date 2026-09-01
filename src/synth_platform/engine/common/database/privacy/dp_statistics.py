"""Higher-level DP statistics built from src/privacy/dp_primitives.py's
mechanisms, each one an explicit query against a
src/privacy/privacy_accountant.py accountant - every function here both
computes a noised statistic AND records exactly what it cost, so a
caller can never use one without the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synth_platform.engine.common.database.privacy.dp_primitives import (
    clip_series,
    dp_count,
    gaussian_mechanism,
    laplace_mechanism,
)
from synth_platform.engine.common.database.privacy.privacy_accountant import PrivacyAccountant


def dp_mean_and_variance(
    series: pd.Series,
    lower: float,
    upper: float,
    epsilon: float,
    accountant: PrivacyAccountant,
    rng: np.random.Generator,
    purpose: str,
) -> tuple[float, float]:
    """Splits epsilon evenly between a DP mean query and a DP variance
    query (sequential composition). Both operate on values already
    clipped to [lower, upper], so each has a fixed, known sensitivity
    regardless of the data's real range.
    """
    clipped = clip_series(series, lower, upper)
    n = len(clipped)
    if n == 0:
        raise ValueError("cannot compute DP mean/variance of an empty series")

    mean_epsilon = epsilon / 2
    variance_epsilon = epsilon / 2

    # Sensitivity of the clipped sum under one row changing: at most
    # (upper - lower) under the "replace one record" neighbouring relation.
    sum_sensitivity = upper - lower
    true_sum = float(clipped.sum())
    noisy_sum = laplace_mechanism(true_sum, sensitivity=sum_sensitivity, epsilon=mean_epsilon, rng=rng)
    accountant.spend(mean_epsilon, purpose=f"{purpose}:mean")
    dp_mean = noisy_sum / n

    # Sum of squared deviations from the DP mean, each term bounded by
    # (upper-lower)^2 since every clipped value lies in [lower, upper].
    squared_deviation_sensitivity = (upper - lower) ** 2
    true_sum_sq_dev = float(((clipped - dp_mean) ** 2).sum())
    noisy_sum_sq_dev = laplace_mechanism(
        true_sum_sq_dev, sensitivity=squared_deviation_sensitivity, epsilon=variance_epsilon, rng=rng
    )
    accountant.spend(variance_epsilon, purpose=f"{purpose}:variance")
    dp_variance = max(noisy_sum_sq_dev / n, 0.0)  # variance can't be negative - post-processing, not a privacy cost

    return dp_mean, dp_variance


def dp_category_histogram(
    series: pd.Series,
    epsilon: float,
    accountant: PrivacyAccountant,
    rng: np.random.Generator,
    purpose: str,
    minimum_noisy_count: float = 1.0,
) -> dict[str, float]:
    """DP-noised category counts: every observed category gets Laplace
    noise added to its true count (sensitivity 1 per category - a single
    row can only ever affect one category's count, so this whole
    histogram is spent as ONE epsilon expenditure via parallel
    composition, not one per category).

    Categories whose *noisy* count falls below minimum_noisy_count are
    dropped - thresholding an already-released noisy value costs no
    additional budget, though this is a practical approximation of
    formal DP partition/stability selection, not a rigorous mechanism of
    its own (a real "which categories exist at all" DP guarantee needs
    one - out of scope here, disclosed rather than overclaimed).
    """
    counts = series.value_counts()
    noisy_counts = {
        str(category): dp_count(int(count), epsilon=epsilon, rng=rng) for category, count in counts.items()
    }
    accountant.spend(epsilon, purpose=f"{purpose}:histogram")
    return {category: count for category, count in noisy_counts.items() if count >= minimum_noisy_count}


def dp_correlation_matrix(
    standardized_df: pd.DataFrame,
    epsilon: float,
    delta: float,
    accountant: PrivacyAccountant,
    rng: np.random.Generator,
    purpose: str,
    clip_norm: float = 3.0,
) -> np.ndarray:
    """DP correlation matrix via a noised second-moment matrix: clip each
    row to a bounded L2 norm (clip_norm), so no single row can contribute
    more than clip_norm^2 to the sum, add Gaussian noise calibrated to
    that bound, then project the noisy matrix onto the nearest valid
    correlation matrix (symmetric, unit diagonal, positive
    semi-definite) - free post-processing, since a noised matrix has no
    reason to already be a valid one.
    """
    if standardized_df.shape[1] == 0:
        return np.zeros((0, 0))

    values = standardized_df.to_numpy(dtype=float)
    row_norms = np.linalg.norm(values, axis=1, keepdims=True)
    scale_factors = np.minimum(1.0, clip_norm / np.where(row_norms == 0, 1.0, row_norms))
    clipped_values = values * scale_factors

    n = len(clipped_values)
    true_second_moment = (clipped_values.T @ clipped_values) / n

    # One row's contribution to the second-moment matrix (after dividing
    # by n) has Frobenius norm at most clip_norm^2 / n.
    sensitivity = (clip_norm**2) / n
    noisy_second_moment = gaussian_mechanism(
        true_second_moment, sensitivity=sensitivity, epsilon=epsilon, delta=delta, rng=rng
    )
    accountant.spend(epsilon, delta=delta, purpose=f"{purpose}:correlation")

    return _nearest_valid_correlation_matrix(noisy_second_moment)


def _nearest_valid_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues_psd = np.clip(eigenvalues, a_min=0, a_max=None)
    psd_matrix = eigenvectors @ np.diag(eigenvalues_psd) @ eigenvectors.T

    diagonal = np.diag(psd_matrix)
    safe_diagonal = np.where(diagonal <= 1e-12, 1.0, diagonal)
    scale = 1.0 / np.sqrt(safe_diagonal)
    correlation = psd_matrix * scale[:, None] * scale[None, :]
    np.fill_diagonal(correlation, 1.0)
    return correlation
