"""Differential-privacy noise mechanisms: the actual mathematics, not a
configuration placeholder. This project's earlier privacy review was
explicit about the difference - "a JSON file containing epsilon does not
make the training private" - so every function here performs a real,
calibrated noise addition with a documented sensitivity assumption,
never just records an epsilon value and calls it done.

Two mechanisms, matching standard usage (Dwork & Roth, "The Algorithmic
Foundations of Differential Privacy"):
- Laplace mechanism: pure epsilon-DP (delta=0), valid for any epsilon > 0.
  Used here for scalar queries with bounded L1 sensitivity - counts,
  clipped means, clipped variances.
- Gaussian mechanism: (epsilon, delta)-DP, calibrated via the classical
  Dwork-Roth bound (Appendix A, Theorem A.1 of the above), which is only
  a valid proof for epsilon in (0, 1) - this module enforces that range
  rather than silently using an out-of-range formula. Used here for the
  vector/matrix-valued correlation-matrix query, where L2 sensitivity and
  Gaussian noise compose more efficiently than Laplace would.

Every mechanism takes an explicit numpy Generator so callers (and tests)
control reproducibility exactly as the rest of this project's generation
code already does.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class InvalidPrivacyParameterError(Exception):
    """Raised when an epsilon/delta/sensitivity value is outside the
    range a mechanism's privacy proof actually covers.
    """


def clip_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Bound every value to [lower, upper] - the sensitivity-control step
    every mechanism below depends on. Without this, a single unbounded
    outlier could make a query's true sensitivity unbounded too, breaking
    every noise calibration that assumes a fixed sensitivity.
    """
    if lower > upper:
        raise InvalidPrivacyParameterError(f"lower bound {lower} must not exceed upper bound {upper}")
    return series.clip(lower=lower, upper=upper)


def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float, rng: np.random.Generator) -> float:
    """Pure epsilon-DP: add Laplace(0, sensitivity/epsilon) noise.

    Valid for any epsilon > 0 - there is no upper bound on epsilon for
    this mechanism's proof, unlike the Gaussian mechanism below.
    """
    if epsilon <= 0:
        raise InvalidPrivacyParameterError(f"epsilon must be > 0, got {epsilon}")
    if sensitivity < 0:
        raise InvalidPrivacyParameterError(f"sensitivity must be >= 0, got {sensitivity}")
    scale = sensitivity / epsilon
    return float(true_value + rng.laplace(loc=0.0, scale=scale))


def gaussian_mechanism_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """The classical (epsilon, delta)-DP Gaussian mechanism noise scale
    (Dwork & Roth, Theorem A.1): sigma = sensitivity * sqrt(2 ln(1.25/delta)) / epsilon.

    That theorem's proof requires epsilon in (0, 1) - this is enforced
    here rather than silently computing a number outside the range the
    proof actually covers. Use the Laplace mechanism instead for
    epsilon >= 1 scalar queries; for the matrix-valued query this module
    exists for, restructure into a smaller per-query epsilon instead.
    """
    if not (0 < epsilon < 1):
        raise InvalidPrivacyParameterError(
            f"the classical Gaussian mechanism bound requires 0 < epsilon < 1, got {epsilon}"
        )
    if not (0 < delta < 1):
        raise InvalidPrivacyParameterError(f"delta must be in (0, 1), got {delta}")
    if sensitivity < 0:
        raise InvalidPrivacyParameterError(f"sensitivity must be >= 0, got {sensitivity}")
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def gaussian_mechanism(
    true_value: np.ndarray, sensitivity: float, epsilon: float, delta: float, rng: np.random.Generator
) -> np.ndarray:
    """Add Gaussian(0, sigma^2) noise, elementwise, to an array-valued query."""
    sigma = gaussian_mechanism_sigma(sensitivity, epsilon, delta)
    noise = rng.normal(loc=0.0, scale=sigma, size=np.shape(true_value))
    return np.asarray(true_value) + noise


def dp_count(true_count: int, epsilon: float, rng: np.random.Generator) -> float:
    """A counting query (how many rows fall in some category/bucket) has
    L1 sensitivity exactly 1: adding or removing one row changes the
    count by at most 1. The result may be negative or non-integer after
    noise - callers must clip/round for their own use (e.g. max(0, ...)
    before using it as a category weight), since "the noisy count itself"
    is the DP-released value, not a post-processed one.
    """
    return laplace_mechanism(float(true_count), sensitivity=1.0, epsilon=epsilon, rng=rng)
