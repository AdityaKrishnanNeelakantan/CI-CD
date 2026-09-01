"""Regenerates an already-sampled categorical column under a different
target distribution, without ever re-fitting the underlying model -
"tweak the ratio after generation, no retraining" for the copula-based
adapters (Safe/DP), which both encode categorical columns the same way
(src/synthesis/adapters/copula_encoders.py's CategoricalEncoder: a
frequency-proportional interval per category over [0, 1], with the
copula sampling a continuous value inside those intervals).

Rebalancing assigns exact integer target counts from the caller's
requested proportions across the already-sampled numeric ranks. A row
whose numeric value sat at the low end of the original distribution
still sits at the low end under the new one (category order by numeric
position is unchanged), so whatever correlation the copula learned
between this column and every other one is approximately preserved, not
exactly - overriding a marginal and expecting the original joint fidelity
to survive completely unchanged is not a sound expectation for any model,
this one included. The marginal counts themselves do match the requested
weights as closely as integer row counts allow.

An override must specify a weight for every category the model actually
learned - never a subset. Silently keeping "whatever the model already
had" for unmentioned categories would require deciding how to split the
leftover weight among them, which is exactly the kind of ambiguous,
silently-guessed behaviour this project avoids everywhere else.
"""

from __future__ import annotations

import pandas as pd

from synth_platform.engine.training.database.adapters.copula_encoders import CategoricalEncoder


class CategoryOverrideError(Exception):
    """Raised when category_overrides doesn't exactly match the model's known categories."""


def normalize_category_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise CategoryOverrideError(f"category override weights must sum to a positive number, got {weights}")
    return {category: weight / total for category, weight in weights.items()}


def build_rebalanced_encoder(original_encoder: CategoricalEncoder, overrides: dict[str, float]) -> CategoricalEncoder:
    known_categories = set(original_encoder.get_proportions())
    missing = known_categories - set(overrides)
    if missing:
        raise CategoryOverrideError(
            f"category_overrides must specify a weight for every category the model learned - "
            f"missing {sorted(missing)}. Known categories: {sorted(known_categories)}."
        )
    unknown = set(overrides) - known_categories
    if unknown:
        raise CategoryOverrideError(
            f"category_overrides mentions categories the model never learned: {sorted(unknown)}. "
            f"Known categories: {sorted(known_categories)}."
        )
    normalized = normalize_category_weights(overrides)
    return CategoricalEncoder.from_frequencies(normalized, minimum_support=0)


def _target_counts(total_rows: int, weights: dict[str, float], category_order: list[str]) -> dict[str, int]:
    normalized = normalize_category_weights(weights)
    raw_counts = {category: total_rows * normalized[category] for category in category_order}
    counts = {category: int(raw_counts[category]) for category in category_order}
    remainder = total_rows - sum(counts.values())

    if remainder > 0:
        ranked = sorted(category_order, key=lambda category: (-(raw_counts[category] - counts[category]), category))
        for category in ranked[:remainder]:
            counts[category] += 1

    return counts


def rebalance_column(
    sampled_encoded: pd.Series, original_encoder: CategoricalEncoder, overrides: dict[str, float]
) -> pd.Series:
    build_rebalanced_encoder(original_encoder, overrides)
    category_order = list(original_encoder.get_proportions())
    counts = _target_counts(len(sampled_encoded), overrides, category_order)

    ordered_index = sampled_encoded.astype("float64").clip(0.0, 1.0).sort_values(kind="mergesort").index
    decoded = pd.Series(index=sampled_encoded.index, dtype="object")
    start = 0
    for category in category_order:
        end = start + counts[category]
        if end > start:
            decoded.loc[ordered_index[start:end]] = category
        start = end
    return decoded
