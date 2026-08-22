"""Rare-category suppression: the same minimum-support/sentinel-bucket
protection MOSTLY AI documents for its own tabular synthesizer. A category
seen only a handful of times can itself identify the one or two source
records that had it, so it must never survive into anything that leaves
the protected training zone - an exported reference profile, or a fitted
encoder's category vocabulary (src/synthesis/adapters/copula_encoders.py).

A fixed minimum_support is a policy choice, not a mathematically proven
anonymity guarantee - it reduces an obvious, cheap re-identification
vector, nothing more.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

RARE_CATEGORY_LABEL = "__RARE__"
DEFAULT_MINIMUM_SUPPORT = 20


def suppress_rare_categories(
    frequencies: dict[str, int], minimum_support: int = DEFAULT_MINIMUM_SUPPORT
) -> dict[str, int]:
    """Merge every category with count < minimum_support into one bucket.

    A category that already meets the threshold keeps its own label and
    count unchanged - there is no partial anonymisation of a safe category.
    """
    safe: dict[str, int] = {}
    rare_total = 0
    for category, count in frequencies.items():
        if count < minimum_support:
            rare_total += count
        else:
            safe[category] = count
    if rare_total > 0:
        safe[RARE_CATEGORY_LABEL] = safe.get(RARE_CATEGORY_LABEL, 0) + rare_total
    return safe


def replace_rare_categories(series: pd.Series, minimum_support: int = DEFAULT_MINIMUM_SUPPORT) -> pd.Series:
    """Row-level counterpart of suppress_rare_categories(): every value
    belonging to a rare category is replaced with RARE_CATEGORY_LABEL, so
    a model fitted on the result can never memorise or reproduce a rare
    value verbatim.
    """
    counts = series.value_counts()
    rare_values: set[Any] = set(counts[counts < minimum_support].index)
    if not rare_values:
        return series.copy()
    return series.apply(lambda v: RARE_CATEGORY_LABEL if v in rare_values else v)
