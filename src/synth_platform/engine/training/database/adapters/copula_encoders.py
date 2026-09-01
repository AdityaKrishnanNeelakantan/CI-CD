"""Per-semantic-type, JSON-serializable encode/decode logic for the
copula-based synthesizer (src/synthesis/adapters/safe_copula_adapter.py).

Deliberately hand-written and narrow (numerical/category/boolean/datetime
only - the 4 sdtypes this project's copula actually models; email/
identifier columns are never encoded here at all, see the adapter's own
docstring) rather than depending on SDV's rdt transformer library, whose
fitted state (numpy arrays, sklearn-like internal objects) has no
to_dict()/from_dict() anywhere and would require cloudpickle to persist -
see src/synthesis/adapters/sdv_adapter.py's docstring for why that risk
matters enough to hand-write a narrower replacement instead.

Every encoder's to_dict()/from_dict() round-trips through plain JSON only
(no numpy types, no fitted objects) so a saved model can be reloaded
without ever executing arbitrary code from the file, unlike pickle.

Categorical encoding uses frequency-based uniform encoding: each category
occupies a contiguous sub-interval of [0, 1] proportional to its (rare-
suppressed, see src/privacy/rare_category.py) frequency. This is the same
idea SDV's own rdt.UniformEncoder uses, hand-written here purely so its
fitted state is a plain list of boundaries rather than a fitted object.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synth_platform.engine.common.database.privacy.rare_category import RARE_CATEGORY_LABEL, replace_rare_categories

DEFAULT_CATEGORY_MINIMUM_SUPPORT = 20


class NumericEncoder:
    """Identity encoder for continuous numeric columns - the copula
    already fits its own univariate marginal per column internally, so no
    scaling is needed here beyond ensuring a clean float64 series.
    """

    def fit(self, series: pd.Series) -> None:
        pass  # nothing to learn - see class docstring

    def encode(self, series: pd.Series, rng: np.random.Generator) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").astype("float64")

    def decode(self, encoded: pd.Series) -> pd.Series:
        return encoded.astype("float64")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "numerical"}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> NumericEncoder:
        return cls()


class BooleanEncoder:
    def fit(self, series: pd.Series) -> None:
        pass

    def encode(self, series: pd.Series, rng: np.random.Generator) -> pd.Series:
        return series.astype(bool).astype("float64")

    def decode(self, encoded: pd.Series) -> pd.Series:
        return encoded >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {"type": "boolean"}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> BooleanEncoder:
        return cls()


class DatetimeEncoder:
    """Encodes as Unix seconds (float) - JSON-safe, and the copula treats
    it exactly like any other continuous numeric column.
    """

    def fit(self, series: pd.Series) -> None:
        pass

    def encode(self, series: pd.Series, rng: np.random.Generator) -> pd.Series:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return (parsed.astype("int64") / 1_000_000_000).astype("float64")

    def decode(self, encoded: pd.Series) -> pd.Series:
        return pd.to_datetime(encoded, unit="s", utc=True).dt.tz_localize(None)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "datetime"}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> DatetimeEncoder:
        return cls()


class CategoricalEncoder:
    def __init__(self, minimum_support: int = DEFAULT_CATEGORY_MINIMUM_SUPPORT) -> None:
        self._minimum_support = minimum_support
        self._categories: list[str] = []
        self._boundaries: list[float] = []  # len(categories) + 1, monotonically 0.0 -> 1.0

    def fit(self, series: pd.Series) -> None:
        suppressed = replace_rare_categories(series.astype(str), self._minimum_support)
        counts = suppressed.value_counts().to_dict()
        self._categories, self._boundaries = self._boundaries_from_counts(counts)

    @staticmethod
    def _boundaries_from_counts(counts: dict[str, float]) -> tuple[list[str], list[float]]:
        total = sum(counts.values()) or 1
        categories = [str(c) for c in counts]
        boundaries = [0.0]
        cumulative = 0.0
        for count in counts.values():
            cumulative += count / total
            boundaries.append(cumulative)
        boundaries[-1] = 1.0
        return categories, boundaries

    @classmethod
    def from_frequencies(cls, frequencies: dict[str, float], minimum_support: int = DEFAULT_CATEGORY_MINIMUM_SUPPORT) -> CategoricalEncoder:
        """Build an encoder directly from already-computed category
        counts/weights (e.g. a DP-noised histogram,
        src/privacy/dp_statistics.py's dp_category_histogram()) rather
        than fitting on a raw series - the counts are trusted as already
        final, no rare-category suppression is applied again here.
        """
        instance = cls(minimum_support=minimum_support)
        if not frequencies:
            # DP noise suppressed every category (all noisy counts fell
            # below minimum_noisy_count) - collapse to a single bucket so
            # encode()/decode() still have a valid [0.0, 1.0) boundary to
            # index into, instead of crashing on the next real value.
            instance._categories = [RARE_CATEGORY_LABEL]
            instance._boundaries = [0.0, 1.0]
            return instance
        instance._categories, instance._boundaries = cls._boundaries_from_counts(frequencies)
        return instance

    def encode(self, series: pd.Series, rng: np.random.Generator) -> pd.Series:
        suppressed = replace_rare_categories(series.astype(str), self._minimum_support)
        category_index = {c: i for i, c in enumerate(self._categories)}
        fallback_index = category_index.get(RARE_CATEGORY_LABEL, 0)

        encoded = []
        for value in suppressed:
            index = category_index.get(str(value), fallback_index)
            low, high = self._boundaries[index], self._boundaries[index + 1]
            encoded.append(low if high <= low else rng.uniform(low, high))
        return pd.Series(encoded, index=series.index, dtype="float64")

    def decode(self, encoded: pd.Series) -> pd.Series:
        clipped = encoded.clip(lower=0.0, upper=0.999999).to_numpy()
        indices = np.searchsorted(self._boundaries, clipped, side="right") - 1
        indices = np.clip(indices, 0, len(self._categories) - 1)
        return pd.Series([self._categories[i] for i in indices], index=encoded.index)

    def get_proportions(self) -> dict[str, float]:
        """The learned proportion of each category - a boundary's width
        *is* its category's proportion, by construction (see fit()/
        _boundaries_from_counts()). Introspection only, for a caller
        deciding what to override.
        """
        return {
            category: round(self._boundaries[i + 1] - self._boundaries[i], 6)
            for i, category in enumerate(self._categories)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "category",
            "categories": self._categories,
            "boundaries": self._boundaries,
            "minimum_support": self._minimum_support,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> CategoricalEncoder:
        instance = cls(minimum_support=state.get("minimum_support", DEFAULT_CATEGORY_MINIMUM_SUPPORT))
        instance._categories = state["categories"]
        instance._boundaries = state["boundaries"]
        return instance


#: Deliberately untyped as `type[Any]`: constructors genuinely diverge
#: (only CategoricalEncoder takes minimum_support), so every consumer of
#: this registry already treats the resolved encoder as duck-typed - a
#: Protocol narrow enough to satisfy every constructor would be too
#: narrow to cover fit()/encode()/decode() as actually called.
ENCODER_BY_TYPE: dict[str, type[Any]] = {
    "numerical": NumericEncoder,
    "boolean": BooleanEncoder,
    "datetime": DatetimeEncoder,
    "category": CategoricalEncoder,
}


def encoder_from_dict(state: dict[str, Any]) -> Any:
    encoder_cls = ENCODER_BY_TYPE[state["type"]]
    return encoder_cls.from_dict(state)
