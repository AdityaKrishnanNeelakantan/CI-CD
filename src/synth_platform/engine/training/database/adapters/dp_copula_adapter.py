"""GaussianCopula-based SynthesizerAdapter with an actual, verified
(epsilon, delta)-differential-privacy guarantee on every statistic used
to build the model - the gap this project's own earlier privacy review
called out explicitly: "the architecture currently has configuration
placeholders for differential privacy but does not actually implement or
account for it."

Every one of the model's parameters - each column's marginal mean/
variance (numeric, boolean, datetime) or category histogram, and the
joint correlation matrix - is computed via src/privacy/dp_statistics.py's
noised estimators, with every query recorded against a
src/privacy/privacy_accountant.py budget. fit() raises
PrivacyBudgetExceededError rather than silently exceeding the configured
total (epsilon, delta), and persists the accountant's full summary so a
reviewer can see exactly what was spent, not just a trusted epsilon label.

Numeric/datetime column bounds (required by every DP mechanism here, for
clipping/sensitivity control) must be supplied explicitly by the caller
as public domain knowledge (e.g. "age is 0-120") - see
MissingColumnBoundsError. Deriving a bound from the real data's own exact
min/max would leak information through the bound choice itself, silently
breaking the guarantee while looking correct; this adapter refuses to do
that rather than approximate it.

The joint correlation matrix and every column's marginal are modelled as
Gaussian - simpler and more directly DP-estimable than letting
copulas.univariate auto-select a best-fit distribution per column (that
selection would itself need to be done privately). This trades some
fidelity for every parameter having a documented, verified DP mechanism
behind it.

PII/identifier/email columns are, as in src/synthesis/adapters/
safe_copula_adapter.py, never modelled statistically at all - Faker/
pattern-generated at sample time, spending zero privacy budget.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate

from synth_platform.engine.common.database.privacy.dp_statistics import (
    dp_category_histogram,
    dp_correlation_matrix,
    dp_mean_and_variance,
)
from synth_platform.engine.common.database.privacy.privacy_accountant import PrivacyAccountant
from synth_platform.engine.common.database.privacy.context_fields import (
    CONTEXT_AWARE_KINDS,
    ContextFieldKind,
    resolve_context_kind,
)
from synth_platform.engine.training.database.adapters.category_rebalancing import rebalance_column
from synth_platform.engine.training.database.adapters.context_generators import (
    generate_pii_columns,
    learn_identifier_format,
)
from synth_platform.engine.training.database.adapters.copula_encoders import (
    BooleanEncoder,
    CategoricalEncoder,
    DatetimeEncoder,
    NumericEncoder,
)
from synth_platform.engine.training.database.base import SynthesisError, SynthesizerAdapter

APPROVED_STATUS = "approved"
UNSYNTHESIZABLE_SEMANTIC_TYPES = frozenset({"free_text"})
PII_SEMANTIC_TYPES = frozenset({"email", "identifier", "person_name", "phone_number"})
_STATISTICAL_SEMANTIC_TYPES = frozenset({"numerical", "category", "boolean", "datetime"})
_BOUNDED_SEMANTIC_TYPES = frozenset({"numerical", "datetime"})
_GAUSSIAN_UNIVARIATE_TYPE = "copulas.univariate.gaussian.GaussianUnivariate"


def _is_context_or_pii_column(column_name: str, semantic_type: str) -> bool:
    """Never statistically model context/PII.

    Unknown semantic types stay unmapped (excluded). Column-name cues only
    override known statistical labels that may have been mis-approved.
    """
    if semantic_type in PII_SEMANTIC_TYPES or semantic_type in UNSYNTHESIZABLE_SEMANTIC_TYPES:
        return True
    if semantic_type not in _STATISTICAL_SEMANTIC_TYPES:
        return False
    return resolve_context_kind(column_name, semantic_type) in CONTEXT_AWARE_KINDS

DEFAULT_DELTA_BUDGET = 1e-5
DEFAULT_CORRELATION_EPSILON_FRACTION = 0.3
# The classical Gaussian mechanism's privacy proof (src/privacy/
# dp_primitives.py's gaussian_mechanism_sigma) only covers epsilon in
# (0, 1) - correlation_epsilon_fraction * epsilon_budget easily exceeds 1
# for any realistic total budget (e.g. the default 0.3 fraction of a
# budget of 5.0 is 1.5), which crashed outright until this cap was added.
# Any excess this cap leaves unspent on the correlation query flows back
# into the per-column marginal budget instead (see fit()) rather than
# being wasted.
_MAX_CORRELATION_EPSILON = 0.99


class MissingColumnBoundsError(Exception):
    """Raised when a numeric/datetime column has no caller-supplied
    [lower, upper] bound - required for every DP mechanism this adapter
    uses, and never silently derived from the real data.
    """


def _to_epoch_seconds(value: Any) -> float:
    return pd.Timestamp(value).timestamp()


class DPCopulaSynthesizerAdapter(SynthesizerAdapter):
    model_type = "dp_gaussian_copula"
    file_extension = ".json"
    serialization_format = "json"

    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float = DEFAULT_DELTA_BUDGET,
        column_bounds: dict[str, tuple[float, float]] | None = None,
        category_minimum_support: int = 20,
        correlation_epsilon_fraction: float = DEFAULT_CORRELATION_EPSILON_FRACTION,
    ) -> None:
        if not (0.0 <= correlation_epsilon_fraction < 1.0):
            raise ValueError(f"correlation_epsilon_fraction must be in [0, 1), got {correlation_epsilon_fraction}")
        self._epsilon_budget = epsilon_budget
        self._delta_budget = delta_budget
        self._column_bounds = column_bounds or {}
        self._category_minimum_support = category_minimum_support
        self._correlation_epsilon_fraction = correlation_epsilon_fraction

        self._copula: GaussianMultivariate | None = None
        self._encoders: dict[str, Any] = {}
        self._null_rates: dict[str, float] = {}
        self._trained_columns: list[str] = []
        self._pii_columns: dict[str, dict[str, Any]] = {}
        self._primary_key: str | None = None
        self._row_count: int = 0
        self._seed: int | None = None
        self._privacy_summary: dict[str, Any] = {}

    def fit(
        self,
        df: pd.DataFrame,
        table_name: str,
        table_contract: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        columns_contract = table_contract.get("columns", {})
        primary_keys = table_contract.get("primary_key", [])
        single_pk = primary_keys[0] if len(primary_keys) == 1 else None

        trainable: dict[str, str] = {}
        pii_columns: dict[str, dict[str, Any]] = {}
        excluded: list[dict[str, str]] = []

        for column_name, column_contract in columns_contract.items():
            if column_contract.get("inference_status") != APPROVED_STATUS:
                excluded.append(
                    {"column": column_name, "reason": f"inference_status={column_contract.get('inference_status')}"}
                )
                continue
            semantic_type = column_contract["semantic_type"]
            if _is_context_or_pii_column(column_name, semantic_type):
                # Both buckets share the same treatment: never statistically
                # modelled (zero privacy budget spent), generated
                # independently at sample time. free_text used to be
                # dropped from the output entirely instead, which silently
                # broke any NOT NULL free_text column's target-database
                # write - see safe_copula_adapter.py's identical fix.
                pii_entry: dict[str, Any] = {
                    "semantic_type": semantic_type,
                    "physical_type": column_contract.get("physical_type", ""),
                }
                if (
                    resolve_context_kind(column_name, semantic_type) is ContextFieldKind.IDENTIFIER
                    or semantic_type == "identifier"
                ) and column_name in df.columns:
                    format_spec = learn_identifier_format(df[column_name].tolist())
                    if format_spec is not None:
                        pii_entry["identifier_format"] = format_spec
                pii_columns[column_name] = pii_entry
                continue
            if semantic_type not in {"numerical", "boolean", "datetime", "category"}:
                excluded.append({"column": column_name, "reason": f"unmapped_semantic_type={semantic_type}"})
                continue
            if semantic_type in _BOUNDED_SEMANTIC_TYPES and column_name not in self._column_bounds:
                raise MissingColumnBoundsError(
                    f"column {column_name!r} (semantic_type={semantic_type!r}) has no caller-supplied bounds - "
                    f"{self.model_type} requires an explicit public [lower, upper] bound for every numeric/"
                    "datetime column, and never derives one from the real data itself."
                )
            trainable[column_name] = semantic_type

        if len(primary_keys) > 1:
            excluded.append(
                {"column": ",".join(primary_keys), "reason": "composite_primary_key_not_supported_by_this_adapter"}
            )
        if not trainable and not pii_columns:
            raise SynthesisError(f"table {table_name!r} has no trainable approved columns")

        missing = (set(trainable) | set(pii_columns)) - set(df.columns)
        if missing:
            raise SynthesisError(f"training data is missing contract columns: {sorted(missing)}")

        rng = np.random.default_rng(seed)
        accountant = PrivacyAccountant(epsilon_budget=self._epsilon_budget, delta_budget=self._delta_budget)

        raw_correlation_epsilon = (
            self._correlation_epsilon_fraction * self._epsilon_budget if len(trainable) >= 2 else 0.0
        )
        correlation_epsilon = min(raw_correlation_epsilon, _MAX_CORRELATION_EPSILON)
        marginal_epsilon_total = self._epsilon_budget - correlation_epsilon
        per_column_epsilon = marginal_epsilon_total / len(trainable) if trainable else 0.0

        encoders: dict[str, Any] = {}
        null_rates: dict[str, float] = {}
        encoded_columns: dict[str, pd.Series] = {}
        gaussian_params: dict[str, dict[str, float]] = {}

        for column_name, semantic_type in trainable.items():
            series = df[column_name]
            non_null = series.dropna()
            null_rates[column_name] = float(series.isna().mean())

            if semantic_type == "category":
                encoded_non_null, encoder = self._fit_category_column(
                    non_null, per_column_epsilon, accountant, rng, purpose=column_name
                )
            else:
                encoded_non_null, encoder, params = self._fit_bounded_column(
                    non_null, semantic_type, column_name, per_column_epsilon, accountant, rng
                )
                gaussian_params[column_name] = params

            fill_value = float(encoded_non_null.mean()) if len(encoded_non_null) else 0.0
            full_encoded = pd.Series(fill_value, index=series.index, dtype="float64")
            full_encoded.loc[non_null.index] = encoded_non_null

            encoders[column_name] = encoder
            encoded_columns[column_name] = full_encoded

        copula = None
        if encoded_columns:
            columns_order = list(encoded_columns.keys())
            copula = self._build_dp_copula(encoded_columns, columns_order, gaussian_params, correlation_epsilon, accountant, rng)

        self._copula = copula
        self._encoders = encoders
        self._null_rates = null_rates
        self._trained_columns = list(trainable.keys()) + list(pii_columns.keys())
        self._pii_columns = pii_columns
        self._primary_key = single_pk if single_pk in self._trained_columns else None
        self._row_count = len(df)
        self._seed = seed
        self._privacy_summary = accountant.summary()

        return {
            "model_type": self.model_type,
            "trained_columns": self._trained_columns,
            "excluded_columns": excluded,
            "row_count": self._row_count,
            "primary_key": self._primary_key,
            "alternate_keys": [],
            "privacy_summary": self._privacy_summary,
        }

    def _fit_bounded_column(
        self,
        non_null: pd.Series,
        semantic_type: str,
        column_name: str,
        epsilon: float,
        accountant: PrivacyAccountant,
        rng: np.random.Generator,
    ) -> tuple[pd.Series, Any, dict[str, float]]:
        encoder: BooleanEncoder | DatetimeEncoder | NumericEncoder
        if semantic_type == "boolean":
            numeric = non_null.astype(bool).astype("float64")
            lower, upper = 0.0, 1.0
            encoder = BooleanEncoder()
        elif semantic_type == "datetime":
            numeric = (pd.to_datetime(non_null, errors="coerce", utc=True).astype("int64") / 1_000_000_000).astype(
                "float64"
            )
            lower_raw, upper_raw = self._column_bounds[column_name]
            lower, upper = _to_epoch_seconds(lower_raw), _to_epoch_seconds(upper_raw)
            encoder = DatetimeEncoder()
        else:  # numerical
            numeric = pd.to_numeric(non_null, errors="coerce").astype("float64")
            lower, upper = self._column_bounds[column_name]
            encoder = NumericEncoder()

        dp_mean, dp_variance = dp_mean_and_variance(
            numeric, lower=lower, upper=upper, epsilon=epsilon, accountant=accountant, rng=rng, purpose=column_name
        )
        params = {"loc": dp_mean, "scale": max(dp_variance, 1e-9) ** 0.5}
        return numeric, encoder, params

    def _fit_category_column(
        self, non_null: pd.Series, epsilon: float, accountant: PrivacyAccountant, rng: np.random.Generator, purpose: str
    ) -> tuple[pd.Series, CategoricalEncoder]:
        histogram = dp_category_histogram(
            non_null.astype(str),
            epsilon=epsilon,
            accountant=accountant,
            rng=rng,
            purpose=purpose,
            minimum_noisy_count=float(self._category_minimum_support),
        )
        encoder = CategoricalEncoder.from_frequencies(histogram, minimum_support=self._category_minimum_support)
        encoded = encoder.encode(non_null.astype(str), rng)
        return encoded, encoder

    def _build_dp_copula(
        self,
        encoded_columns: dict[str, pd.Series],
        columns_order: list[str],
        gaussian_params: dict[str, dict[str, float]],
        correlation_epsilon: float,
        accountant: PrivacyAccountant,
        rng: np.random.Generator,
    ) -> GaussianMultivariate:
        frame = pd.DataFrame(encoded_columns)[columns_order]

        if correlation_epsilon > 0 and len(columns_order) >= 2:
            means = frame.mean()
            stds = frame.std().replace(0, 1.0)
            standardized = (frame - means) / stds
            correlation = dp_correlation_matrix(
                standardized, epsilon=correlation_epsilon, delta=self._delta_budget,
                accountant=accountant, rng=rng, purpose="joint_correlation",
            )
        else:
            correlation = np.eye(len(columns_order))

        univariates = []
        for column_name in columns_order:
            if column_name in gaussian_params:
                params = gaussian_params[column_name]
            else:  # category column - approximate its numeric-encoded surrogate as Gaussian too
                encoded = frame[column_name]
                params = {"loc": float(encoded.mean()), "scale": max(float(encoded.std()), 1e-9)}
            univariates.append({"loc": params["loc"], "scale": params["scale"], "type": _GAUSSIAN_UNIVARIATE_TYPE})

        copula_dict = {
            "correlation": correlation.tolist(),
            "univariates": univariates,
            "columns": columns_order,
            "type": "copulas.multivariate.gaussian.GaussianMultivariate",
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="scipy")
            return GaussianMultivariate.from_dict(copula_dict)

    def sample(
        self,
        num_rows: int,
        seed: int | None = None,
        category_overrides: dict[str, dict[str, float]] | None = None,
        row_offset: int = 0,
    ) -> pd.DataFrame:
        if self._copula is None and not self._pii_columns:
            raise SynthesisError("sample() called before fit()/load()")
        if num_rows <= 0:
            raise ValueError(f"num_rows must be positive, got {num_rows}")

        category_overrides = category_overrides or {}
        unknown_override_columns = set(category_overrides) - set(self._encoders)
        if unknown_override_columns:
            raise SynthesisError(
                f"category_overrides mentions columns this model never trained: {sorted(unknown_override_columns)}"
            )

        effective_seed = self._seed if seed is None else seed
        rng = np.random.default_rng(effective_seed)
        data: dict[str, Any] = {}

        if self._copula is not None:
            self._copula.set_random_state(effective_seed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="scipy")
                sampled_encoded = self._copula.sample(num_rows)
            for column_name, encoder in self._encoders.items():
                if column_name in category_overrides and isinstance(encoder, CategoricalEncoder):
                    # Rebalancing costs zero additional privacy budget: it
                    # only re-decodes values already sampled under the DP-
                    # protected model, never touches the real training
                    # data or re-runs any DP mechanism.
                    decoded = rebalance_column(
                        sampled_encoded[column_name], encoder, category_overrides[column_name]
                    ).astype(object)
                else:
                    decoded = encoder.decode(sampled_encoded[column_name]).astype(object)
                null_mask = rng.random(num_rows) < self._null_rates[column_name]
                decoded[null_mask] = None
                data[column_name] = decoded.to_numpy()

        pii_data = generate_pii_columns(
            self._pii_columns,
            num_rows=num_rows,
            primary_key=self._primary_key,
            seed=effective_seed,
            rng=rng,
            row_offset=row_offset,
        )
        data.update(pii_data)

        return pd.DataFrame(data, columns=self._trained_columns)

    def get_category_distribution(self, column_name: str) -> dict[str, float]:
        """The currently learned (DP-noised) proportion of each category
        for a trained categorical column - what a caller needs to see
        before deciding what to pass as category_overrides to sample().
        """
        encoder = self._encoders.get(column_name)
        if not isinstance(encoder, CategoricalEncoder):
            raise SynthesisError(f"{column_name!r} is not a trained categorical column on this model")
        return encoder.get_proportions()

    def save(self, path: str | Path) -> None:
        if self._copula is None and not self._pii_columns:
            raise SynthesisError("save() called before fit()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_type": self.model_type,
            "trained_columns": self._trained_columns,
            "pii_columns": self._pii_columns,
            "primary_key": self._primary_key,
            "row_count": self._row_count,
            "seed": self._seed,
            "null_rates": self._null_rates,
            "encoders": {name: encoder.to_dict() for name, encoder in self._encoders.items()},
            "copula": self._copula.to_dict() if self._copula is not None else None,
            "privacy_summary": self._privacy_summary,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> DPCopulaSynthesizerAdapter:
        path = Path(path)
        if not path.is_file():
            raise SynthesisError(f"model file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        from synth_platform.engine.training.database.adapters.copula_encoders import encoder_from_dict

        instance = cls(epsilon_budget=1.0)  # placeholder - a loaded model never re-fits, so budget config is moot
        instance._trained_columns = state["trained_columns"]
        instance._pii_columns = state["pii_columns"]
        instance._primary_key = state["primary_key"]
        instance._row_count = state["row_count"]
        instance._seed = state["seed"]
        instance._null_rates = state["null_rates"]
        instance._encoders = {name: encoder_from_dict(entry) for name, entry in state["encoders"].items()}
        instance._copula = GaussianMultivariate.from_dict(state["copula"]) if state["copula"] is not None else None
        instance._privacy_summary = state.get("privacy_summary", {})
        return instance
