"""Pickle-free GaussianCopula-based SynthesizerAdapter.

Uses copulas.multivariate.GaussianMultivariate directly - the same
statistical core SDV's GaussianCopulaSynthesizer wraps (see
src/synthesis/adapters/sdv_adapter.py's docstring) - but with a
hand-written encode/decode layer (src/synthesis/adapters/copula_encoders.py)
instead of SDV's rdt transformers, because rdt's fitted transformer state
has no JSON-serializable form and would require cloudpickle. Every
save()/load() round-trip here is plain JSON: no pickle, no cloudpickle,
anywhere in this adapter or anything it writes to disk - the file can be
inspected or reloaded without ever executing code from it.

PII columns (email, identifier) are never fitted into the copula at all,
matching sdv_adapter.py's pii=True behaviour: excluded from the joint
distribution entirely and generated independently via Faker or a
deterministic sequence at sample time, so a real value in one of them has
zero statistical influence on the synthetic output.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import BetaUnivariate

from synth_platform.engine.training.database.adapters.category_rebalancing import rebalance_column
from synth_platform.engine.training.database.adapters.context_generators import (
    generate_pii_columns,
    learn_identifier_format,
)
from synth_platform.engine.training.database.adapters.copula_encoders import (
    DEFAULT_CATEGORY_MINIMUM_SUPPORT,
    ENCODER_BY_TYPE,
    CategoricalEncoder,
    encoder_from_dict,
)
from synth_platform.engine.training.database.base import SynthesisError, SynthesizerAdapter
from synth_platform.engine.common.database.privacy.context_fields import (
    CONTEXT_AWARE_KINDS,
    ContextFieldKind,
    resolve_context_kind,
)

APPROVED_STATUS = "approved"
UNSYNTHESIZABLE_SEMANTIC_TYPES = frozenset({"free_text"})
PII_SEMANTIC_TYPES = frozenset({"email", "identifier", "person_name", "phone_number"})
# Statistical types that may be mis-labelled PII (name cues can override).
_STATISTICAL_SEMANTIC_TYPES = frozenset(ENCODER_BY_TYPE)


def _is_context_or_pii_column(column_name: str, semantic_type: str) -> bool:
    """Never statistically model context/PII — only DNA columns enter the copula.

    Unknown/unmapped semantic types are *not* promoted to free-text generation
    here: they must be excluded so fit fails closed when nothing is generatable.
    Column-name context cues only override known statistical labels
    (e.g. an ``email`` column mis-approved as ``category``).
    """
    if semantic_type in PII_SEMANTIC_TYPES or semantic_type in UNSYNTHESIZABLE_SEMANTIC_TYPES:
        return True
    if semantic_type not in _STATISTICAL_SEMANTIC_TYPES:
        return False
    return resolve_context_kind(column_name, semantic_type) in CONTEXT_AWARE_KINDS


class SafeCopulaSynthesizerAdapter(SynthesizerAdapter):
    model_type = "safe_gaussian_copula"
    file_extension = ".json"
    serialization_format = "json"

    def __init__(self, category_minimum_support: int = DEFAULT_CATEGORY_MINIMUM_SUPPORT) -> None:
        self._category_minimum_support = category_minimum_support
        self._copula: GaussianMultivariate | None = None
        self._encoders: dict[str, Any] = {}
        self._null_rates: dict[str, float] = {}
        self._trained_columns: list[str] = []
        self._pii_columns: dict[str, dict[str, Any]] = {}
        self._primary_key: str | None = None
        self._row_count: int = 0
        self._seed: int | None = None

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
                # modelled, generated independently at sample time (Faker/
                # rules) so a real value has zero influence on the synthetic
                # output. free_text used to be dropped from the output
                # entirely instead - which silently broke any NOT NULL
                # free_text column's target-database write (a confirmed
                # real bug: dataset_metadata.value, IntegrityError, found
                # running this pipeline against a real medical fixture).
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
            if semantic_type not in ENCODER_BY_TYPE:
                excluded.append({"column": column_name, "reason": f"unmapped_semantic_type={semantic_type}"})
                continue
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

        encoders: dict[str, Any] = {}
        null_rates: dict[str, float] = {}
        encoded_columns: dict[str, pd.Series] = {}

        for column_name, semantic_type in trainable.items():
            series = df[column_name]
            non_null = series.dropna()
            null_rates[column_name] = float(series.isna().mean())

            encoder = (
                ENCODER_BY_TYPE[semantic_type](minimum_support=self._category_minimum_support)
                if semantic_type == "category"
                else ENCODER_BY_TYPE[semantic_type]()
            )
            encoder.fit(non_null)
            encoded_non_null = encoder.encode(non_null, rng)

            fill_value = float(encoded_non_null.mean()) if len(encoded_non_null) else 0.0
            full_encoded = pd.Series(fill_value, index=series.index, dtype="float64")
            full_encoded.loc[non_null.index] = encoded_non_null

            encoders[column_name] = encoder
            encoded_columns[column_name] = full_encoded

        copula = None
        if encoded_columns:
            training_frame = pd.DataFrame(encoded_columns)
            copula = GaussianMultivariate(distribution=BetaUnivariate)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="scipy")
                copula.fit(training_frame)

        self._copula = copula
        self._encoders = encoders
        self._null_rates = null_rates
        self._trained_columns = list(trainable.keys()) + list(pii_columns.keys())
        self._pii_columns = pii_columns
        self._primary_key = single_pk if single_pk in self._trained_columns else None
        self._row_count = len(df)
        self._seed = seed

        return {
            "model_type": self.model_type,
            "trained_columns": self._trained_columns,
            "excluded_columns": excluded,
            "row_count": self._row_count,
            "primary_key": self._primary_key,
            "alternate_keys": [],
        }

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
            # GaussianMultivariate.sample() draws from numpy's global RNG
            # state, seeded via its own set_random_state() - passing a
            # seed to np.random.default_rng() above has no effect on it
            # at all (verified empirically: two sample() calls with the
            # "same" seed produced different rows until this was added).
            self._copula.set_random_state(effective_seed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="scipy")
                sampled_encoded = self._copula.sample(num_rows)
            for column_name, encoder in self._encoders.items():
                null_mask = rng.random(num_rows) < self._null_rates[column_name]
                if column_name in category_overrides and isinstance(encoder, CategoricalEncoder):
                    # Re-decode the SAME sampled numeric ranks under exact
                    # requested marginal counts - no re-fitting, see
                    # category_rebalancing.py for the preservation tradeoff.
                    decoded = pd.Series(index=sampled_encoded.index, dtype=object)
                    non_null_index = sampled_encoded.index[~null_mask]
                    decoded.loc[non_null_index] = rebalance_column(
                        sampled_encoded.loc[non_null_index, column_name],
                        encoder,
                        category_overrides[column_name],
                    ).astype(object)
                else:
                    decoded = encoder.decode(sampled_encoded[column_name]).astype(object)
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
        """The currently learned proportion of each category for a
        trained categorical column - what a caller needs to see before
        deciding what to pass as category_overrides to sample().
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
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> SafeCopulaSynthesizerAdapter:
        path = Path(path)
        if not path.is_file():
            raise SynthesisError(f"model file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        instance = cls()
        instance._trained_columns = state["trained_columns"]
        instance._pii_columns = state["pii_columns"]
        instance._primary_key = state["primary_key"]
        instance._row_count = state["row_count"]
        instance._seed = state["seed"]
        instance._null_rates = state["null_rates"]
        instance._encoders = {name: encoder_from_dict(entry) for name, entry in state["encoders"].items()}
        instance._copula = GaussianMultivariate.from_dict(state["copula"]) if state["copula"] is not None else None

        return instance
