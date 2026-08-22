"""Profiling service for connectors and materialized relational datasets."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from synth_platform.domain.profiling.models import (
    CategoricalStatistics, ColumnProfile, DataProfile, NumericStatistics,
    SamplingRecord, SensitivityClass, TableProfile,
)
from synth_platform.domain.relational.models import CardinalityModel
from synth_platform.domain.schema.models import DatabaseSchema
from synth_platform.domain.schema.source import SourceConnector
from synth_platform.domain.semantics.dependencies import DependencyGraph
from synth_platform.engine.inference.platform.service import compile_conditional, infer_dependency_graph

_GRID = [i / 20 for i in range(21)]
_PII_HINTS = {
    "email": "email", "phone": "phone", "ssn": "ssn", "social_security": "ssn",
    "card_number": "card_number", "address": "address", "first_name": "person_name",
    "last_name": "person_name", "full_name": "person_name",
}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MODELLABLE = ("numeric", "datetime", "categorical")


def _looks_email(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(50)
    return bool(len(sample)) and sum(bool(_EMAIL_RE.match(v)) for v in sample) / len(sample) > 0.8


def _classify(column: str, s: pd.Series, logical_type: str | None = None):
    logical = (logical_type or "").lower()
    if logical in {"date", "datetime", "timestamp"}:
        return logical, "datetime", SensitivityClass.PUBLIC
    if logical == "boolean":
        return "boolean", "categorical", SensitivityClass.PUBLIC
    if logical == "categorical":
        return "category", "categorical", SensitivityClass.PUBLIC
    if logical in {"integer", "number"}:
        return logical, "numeric", SensitivityClass.PUBLIC
    if logical in {"free_text", "text"}:
        return "text", "text", SensitivityClass.FREE_TEXT
    if logical in {"identifier", "pii_email", "pii_phone", "pii_name", "pii_ssn"}:
        sens = (SensitivityClass.DIRECT_IDENTIFIER if logical.startswith("pii_")
                else SensitivityClass.QUASI_IDENTIFIER)
        return logical.removeprefix("pii_"), "identifier", sens
    low = column.lower()
    for hint, sem in _PII_HINTS.items():
        if hint in low:
            return sem, "pii", SensitivityClass.DIRECT_IDENTIFIER
    if _looks_email(s):
        return "email", "pii", SensitivityClass.DIRECT_IDENTIFIER
    nn = s.dropna()
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", "datetime", SensitivityClass.PUBLIC
    if pd.api.types.is_numeric_dtype(s):
        if len(nn) > 0 and nn.nunique() == len(nn) and "id" in low:
            return "identifier", "identifier", SensitivityClass.QUASI_IDENTIFIER
        return "numeric", "numeric", SensitivityClass.PUBLIC
    card = nn.nunique()
    if 0 < card <= max(2, int(0.5 * max(1, len(nn)))) and card <= 50:
        sens = SensitivityClass.PUBLIC if card <= 20 else SensitivityClass.QUASI_IDENTIFIER
        return "category", "categorical", sens
    return "text", "text", SensitivityClass.FREE_TEXT


def _numeric_stats(s: pd.Series, is_datetime: bool = False) -> NumericStatistics:
    if is_datetime:
        parsed = pd.to_datetime(s, errors="coerce", utc=True).dropna()
        v = (parsed.astype("int64") / 1_000_000_000).to_numpy(float)
        representation = "datetime_epoch_seconds"
    else:
        v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
        representation = "numeric"
    if v.size == 0:
        return NumericStatistics(minimum=0, maximum=0, mean=0,
                                 quantiles=[0.0] * len(_GRID), representation=representation)
    return NumericStatistics(minimum=float(v.min()), maximum=float(v.max()),
                             mean=float(v.mean()),
                             quantiles=np.quantile(v, _GRID).astype(float).tolist(),
                             representation=representation)


def _cat_stats(s: pd.Series) -> CategoricalStatistics:
    vc = s.dropna().astype(str).value_counts(normalize=True)
    return CategoricalStatistics(values=list(vc.index), probabilities=[float(p) for p in vc.values])


def _compile_dependencies(df, cols, key_columns):
    modelling = df.copy()
    for name, profile in cols.items():
        if profile.physical_type == "datetime" and name in modelling:
            parsed = pd.to_datetime(modelling[name], errors="coerce", utc=True)
            modelling[name] = parsed.astype("int64").where(parsed.notna()) / 1_000_000_000
    modellable = [name for name, p in cols.items()
                  if p.physical_type in _MODELLABLE and name in modelling.columns
                  and name not in key_columns]
    if len(modellable) < 2:
        return DependencyGraph(), {}
    dep_graph = infer_dependency_graph(modelling, modellable)
    conditionals = {}
    for name in dep_graph.topological_order(modellable):
        is_numeric = cols[name].physical_type in ("numeric", "datetime")
        parents = [p for p in dep_graph.parents_of(name) if p in modellable]
        conditionals[name] = compile_conditional(modelling, name, parents, is_numeric)
    return dep_graph, conditionals


def _cardinality_models(schema: DatabaseSchema, tables: dict[str, pd.DataFrame]) -> tuple[dict[str, float], dict[str, CardinalityModel]]:
    ratios: dict[str, float] = {}
    models: dict[str, CardinalityModel] = {}
    for fk in schema.foreign_keys:
        parent = tables[fk.parent_table]
        child = tables[fk.child_table]
        key = f"{fk.child_table}->{fk.parent_table}"
        parent_keys = (parent[fk.parent_column].dropna().tolist()
                       if fk.parent_column in parent else [])
        counts = Counter(child[fk.child_column].dropna().tolist()
                         if fk.child_column in child else [])
        observed = [int(counts.get(pk, 0)) for pk in parent_keys]
        if not observed:
            observed = [0]
        freq = Counter(observed)
        values = sorted(freq)
        total = sum(freq.values()) or 1
        probs = [freq[v] / total for v in values]
        ratios[key] = float(np.mean(observed))
        context_column = None
        conditional_values = {}
        conditional_probabilities = {}
        # Compile a bounded parent-state -> child-count model using the first
        # low-cardinality non-key parent attribute. This is source-generic.
        for candidate in parent.columns:
            if candidate == fk.parent_column:
                continue
            series = parent[candidate]
            card = int(series.dropna().nunique())
            if 0 < card <= 32:
                context_column = candidate
                grouped = {}
                for pk_value, state in zip(parent_keys, series.tolist()):
                    grouped.setdefault(str(state), []).append(int(counts.get(pk_value, 0)))
                for state, state_counts in grouped.items():
                    state_freq = Counter(state_counts)
                    state_values = sorted(state_freq)
                    state_total = sum(state_freq.values()) or 1
                    conditional_values[state] = state_values
                    conditional_probabilities[state] = [state_freq[v] / state_total for v in state_values]
                break
        models[key] = CardinalityModel(
            values=values, probabilities=probs, minimum=min(values),
            maximum=max(values), mean=float(np.mean(observed)),
            parent_context_column=context_column,
            conditional_values=conditional_values,
            conditional_probabilities=conditional_probabilities)
    return ratios, models


def profile_dataset(
    schema: DatabaseSchema,
    frames: dict[str, pd.DataFrame],
    sampling: dict[str, SamplingRecord],
) -> tuple[DataProfile, dict[tuple[str, str], set[str]]]:
    tables: dict[str, TableProfile] = {}
    raw_sensitive: dict[tuple[str, str], set[str]] = {}
    for tname, tschema in schema.tables.items():
        df = frames[tname]
        cols = {}
        column_schema = {c.name: c for c in tschema.columns}
        for col in tschema.column_names:
            s = df[col] if col in df.columns else pd.Series([], dtype=object)
            sem, phys, sens = _classify(col, s, column_schema[col].logical_type)
            prof = ColumnProfile(
                table_name=tname, column_name=col, physical_type=phys,
                semantic_type=sem, nullable=column_schema[col].nullable,
                cardinality=int(s.dropna().nunique()) if len(s) else 0,
                missing_rate=float(s.isna().mean()) if len(s) else 0.0,
                sensitivity=sens)
            if phys in ("numeric", "datetime"):
                prof.numeric = _numeric_stats(s, is_datetime=(phys == "datetime"))
            elif phys == "categorical":
                prof.categorical = _cat_stats(s)
            cols[col] = prof
            if sens != SensitivityClass.PUBLIC and len(s):
                raw_sensitive[(tname, col)] = set(s.dropna().astype(str).unique())
        key_cols = set(tschema.primary_key_columns)
        if tschema.primary_key:
            key_cols.add(tschema.primary_key)
        key_cols |= {fk.child_column for fk in schema.foreign_keys
                     if fk.child_table == tname}
        dep_graph, conditionals = _compile_dependencies(df, cols, key_cols)
        tables[tname] = TableProfile(table_name=tname, row_count=int(len(df)),
                                     columns=cols, dependencies=dep_graph,
                                     conditionals=conditionals)
    ratios, card_models = _cardinality_models(schema, frames)
    return DataProfile(tables=tables, sampling=sampling,
                       child_per_parent=ratios,
                       cardinality_models=card_models), raw_sensitive


class ProfilingService:
    def run_dataset(self, schema: DatabaseSchema, frames: dict[str, pd.DataFrame],
                    sampling: dict[str, SamplingRecord]):
        return profile_dataset(schema, frames, sampling)

    def run(self, connector: SourceConnector, schema: DatabaseSchema,
            max_rows: int, seed: int):
        frames = {}
        sampling = {}
        for name, table in schema.tables.items():
            frame = connector.sample_table(name, max_rows, seed)
            frames[name] = frame
            sampling[name] = SamplingRecord(
                strategy="deterministic_bounded", seed=seed,
                population_count=table.row_count, sample_count=len(frame),
                bias_warning=("" if len(frame) >= table.row_count
                              else "bounded sample; statistics are sample-based"))
        return profile_dataset(schema, frames, sampling)
