"""Dependency inference + conditional-model compilation (engine).

Generic and domain-free. Two responsibilities, both at compile time:

1. infer_dependency_graph(frame, columns) -> DependencyGraph
   Screens column pairs by normalized mutual information (NMI). For each column,
   its single strongest already-ordered predecessor above a threshold becomes
   its parent. Producing a *single-parent* DAG keeps the conditional tables
   dense enough to be reliable and cannot create cycles (a child only points to
   an earlier column). This is the honest baseline; multi-parent is a future
   extension behind the same interface.

2. compile_conditionals(frame, dep_graph, ...) -> dict[column, ConditionalModel]
   For each column with parents, buckets the child's distribution by the
   discretized parent context; roots get an unconditional model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synth_platform.domain.generation.conditional import (
    CategoricalConditional, ConditionalModel, NumericConditional,
)
from synth_platform.domain.semantics.dependencies import (
    DependencyEdge, DependencyGraph,
)

_NUMERIC_BINS = 4          # numeric parent -> quartile bands (discretization)
_QUANTILE_GRID = [i / 20 for i in range(21)]
_MIN_BUCKET = 8            # contexts with fewer rows fall back to the marginal
_NMI_THRESHOLD = 0.35      # below this, treat as independent (finite-sample
                           # NMI noise between independent columns runs ~0.05-0.30)
_MIN_ROWS_FOR_EDGE = 30    # too few rows -> NMI unreliable, infer no edges


def _discretize(s: pd.Series) -> pd.Series:
    """Categorical -> string; numeric -> quantile-band label. Generic."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > _NUMERIC_BINS:
        try:
            return pd.qcut(s, _NUMERIC_BINS, labels=False, duplicates="drop").astype("Int64").astype(str)
        except (ValueError, IndexError):
            return s.astype(str)
    return s.astype(str)


def _nmi(a: pd.Series, b: pd.Series) -> float:
    """Normalized mutual information between two discretized series, in [0,1]."""
    da, db = _discretize(a).fillna("∅"), _discretize(b).fillna("∅")
    ct = pd.crosstab(da, db).to_numpy(dtype=float)
    n = ct.sum()
    if n == 0:
        return 0.0
    pij = ct / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(pij * np.log(pij / (pi * pj + 1e-12) + 1e-12) * (pij > 0))
    ha = -np.nansum(pi * np.log(pi + 1e-12))
    hb = -np.nansum(pj * np.log(pj + 1e-12))
    denom = min(ha, hb)
    return float(max(0.0, mi / denom)) if denom > 1e-12 else 0.0


def infer_dependency_graph(frame: pd.DataFrame, columns: list[str]) -> DependencyGraph:
    """Single-parent DAG: each column points to its strongest earlier predictor.
    Column order fixes acyclicity; ties broken by the column order given."""
    edges: list[DependencyEdge] = []
    if len(frame) < _MIN_ROWS_FOR_EDGE:
        return DependencyGraph(edges=edges)
    usable = [c for c in columns if c in frame.columns and frame[c].notna().any()]
    for i, child in enumerate(usable):
        best_parent, best_nmi = None, _NMI_THRESHOLD
        for parent in usable[:i]:  # only earlier columns -> no cycles
            score = _nmi(frame[parent], frame[child])
            if score > best_nmi:
                best_parent, best_nmi = parent, score
        if best_parent is not None:
            edges.append(DependencyEdge(parent=best_parent, child=child, strength=best_nmi))
    return DependencyGraph(edges=edges)


def _context_key(values: tuple) -> str:
    return "|".join(str(v) for v in values)


def _categorical_dist(s: pd.Series) -> CategoricalConditional:
    vc = s.dropna().astype(str).value_counts(normalize=True)
    return CategoricalConditional(values=list(vc.index), probabilities=[float(p) for p in vc.values])


def _numeric_dist(s: pd.Series) -> NumericConditional:
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if v.size == 0:
        return NumericConditional(quantiles=[0.0] * len(_QUANTILE_GRID), minimum=0.0, maximum=0.0)
    return NumericConditional(quantiles=np.quantile(v, _QUANTILE_GRID).astype(float).tolist(),
                              minimum=float(v.min()), maximum=float(v.max()))


def compile_conditional(
    frame: pd.DataFrame, child: str, parents: list[str], child_is_numeric: bool
) -> ConditionalModel:
    kind = "numeric" if child_is_numeric else "categorical"
    model = ConditionalModel(parents=parents, kind=kind)
    # unconditional fallback (also the model for roots)
    if child_is_numeric:
        model.numeric_fallback = _numeric_dist(frame[child])
    else:
        model.categorical_fallback = _categorical_dist(frame[child])
    if not parents:
        return model
    disc = {p: _discretize(frame[p]).fillna("∅") for p in parents}
    ctx = pd.DataFrame(disc)
    grouper = [ctx[p] for p in parents]
    groups = ctx.groupby(grouper[0] if len(grouper) == 1 else grouper).groups
    for key_tuple, idx in groups.items():
        if len(idx) < _MIN_BUCKET:
            continue
        key = _context_key(key_tuple if isinstance(key_tuple, tuple) else (key_tuple,))
        sub = frame.loc[idx, child]
        if child_is_numeric:
            model.numeric_by_context[key] = _numeric_dist(sub)
        else:
            model.categorical_by_context[key] = _categorical_dist(sub)
    return model
