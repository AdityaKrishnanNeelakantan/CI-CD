"""Column generators: a registry of strategies resolved by the compiled plan.

Replaces the former `sample_column` type-switch (ARCHITECTURE_AUDIT A-08, OCP)
with a `ColumnGenerator` Protocol + registry. The executor calls
`generator.sample(ctx)` with no type knowledge; adding a behavior (sequence,
timeline, state) is a new registry entry, not an edited `if` ladder.

Conditional generation (RC-1/A-01) lives here: a column with a compiled
`ConditionalModel` that has parents draws per-row from P(child | parent-context),
falling back to the marginal for unseen contexts. Independent marginal sampling
survives ONLY as the degenerate no-parent (root) case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from synth_platform.domain.generation.conditional import ConditionalModel
from synth_platform.domain.profiling.models import ColumnProfile
from synth_platform.engine.generation.pii import generate_pii


@dataclass
class ColumnContext:
    """Everything a generator needs: row count, RNG, the column profile, the
    compiled conditional model, and the already-generated columns of this table
    (for conditioning on parent values)."""
    n: int
    rng: np.random.Generator
    profile: ColumnProfile
    conditional: ConditionalModel | None
    generated: dict[str, np.ndarray]
    action: str


def _numeric_from(quantiles, minimum, maximum, u):
    grid = np.linspace(0, 1, len(quantiles))
    vals = np.interp(u, grid, quantiles)
    spread = (maximum - minimum) or 1.0
    return np.clip(vals, minimum, maximum), spread


def _discretize_value(v, is_numeric_parent, edges):
    """Match the compile-time discretization for a single parent value."""
    if is_numeric_parent and edges is not None:
        return str(int(np.digitize([float(v)], edges)[0])) if pd.notna(v) else "∅"
    return "∅" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def _conditional_sample(ctx: ColumnContext) -> np.ndarray:
    """Draw P(child | parent-context) row-by-row, grouping rows that share a
    context so each group is one vectorized draw. Falls back to the marginal."""
    cm = ctx.conditional
    rng, n = ctx.rng, ctx.n
    parents = cm.parents
    # build a context key per row from already-generated parent columns
    keys = np.empty(n, dtype=object)
    parent_arrays = {p: ctx.generated.get(p) for p in parents}
    for i in range(n):
        parts = []
        for p in parents:
            arr = parent_arrays[p]
            parts.append("∅" if arr is None else str(arr[i]))
        keys[i] = "|".join(parts)

    out = np.empty(n, dtype=object) if cm.kind == "categorical" else np.empty(n, dtype=float)
    ctx_map = cm.categorical_by_context if cm.kind == "categorical" else cm.numeric_by_context
    for key in np.unique(keys):
        mask = keys == key
        m = int(mask.sum())
        dist = ctx_map.get(key)
        if cm.kind == "categorical":
            d = dist or cm.categorical_fallback
            if d is None or not d.values:
                out[mask] = ""
                continue
            p = np.asarray(d.probabilities, float)
            p = p / p.sum() if p.sum() > 0 else np.ones(len(d.values)) / len(d.values)
            out[mask] = rng.choice(d.values, size=m, p=p)
        else:
            d = dist or cm.numeric_fallback
            if d is None:
                out[mask] = 0.0
                continue
            vals, spread = _numeric_from(d.quantiles, d.minimum, d.maximum, rng.random(m))
            out[mask] = np.clip(vals + rng.normal(0, spread * 0.01, m), d.minimum, d.maximum)
    return out


def _marginal_numeric(prof: ColumnProfile, rng, n):
    ns = prof.numeric
    if ns is None:
        return np.zeros(n)
    vals, spread = _numeric_from(ns.quantiles, ns.minimum, ns.maximum, rng.random(n))
    return np.clip(vals + rng.normal(0, spread * 0.01, n), ns.minimum, ns.maximum)


def _marginal_categorical(prof: ColumnProfile, rng, n):
    cs = prof.categorical
    if cs is None or not cs.values:
        return np.array([""] * n, dtype=object)
    p = np.array(cs.probabilities, float)
    p = p / p.sum() if p.sum() > 0 else np.ones(len(cs.values)) / len(cs.values)
    return rng.choice(cs.values, size=n, p=p)


def sample_column(ctx: ColumnContext) -> np.ndarray:
    """Resolve and run the right generator. Conditional first when the compiled
    model has parents; otherwise the marginal/degenerate strategies."""
    prof, rng, n = ctx.profile, ctx.rng, ctx.n
    if ctx.action == "transform" or prof.physical_type == "pii":
        return np.array(generate_pii(prof.semantic_type or "person_name", rng, n), dtype=object)
    if ctx.action == "exclude" or prof.physical_type == "text":
        return np.array([None] * n, dtype=object)
    if prof.physical_type == "identifier":
        return np.arange(1, n + 1)
    # CONDITIONAL path: a compiled model with parents preserves correlation.
    if ctx.conditional is not None and ctx.conditional.is_conditional:
        return _conditional_sample(ctx)
    # degenerate root case: marginal only.
    if prof.physical_type in ("numeric", "datetime"):
        return _marginal_numeric(prof, rng, n)
    if prof.physical_type == "categorical":
        return _marginal_categorical(prof, rng, n)
    return np.array([None] * n, dtype=object)
