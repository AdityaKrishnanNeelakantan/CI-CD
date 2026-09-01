"""Compile-time privacy redaction: transform the profile BEFORE serialization.

The mandate: "Transform before serialization. Never after." This pass rewrites a
DataProfile under a PrivacyPolicy so that no raw value from a non-public column
survives into the compiled artifact. It runs at compile time, between profiling
and artifact assembly — not at generation time, not at load time.

Actions (from domain/privacy/models.PrivacyAction):
  PRESERVE   keep the profile as-is (public business values)
  BUCKET     replace category labels with stable placeholders CATEGORY_0001…
             (distribution shape kept; meaning removed)
  SUPPRESS   replace labels with placeholders AND drop the values from any
             conditional model that keys on them
  TRANSFORM  clear stored marginal values (identifiers are synthesized at
             generation from reserved namespaces, never from stored data)
  EXCLUDE    clear the column's stored statistics entirely (free text)

A verifier (verify_source_free) then PROVES the invariant on the finished
artifact — a compile-time gate, not a hope.
"""
from __future__ import annotations

from synth_platform.domain.privacy.models import PrivacyAction, PrivacyPolicy
from synth_platform.domain.profiling.models import (
    CategoricalStatistics, DataProfile,
)


def _placeholder_map(values: list[str]) -> dict[str, str]:
    return {v: f"CATEGORY_{i + 1:04d}" for i, v in enumerate(values)}


def redact_profile(profile: DataProfile, policy: PrivacyPolicy) -> DataProfile:
    """Return a copy of the profile with non-public column values transformed.
    Also rewrites conditional models so no raw sensitive value survives in a
    bucket key or a bucketed distribution."""
    actions = {(p.table, p.column): p.action for p in policy.columns}
    prof = profile.model_copy(deep=True)

    for tname, tprof in prof.tables.items():
        remap: dict[str, dict[str, str]] = {}   # column -> {raw -> placeholder}
        cleared: set[str] = set()

        for col, cprof in tprof.columns.items():
            action = actions.get((tname, col), PrivacyAction.PRESERVE)
            if action in (PrivacyAction.BUCKET, PrivacyAction.SUPPRESS) and cprof.categorical:
                mapping = _placeholder_map(cprof.categorical.values)
                remap[col] = mapping
                cprof.categorical = CategoricalStatistics(
                    values=list(mapping.values()),
                    probabilities=cprof.categorical.probabilities)
            elif action == PrivacyAction.TRANSFORM:
                cprof.categorical = None      # synthesized at generation, not stored
                cprof.numeric = None
                cleared.add(col)
            elif action == PrivacyAction.EXCLUDE:
                cprof.categorical = None
                cprof.numeric = None
                cleared.add(col)

        _rewrite_conditionals(tprof, remap, cleared)

    return prof


def _rewrite_conditionals(tprof, remap: dict, cleared: set) -> None:
    """Apply the same label remap to conditional models: relabel child
    categorical values, remap parent-context keys, and drop models for cleared
    columns (and any model conditioned on a cleared parent)."""
    new_conditionals = {}
    for child, model in tprof.conditionals.items():
        if child in cleared or any(p in cleared for p in model.parents):
            continue  # cannot condition on / model a removed column
        # relabel the child's own categorical distributions
        if model.kind == "categorical" and child in remap:
            m = remap[child]
            if model.categorical_fallback:
                model.categorical_fallback.values = [
                    m.get(v, v) for v in model.categorical_fallback.values]
            for cc in model.categorical_by_context.values():
                cc.values = [m.get(v, v) for v in cc.values]
        # remap parent-context keys when a parent was bucketed
        if any(p in remap for p in model.parents):
            model.categorical_by_context = _remap_keys(
                model.categorical_by_context, model.parents, remap)
            model.numeric_by_context = _remap_keys(
                model.numeric_by_context, model.parents, remap)
        new_conditionals[child] = model
    tprof.conditionals = new_conditionals


def _remap_keys(by_context: dict, parents: list[str], remap: dict) -> dict:
    out = {}
    for key, dist in by_context.items():
        parts = key.split("|")
        new_parts = [remap[p].get(part, part) if p in remap else part
                     for p, part in zip(parents, parts)]
        out["|".join(new_parts)] = dist
    return out
