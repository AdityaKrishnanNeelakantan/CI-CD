"""Compile-time privacy verifier: PROVE the artifact is source-free of sensitive
values. Stronger than the load-time forbidden-payload scan (which only catches
credentials/URLs) — this asserts the privacy invariant itself:

    no value from a column whose policy action is BUCKET/SUPPRESS/TRANSFORM/
    EXCLUDE appears anywhere in the compiled profile.

Runs at compile time, after redaction, before the artifact is written. If it
raises, redaction is incomplete — a bug, not a warning.
"""
from __future__ import annotations

from synth_platform.domain.privacy.models import PrivacyAction, PrivacyPolicy
from synth_platform.domain.profiling.models import DataProfile
from synth_platform.errors import PrivacyLeakError

_TRANSFORMED = (PrivacyAction.BUCKET, PrivacyAction.SUPPRESS,
                PrivacyAction.TRANSFORM, PrivacyAction.EXCLUDE)


def verify_source_free(
    profile: DataProfile, policy: PrivacyPolicy, raw_values: dict[tuple[str, str], set[str]]
) -> None:
    """`raw_values[(table, column)]` is the set of real values seen for that
    column at profile time. For every non-public column, assert none of those
    raw values survive in the redacted profile (marginals or conditionals)."""
    actions = {(p.table, p.column): p.action for p in policy.columns}
    for tname, tprof in profile.tables.items():
        for col, cprof in tprof.columns.items():
            if actions.get((tname, col), PrivacyAction.PRESERVE) not in _TRANSFORMED:
                continue
            forbidden = raw_values.get((tname, col), set())
            if not forbidden:
                continue
            _assert_absent(forbidden, tprof, col, tname)


def _assert_absent(forbidden: set[str], tprof, col: str, tname: str) -> None:
    # 1) the column's own marginal must not contain a raw value
    cprof = tprof.columns[col]
    if cprof.categorical and set(cprof.categorical.values) & forbidden:
        raise PrivacyLeakError(
            f"{tname}.{col}: raw sensitive value survived in marginal")
    # 2) no conditional model may store the raw value as a child label or a
    #    parent-context key component
    for child, model in tprof.conditionals.items():
        if child == col:
            labels = set()
            if model.categorical_fallback:
                labels |= set(model.categorical_fallback.values)
            for cc in model.categorical_by_context.values():
                labels |= set(cc.values)
            if labels & forbidden:
                raise PrivacyLeakError(
                    f"{tname}.{col}: raw value survived in a conditional distribution")
        if col in model.parents:
            keys = set()
            for k in list(model.categorical_by_context) + list(model.numeric_by_context):
                keys |= set(k.split("|"))
            if keys & forbidden:
                raise PrivacyLeakError(
                    f"{tname}.{col}: raw value survived in a parent-context key")
