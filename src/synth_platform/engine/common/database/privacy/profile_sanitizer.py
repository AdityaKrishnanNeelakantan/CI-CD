"""Enforces the privacy boundary between the protected training zone and
a portable generator artifact.

src/profiling/profiler.py deliberately keeps exact minimum/maximum/
category_frequencies/representative_examples in its output - in-pipeline
consumers (semantic inference, cleaning) need that exact evidence, and
nothing about profiling itself crosses a trust boundary. Once a profile
is about to be written into a generator ZIP that leaves the training
environment, those exact fields are exactly the ones the project's own
privacy review flags as unsafe (real outliers, rare-category vocabularies,
retained sample values). This module is the one place that decides what
survives that boundary - src/artifact/builder.py calls it and nothing
else is allowed to write reference_profile.json's content.
"""

from __future__ import annotations

from typing import Any

from synth_platform.engine.inference.database.contract import SENSITIVE_SEMANTIC_TYPES

# Every one of these has a safe, already-computed companion field
# (generation_lower_bound/upper_bound, safe_category_frequencies) except
# representative_examples, which has no safe companion because a
# "representative value" is definitionally a real value - there is
# nothing to redact into and still call it representative.
_UNSAFE_COLUMN_KEYS = frozenset({"minimum", "maximum", "category_frequencies", "representative_examples"})

# safe_category_frequencies is only "safe" from RARE-value overexposure
# (src/privacy/rare_category.py's minimum_support suppression) - it does
# NOT know about semantic sensitivity, because profiling runs before
# semantic inference has decided anything. A column that turns out to be
# person_name/phone_number/email/identifier can easily have every one of
# its values appear far more often than any reasonable minimum_support
# threshold (confirmed: 16 real patient surnames, each 100+ occurrences,
# verbatim with exact counts, survived into an exported artifact this
# way) - genuinely safe for an ordinary category (blood_type, state) but
# a real PII-vocabulary leak for these types.
_SENSITIVE_COLUMN_KEYS = frozenset({"safe_category_frequencies"})


def _sanitize_columns(columns: dict[str, Any], sensitive_columns: frozenset[str] | None = None) -> dict[str, Any]:
    sensitive_columns = sensitive_columns or frozenset()
    result = {}
    for column_name, column_profile in columns.items():
        drop_keys = _UNSAFE_COLUMN_KEYS
        if column_name in sensitive_columns:
            drop_keys = drop_keys | _SENSITIVE_COLUMN_KEYS
        result[column_name] = {k: v for k, v in column_profile.items() if k not in drop_keys}
    return result


def _sensitive_columns_by_table(dataset_contract: dict[str, Any] | None) -> dict[str, frozenset[str]]:
    if not dataset_contract:
        return {}
    return {
        table_name: frozenset(
            column_name
            for column_name, column_contract in table.get("columns", {}).items()
            if column_contract.get("semantic_type") in SENSITIVE_SEMANTIC_TYPES
        )
        for table_name, table in dataset_contract.get("tables", {}).items()
    }


def sanitize_profile_for_export(
    profile: dict[str, Any], dataset_contract: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Accepts either a single src/profiling/profiler.py profile_table()
    result (top-level "columns") or - the shape actually passed as
    reference_profile in this project - a run_profiling() multi-table
    wrapper ({"tables": {table_name: {"columns": {...}}}}).

    dataset_contract (the approved semantic-type decisions, available by
    the time an artifact is built) drives the additional per-column
    sensitive-key stripping above - optional and backward compatible
    (omitting it just skips that extra protection) so any other caller
    of this function is unaffected.
    """
    if "tables" in profile:
        sensitive = _sensitive_columns_by_table(dataset_contract)
        sanitized_tables = {
            table_name: {
                **table,
                "columns": _sanitize_columns(table.get("columns", {}), sensitive.get(table_name)),
            }
            for table_name, table in profile["tables"].items()
        }
        return {**profile, "tables": sanitized_tables}

    return {**profile, "columns": _sanitize_columns(profile.get("columns", {}))}
