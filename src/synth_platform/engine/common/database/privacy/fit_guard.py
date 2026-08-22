"""Hard guards so raw context/PII values never enter model fit.

Database Twin learns only statistical DNA (numeric/datetime/boolean
distributions and safe category frequencies after rare-category
suppression). Context-aware columns are replaced by deterministic Faker
stand-ins before ``SynthesizerAdapter.fit`` and are excluded from the
copula entirely.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from synth_platform.engine.common.database.privacy.context_fields import (
    CONTEXT_AWARE_KINDS,
    resolve_context_kind,
)

APPROVED_STATUS = "approved"


class FitPrivacyError(RuntimeError):
    """Raised when a fit frame still contains raw source context values."""


def context_aware_columns(table_contract: dict[str, Any]) -> list[str]:
    """Approved columns that must be masked and never statistically fitted."""
    names: list[str] = []
    for column_name, column_contract in table_contract.get("columns", {}).items():
        if column_contract.get("inference_status") != APPROVED_STATUS:
            continue
        kind = resolve_context_kind(column_name, column_contract.get("semantic_type"))
        if kind in CONTEXT_AWARE_KINDS:
            names.append(column_name)
    return names


def collect_non_null_strings(series: pd.Series) -> set[str]:
    values: set[str] = set()
    for raw in series.tolist():
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        text = str(raw).strip()
        if text:
            values.add(text)
    return values


def assert_no_raw_context_in_fit_frame(
    source_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    table_contract: dict[str, Any],
    *,
    table_name: str = "",
    max_unchanged_ratio: float = 0.25,
) -> None:
    """Ensure context-aware columns were rewritten before fit.

    Requires that most source values were replaced. Occasional Faker
    collisions with the source string are allowed (especially when the
    source fixture was itself Faker-generated); a no-op mask is not.
    """
    leaks: list[str] = []
    for column_name in context_aware_columns(table_contract):
        if column_name not in source_df.columns or column_name not in fit_df.columns:
            continue
        source_series = source_df[column_name]
        fit_series = fit_df[column_name]
        comparable = 0
        unchanged = 0
        for raw, masked in zip(source_series.tolist(), fit_series.tolist()):
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            comparable += 1
            if str(raw).strip() == str(masked).strip():
                unchanged += 1
        if comparable == 0:
            continue
        ratio = unchanged / comparable
        if ratio > max_unchanged_ratio:
            leaks.append(
                f"{column_name}: {unchanged}/{comparable} values unchanged after mask "
                f"({ratio:.0%} > {max_unchanged_ratio:.0%} allowed)"
            )
    if leaks:
        where = f" table={table_name!r}" if table_name else ""
        raise FitPrivacyError(
            "pre-input mask failed: raw context values would enter model fit"
            f"{where}: " + "; ".join(leaks)
        )


def blob_contains_any(blob: str, forbidden: Iterable[str]) -> list[str]:
    """Return forbidden strings that appear as substrings in ``blob``."""
    hits: list[str] = []
    for value in forbidden:
        text = str(value).strip()
        if len(text) < 3:
            continue
        if text in blob:
            hits.append(text)
    return hits
