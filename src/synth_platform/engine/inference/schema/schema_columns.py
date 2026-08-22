"""Helpers for building schema columns from profiles and structured review issues."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from synth_platform.engine.inference.schema.schema import Column


def unique_int_capacity(low: int, high: int) -> int:
    """Inclusive count of integer values in [low, high]."""
    return max(0, int(high) - int(low) + 1)


def minimum_unique_int_high(low: int, row_count: int, *, buffer: int = 100) -> int:
    """Minimum inclusive max for a unique int column serving row_count rows."""
    return int(low) + max(1, int(row_count)) - 1 + int(buffer)


def resolve_unique_int_bounds(
    params: Dict[str, Any],
    row_count: int,
    *,
    buffer: int = 100,
) -> tuple[int, int]:
    """Return low/high bounds large enough for row_count unique integers."""
    low = int(params.get("min", 0))
    high = int(params.get("max", 1000))
    needed_high = minimum_unique_int_high(low, row_count, buffer=buffer)
    if unique_int_capacity(low, high) < max(1, int(row_count)):
        high = needed_high
    return low, high


def align_unique_int_ranges(
    columns: Sequence[Column],
    row_count: int,
    *,
    buffer: int = 100,
) -> None:
    """Tighten unique int max bounds to match a table's row count (e.g. preview scaling)."""
    for column in columns:
        if column.type != "int" or not column.unique:
            continue
        params = dict(column.distribution_params or {})
        low = int(params.get("min", 0))
        needed_high = minimum_unique_int_high(low, row_count, buffer=buffer)
        params["max"] = needed_high
        column.distribution_params = params


def categorical_choices_from_profile(col_profile: Any) -> Optional[List[str]]:
    """Derive categorical choices from a source ColumnProfile marginal."""
    marginal = getattr(col_profile, "marginal", None) or {}
    if isinstance(marginal, dict):
        if marginal.get("choices"):
            return [str(value) for value in marginal["choices"]]
        top_values = marginal.get("top_values") or []
        if top_values:
            choices = [
                str(item["value"])
                for item in top_values
                if isinstance(item, dict) and item.get("value") is not None
            ]
            if choices:
                return choices
    return None


def column_from_source_profile(name: str, col_profile: Any) -> Column:
    """Build a schema Column from a source profile column without silent Unknown fallback."""
    kind = str(getattr(col_profile, "kind", "text"))
    type_map = {
        "numeric": "float",
        "categorical": "categorical",
        "date": "date",
        "boolean": "boolean",
        "id_like": "int",
        "text": "text",
    }
    col_type = type_map.get(kind, "text")
    params: Dict[str, Any] = {}

    if col_type == "categorical":
        choices = categorical_choices_from_profile(col_profile)
        if choices:
            params = {"choices": choices, "_derived_from_profile": True}
        else:
            params = {"_missing_categorical_choices": True}

    return Column(name=name, type=col_type, distribution_params=params)


def collect_categorical_review_issues(columns: Sequence[Column]) -> List[Dict[str, Any]]:
    """Return structured REVIEW issues for categorical columns missing explicit choices."""
    issues: List[Dict[str, Any]] = []
    for column in columns:
        if column.type != "categorical":
            continue
        params = column.distribution_params or {}
        if params.get("_missing_categorical_choices"):
            issues.append(
                {
                    "column": column.name,
                    "issue": "missing_categorical_choices",
                    "fallback_used": bool(params.get("_categorical_fallback")),
                    "fallback_value": (params.get("choices") or [None])[0]
                    if params.get("_categorical_fallback")
                    else None,
                    "severity": "REVIEW",
                    "fix_suggestion": "Provide choices in schema or source profile.",
                }
            )
        elif params.get("_categorical_fallback"):
            issues.append(
                {
                    "column": column.name,
                    "issue": "missing_categorical_choices",
                    "fallback_used": True,
                    "fallback_value": (params.get("choices") or ["Unknown"])[0],
                    "severity": "REVIEW",
                    "fix_suggestion": "Provide choices in schema or source profile.",
                }
            )
    return issues
