"""Build safe row context packets for LLM text generation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from synth_platform.engine.generation.schema.pii_columns import resolve_column_semantic
from synth_platform.engine.inference.schema.schema import Column

MAX_CONTEXT_COLUMNS = 8
MAX_CONTEXT_VALUE_LEN = 80


def _is_safe_context_column(column_name: str, column: Optional[Column] = None) -> bool:
    if column is not None and column.type != "text":
        if column.type in {"int", "float", "boolean", "categorical", "date", "datetime"}:
            return True
        return False
    semantic = resolve_column_semantic(column_name)
    if semantic:
        return False
    lowered = column_name.lower()
    blocked = (
        "email",
        "phone",
        "address",
        "ssn",
        "name",
        "password",
        "account",
        "routing",
        "card",
        "iban",
        "username",
    )
    return not any(part in lowered for part in blocked)


def _redact_value(value: Any, *, column_name: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if column_name and resolve_column_semantic(column_name):
        return "[redacted]"
    if "@" in text and "." in text:
        return "[redacted]"
    if any(ch.isdigit() for ch in text) and len(text) > 6:
        return "[redacted]"
    if len(text) > MAX_CONTEXT_VALUE_LEN:
        text = text[: MAX_CONTEXT_VALUE_LEN - 3] + "..."
    return text


def build_safe_row_context(
    row: Mapping[str, Any],
    *,
    target_column: str,
    columns: Optional[Sequence[Column]] = None,
    allowed_columns: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Extract safe, non-sensitive context from one row."""
    col_map = {c.name: c for c in (columns or [])}
    keys = list(allowed_columns) if allowed_columns else list(row.keys())
    context: Dict[str, str] = {}
    for key in keys:
        if key == target_column:
            continue
        column = col_map.get(key)
        if not _is_safe_context_column(key, column):
            continue
        value = _redact_value(row.get(key), column_name=key)
        if value:
            context[key] = value
        if len(context) >= MAX_CONTEXT_COLUMNS:
            break
    return context


def build_batch_contexts(
    table_data: Optional[pd.DataFrame],
    *,
    target_column: str,
    columns: Optional[Sequence[Column]] = None,
) -> List[Dict[str, str]]:
    if table_data is None or table_data.empty:
        return []
    contexts: List[Dict[str, str]] = []
    for _, row in table_data.iterrows():
        contexts.append(
            build_safe_row_context(
                row,
                target_column=target_column,
                columns=columns,
            )
        )
    return contexts


def context_cache_key(context: Dict[str, str]) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(context.items()))
