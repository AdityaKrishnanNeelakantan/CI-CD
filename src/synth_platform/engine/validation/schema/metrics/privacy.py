"""Privacy metrics: PII replay and exact row overlap."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd

from synth_platform.engine.generation.schema.pii_columns import validate_no_pii_replay


def _normalize_row(row: Iterable[Any]) -> tuple[str, ...]:
    return tuple("" if pd.isna(v) else str(v).strip().lower() for v in row)


def exact_row_match_rate(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Measure how many synthetic rows exactly match a source row."""
    shared = [c for c in (columns or source.columns) if c in source.columns and c in synthetic.columns]
    if not shared or source.empty or synthetic.empty:
        return {
            "metric": "exact_row_match",
            "score": 0.0,
            "rate": 0.0,
            "target": 0.0,
            "passed": True,
            "matching_rows": 0,
            "synthetic_rows": len(synthetic),
        }

    source_rows = {_normalize_row(row) for row in source[shared].itertuples(index=False, name=None)}
    matches = 0
    for row in synthetic[shared].itertuples(index=False, name=None):
        if _normalize_row(row) in source_rows:
            matches += 1
    rate = matches / max(len(synthetic), 1)
    return {
        "metric": "exact_row_match",
        "score": round(rate, 6),
        "rate": round(rate, 6),
        "target": 0.0,
        "passed": rate <= 0.001,
        "matching_rows": matches,
        "synthetic_rows": len(synthetic),
        "columns_compared": list(shared),
    }


def compute_privacy_metrics(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    fresh_columns: Sequence[str],
) -> Dict[str, Any]:
    """Return PII replay and exact-row overlap metrics."""
    replay = validate_no_pii_replay(source, synthetic, fresh_columns)
    exact = exact_row_match_rate(source, synthetic)
    return {
        "pii_replay_count": {
            "metric": "pii_replay_count",
            "score": float(replay.get("total_replay_values", 0)),
            "target": 0,
            "passed": bool(replay.get("passed", False)),
            "column_replays": replay.get("column_replays", {}),
        },
        "exact_row_match": exact,
    }
