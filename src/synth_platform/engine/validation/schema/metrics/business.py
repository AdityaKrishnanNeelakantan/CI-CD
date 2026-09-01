"""Business-level grouped metric fidelity."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PREVIEW_ROW_THRESHOLD = 500
MIN_CATEGORY_ROWS = 3
FULL_BUSINESS_TARGET = 0.80
PREVIEW_BUSINESS_TARGET = 0.75


def _capped_relative_error(real: float, synthetic: float, *, epsilon: float = 1e-6) -> float:
    denom = max(abs(real), epsilon)
    return min(1.0, abs(synthetic - real) / denom)


def _business_target(synthetic_rows: int) -> float:
    if synthetic_rows > 0 and synthetic_rows < PREVIEW_ROW_THRESHOLD:
        return PREVIEW_BUSINESS_TARGET
    return FULL_BUSINESS_TARGET


def _align_source_for_comparison(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    compare_seed: int = 42,
) -> pd.DataFrame:
    """Use a same-size source slice for preview comparisons when source is much larger."""
    if synthetic.empty or len(source) <= len(synthetic):
        return source
    if len(synthetic) < min(len(source), PREVIEW_ROW_THRESHOLD):
        return source.sample(n=len(synthetic), random_state=int(compare_seed))
    return source


def _detect_group_numeric_pairs(
    df: pd.DataFrame,
    *,
    max_cardinality: int = 12,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    categorical_cols = []
    numeric_cols = []
    blocked_name_parts = ("email", "phone", "name", "id", "uuid", "address")
    for col in df.columns:
        lowered = str(col).lower()
        if any(part in lowered for part in blocked_name_parts):
            continue
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            numeric_cols.append(col)
            continue
        nunique = series.nunique(dropna=True)
        if 1 < nunique <= max_cardinality:
            categorical_cols.append(col)
    for group_col in categorical_cols:
        for num_col in numeric_cols:
            if group_col != num_col:
                pairs.append((group_col, num_col))
    return pairs[:20]


def _grouped_mean_and_counts(df: pd.DataFrame, group_col: str, num_col: str) -> Tuple[Dict[str, float], Dict[str, int]]:
    frame = (
        df[[group_col, num_col]]
        .dropna()
        .assign(_num=lambda data: pd.to_numeric(data[num_col], errors="coerce"))
        .dropna(subset=["_num"])
    )
    if frame.empty:
        return {}, {}
    grouped = frame.groupby(group_col, observed=True)["_num"]
    means = {str(k): float(v) for k, v in grouped.mean().items()}
    counts = {str(k): int(v) for k, v in grouped.count().items()}
    return means, counts


def compute_business_fidelity(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    grouped_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    compare_seed: int = 42,
    min_category_rows: int = MIN_CATEGORY_ROWS,
) -> Dict[str, Any]:
    """Compare grouped business metrics between source and synthetic tables."""
    source_compare = _align_source_for_comparison(source, synthetic, compare_seed=compare_seed)
    pairs = list(grouped_pairs or _detect_group_numeric_pairs(source_compare))
    metrics: List[Dict[str, Any]] = []
    errors: List[float] = []
    preview_mode = len(synthetic) > 0 and len(synthetic) < PREVIEW_ROW_THRESHOLD

    for group_col, num_col in pairs:
        if group_col not in source_compare.columns or num_col not in source_compare.columns:
            continue
        if group_col not in synthetic.columns or num_col not in synthetic.columns:
            continue
        real, real_counts = _grouped_mean_and_counts(source_compare, group_col, num_col)
        syn, syn_counts = _grouped_mean_and_counts(synthetic, group_col, num_col)
        categories = sorted(set(real.keys()) & set(syn.keys()))
        pair_errors = []
        for cat in categories:
            if real_counts.get(cat, 0) < min_category_rows or syn_counts.get(cat, 0) < min_category_rows:
                continue
            err = _capped_relative_error(real[cat], syn[cat])
            pair_errors.append(err)
        if pair_errors:
            pair_score = 1.0 - float(np.mean(pair_errors))
            errors.extend(pair_errors)
            metrics.append(
                {
                    "group_column": group_col,
                    "metric_column": num_col,
                    "metric_type": "grouped_mean",
                    "score": round(pair_score, 6),
                    "categories_compared": len(pair_errors),
                }
            )

    for col in source_compare.columns:
        if col not in synthetic.columns:
            continue
        lowered = str(col).lower()
        if any(part in lowered for part in ("email", "phone", "name", "id", "uuid", "address")):
            continue
        if source_compare[col].nunique(dropna=True) > 12:
            continue
        if pd.api.types.is_bool_dtype(source_compare[col]) or source_compare[col].astype(str).str.lower().isin(
            {"true", "false", "yes", "no", "0", "1"}
        ).mean() > 0.8:
            real_rate = float(source_compare[col].astype(str).str.lower().isin({"true", "1", "yes"}).mean())
            syn_rate = float(synthetic[col].astype(str).str.lower().isin({"true", "1", "yes"}).mean())
            err = _capped_relative_error(real_rate, syn_rate)
            errors.append(err)
            metrics.append(
                {
                    "group_column": col,
                    "metric_column": col,
                    "metric_type": "positive_rate",
                    "score": round(1.0 - err, 6),
                }
            )

    score = 1.0 - float(np.mean(errors)) if errors else 1.0
    target = _business_target(len(synthetic))
    note = None
    if not errors:
        note = "Not enough stable grouped categories on this preview sample; skipped business pattern check."
        score = 1.0
    elif preview_mode:
        note = (
            f"Preview sample ({len(synthetic):,} rows): compared overlapping categories with "
            f">={min_category_rows} rows each; target relaxed to {target:.0%}."
        )

    return {
        "metric": "business_fidelity",
        "score": round(max(0.0, score), 6),
        "target": target,
        "passed": score >= target,
        "metrics": metrics,
        "preview_mode": preview_mode,
        "note": note,
    }
