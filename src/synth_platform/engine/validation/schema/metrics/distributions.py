"""Distribution fidelity metrics: KS, TV, missingness, correlation similarity."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from synth_platform.engine.inference.schema.schema import SchemaConfig


def _ks_similarity(real: pd.Series, synthetic: pd.Series) -> tuple[Optional[float], Dict[str, Any]]:
    real_num = pd.to_numeric(real, errors="coerce").dropna()
    syn_num = pd.to_numeric(synthetic, errors="coerce").dropna()
    if len(real_num) < 5 or len(syn_num) < 5:
        return None, {"method": None, "sampled_rows": 0}
    sampled_rows = int(min(len(real_num), len(syn_num)))
    method = "asymp" if sampled_rows > 5000 else "auto"
    statistic, _ = stats.ks_2samp(real_num, syn_num, method=method)
    score = float(max(0.0, 1.0 - statistic))
    return score, {
        "method": method,
        "sampled_rows": sampled_rows,
        "warning_suppressed": False,
    }


def _tv_similarity(real: pd.Series, synthetic: pd.Series) -> Optional[float]:
    real_rates = real.astype(str).value_counts(normalize=True, dropna=True)
    syn_rates = synthetic.astype(str).value_counts(normalize=True, dropna=True)
    categories = sorted(set(real_rates.index) | set(syn_rates.index))
    if not categories:
        return None
    tv = 0.5 * sum(abs(float(real_rates.get(cat, 0.0)) - float(syn_rates.get(cat, 0.0))) for cat in categories)
    return float(max(0.0, 1.0 - tv))


def _missing_similarity(real: pd.Series, synthetic: pd.Series) -> float:
    real_null = float(real.isna().mean())
    syn_null = float(synthetic.isna().mean())
    return float(max(0.0, 1.0 - abs(real_null - syn_null)))


def _correlation_matrix_similarity(real: pd.DataFrame, synthetic: pd.DataFrame, columns: Sequence[str]) -> Optional[float]:
    cols = [c for c in columns if c in real.columns and c in synthetic.columns]
    if len(cols) < 2:
        return None
    real_corr = real[cols].corr(numeric_only=True)
    syn_corr = synthetic[cols].corr(numeric_only=True)
    if real_corr.empty or syn_corr.empty:
        return None
    delta = (real_corr - syn_corr).abs().values
    delta = delta[~np.isnan(delta)]
    if delta.size == 0:
        return None
    return float(max(0.0, 1.0 - float(np.mean(delta)) / 2.0))


def compare_source_synthetic_distributions(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    numeric_columns: Optional[Sequence[str]] = None,
    categorical_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compare source vs synthetic columns using KS/TV/missing/correlation metrics."""
    shared = [c for c in source.columns if c in synthetic.columns]
    numeric_cols = list(numeric_columns or [])
    if not numeric_cols:
        numeric_cols = [
            c
            for c in shared
            if pd.api.types.is_numeric_dtype(source[c]) and pd.api.types.is_numeric_dtype(synthetic[c])
        ]
    categorical_cols = list(categorical_columns or [])
    if not categorical_cols:
        categorical_cols = [
            c
            for c in shared
            if not pd.api.types.is_numeric_dtype(source[c]) and source[c].nunique(dropna=True) <= 50
        ]

    ks_scores: List[float] = []
    tv_scores: List[float] = []
    missing_scores: List[float] = []
    per_column: List[Dict[str, Any]] = []
    ks_methods: List[str] = []

    for col in shared:
        entry: Dict[str, Any] = {"column": col}
        entry["missing_similarity"] = round(_missing_similarity(source[col], synthetic[col]), 6)
        missing_scores.append(entry["missing_similarity"])

        if col in numeric_cols:
            ks, ks_meta = _ks_similarity(source[col], synthetic[col])
            if ks is not None:
                entry["ks_similarity"] = round(ks, 6)
                entry["ks_method"] = ks_meta.get("method")
                entry["ks_sampled_rows"] = ks_meta.get("sampled_rows")
                ks_scores.append(ks)
                if ks_meta.get("method"):
                    ks_methods.append(str(ks_meta["method"]))
        if col in categorical_cols:
            tv = _tv_similarity(source[col], synthetic[col])
            if tv is not None:
                entry["tv_similarity"] = round(tv, 6)
                tv_scores.append(tv)
        per_column.append(entry)

    corr_similarity = _correlation_matrix_similarity(source, synthetic, numeric_cols)

    return {
        "columns": per_column,
        "ks_similarity": {
            "metric": "ks_similarity",
            "score": round(float(np.mean(ks_scores)), 6) if ks_scores else None,
            "target": 0.85,
            "passed": (float(np.mean(ks_scores)) >= 0.85) if ks_scores else None,
            "column_count": len(ks_scores),
            "method": ks_methods[0] if ks_methods else None,
            "sampled_rows": max(
                (entry.get("ks_sampled_rows") or 0 for entry in per_column if entry.get("ks_sampled_rows")),
                default=0,
            ),
            "fidelity_mode": "sampled" if any((entry.get("ks_sampled_rows") or 0) > 5000 for entry in per_column) else "exact",
        },
        "tv_similarity": {
            "metric": "tv_similarity",
            "score": round(float(np.mean(tv_scores)), 6) if tv_scores else None,
            "target": 0.85,
            "passed": (float(np.mean(tv_scores)) >= 0.85) if tv_scores else None,
            "column_count": len(tv_scores),
        },
        "missing_similarity": {
            "metric": "missing_similarity",
            "score": round(float(np.mean(missing_scores)), 6) if missing_scores else 1.0,
            "target": 0.90,
            "passed": (float(np.mean(missing_scores)) >= 0.90) if missing_scores else True,
            "column_count": len(missing_scores),
        },
        "correlation_similarity": {
            "metric": "correlation_similarity",
            "score": round(corr_similarity, 6) if corr_similarity is not None else None,
            "target": 0.80,
            "passed": (corr_similarity >= 0.80) if corr_similarity is not None else None,
        },
    }


def compare_schema_null_rates(
    tables: Dict[str, pd.DataFrame],
    schema: SchemaConfig,
) -> Dict[str, Any]:
    """Schema-driven missingness proxy using declared null_probability."""
    scores: List[float] = []
    per_column: List[Dict[str, Any]] = []
    for table in schema.tables:
        df = tables.get(table.name)
        if df is None:
            continue
        for col in schema.get_columns(table.name):
            if col.name not in df.columns:
                continue
            actual_null = float(df[col.name].isna().mean())
            expected_null = float((col.distribution_params or {}).get("null_probability", 0.0))
            if not col.nullable and expected_null == 0.0:
                score = 1.0 if actual_null == 0.0 else max(0.0, 1.0 - actual_null)
            else:
                score = max(0.0, 1.0 - abs(actual_null - expected_null))
            scores.append(score)
            per_column.append(
                {
                    "table": table.name,
                    "column": col.name,
                    "expected_null_rate": expected_null,
                    "actual_null_rate": round(actual_null, 6),
                    "missing_similarity": round(score, 6),
                }
            )
    avg = float(np.mean(scores)) if scores else 1.0
    return {
        "metric": "missing_similarity",
        "score": round(avg, 6),
        "target": 0.90,
        "passed": avg >= 0.90,
        "columns": per_column,
    }
