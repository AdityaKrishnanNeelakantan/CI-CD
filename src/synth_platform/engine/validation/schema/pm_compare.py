"""Product-manager style comparison of source vs synthetic CSV outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from synth_platform.engine.generation.schema.pii_columns import (
    column_requires_fresh_generation,
    validate_no_pii_replay,
    validate_row_uniqueness,
)
from synth_platform.engine.inference.schema.source_driven import (
    SourceDataProfiler,
    build_fidelity_report,
    fresh_column_names,
    identifier_column_names,
)

# Demo-only columns from the optional banking example schema — not expected in source-driven output.
_BANKING_DEMO_COLUMNS = frozenset(
    {
        "routing_number",
        "ifsc_code",
        "aadhaar",
        "case_note",
        "merchant",
        "posted_at",
        "transaction_id",
        "account_id",
    }
)


@dataclass
class PMCheck:
    name: str
    passed: bool
    severity: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _schema_check(source: pd.DataFrame, synthetic: pd.DataFrame) -> PMCheck:
    source_cols = list(source.columns)
    synthetic_cols = list(synthetic.columns)
    missing = [c for c in source_cols if c not in synthetic_cols]
    extra = [c for c in synthetic_cols if c not in source_cols]
    passed = not missing and not extra
    return PMCheck(
        name="schema_parity",
        passed=passed,
        severity="blocker" if not passed else "info",
        summary="Synthetic output preserves the source column set and order."
        if passed
        else f"Schema mismatch: missing={missing or 'none'}, extra={extra or 'none'}",
        details={
            "source_columns": source_cols,
            "synthetic_columns": synthetic_cols,
            "missing_in_synthetic": missing,
            "extra_in_synthetic": extra,
            "column_order_preserved": source_cols == synthetic_cols,
        },
    )


def _row_count_check(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    expected_rows: Optional[int],
) -> PMCheck:
    target = expected_rows if expected_rows is not None else len(source)
    actual = len(synthetic)
    passed = actual == target
    return PMCheck(
        name="row_count",
        passed=passed,
        severity="blocker" if not passed else "info",
        summary=f"Synthetic row count is {actual:,} (expected {target:,}).",
        details={"expected_rows": target, "actual_rows": actual, "source_rows": len(source)},
    )


def _genericness_check(source: pd.DataFrame, synthetic: pd.DataFrame) -> PMCheck:
    extra_demo_cols = [c for c in synthetic.columns if str(c).lower() in _BANKING_DEMO_COLUMNS]
    renamed = [c for c in synthetic.columns if c not in source.columns]
    source_named = not any(str(c).startswith("col_") for c in synthetic.columns)
    passed = not extra_demo_cols and not renamed and source_named
    notes: List[str] = []
    if extra_demo_cols:
        notes.append(f"Output contains demo-only banking columns: {extra_demo_cols}")
    if renamed:
        notes.append("Synthetic file introduces columns not present in the source upload.")
    if not source_named:
        notes.append("Synthetic columns look auto-renamed instead of preserving source headers.")
    return PMCheck(
        name="genericness",
        passed=passed,
        severity="warning" if not passed else "info",
        summary="Output is source-schema driven with no demo-schema injection."
        if passed
        else "; ".join(notes),
        details={
            "preserves_source_headers": list(source.columns) == list(synthetic.columns),
            "demo_columns_detected": extra_demo_cols,
            "engine_mode": "source_driven",
        },
    )


def _privacy_check(source: pd.DataFrame, synthetic: pd.DataFrame, profile: Any) -> PMCheck:
    fresh_cols = fresh_column_names(profile)
    privacy = validate_no_pii_replay(source, synthetic, fresh_cols)
    unique_targets = fresh_column_names(profile) + identifier_column_names(profile)
    uniqueness = validate_row_uniqueness(synthetic, unique_targets)
    passed = bool(privacy["passed"] and uniqueness["passed"])
    return PMCheck(
        name="privacy_and_uniqueness",
        passed=passed,
        severity="blocker" if not passed else "info",
        summary="No PII replay and identifier columns remain unique."
        if passed
        else "Privacy or uniqueness gate failed.",
        details={
            "fresh_columns": fresh_cols,
            "privacy": privacy,
            "uniqueness": uniqueness,
        },
    )


def _null_and_empty_check(source: pd.DataFrame, synthetic: pd.DataFrame) -> PMCheck:
    empty_cols = [c for c in synthetic.columns if synthetic[c].isna().all()]
    all_null_delta = {
        c: round(float(synthetic[c].isna().mean() - source[c].isna().mean()), 4)
        for c in source.columns
        if c in synthetic.columns
    }
    large_null_swings = {
        c: delta for c, delta in all_null_delta.items() if abs(delta) > 0.25
    }
    passed = not empty_cols and len(large_null_swings) <= max(1, len(source.columns) // 4)
    return PMCheck(
        name="null_quality",
        passed=passed,
        severity="warning" if not passed else "info",
        summary="Null rates are broadly stable and no column is entirely empty."
        if passed
        else f"Null-quality issues: empty={empty_cols or 'none'}, large_swings={list(large_null_swings)}",
        details={
            "empty_columns": empty_cols,
            "null_rate_delta_by_column": all_null_delta,
            "large_null_swings": large_null_swings,
        },
    )


def _categorical_drift_check(source: pd.DataFrame, synthetic: pd.DataFrame) -> PMCheck:
    drift: List[Dict[str, Any]] = []
    for col in source.columns:
        if col not in synthetic.columns:
            continue
        if pd.api.types.is_numeric_dtype(source[col]):
            continue
        src_vals = set(source[col].dropna().astype(str).unique())
        syn_vals = set(synthetic[col].dropna().astype(str).unique())
        if not src_vals:
            continue
        novel = sorted(syn_vals - src_vals)
        if novel:
            drift.append(
                {
                    "column": col,
                    "novel_values_count": len(novel),
                    "novel_values_sample": novel[:10],
                }
            )
    passed = len(drift) <= 2
    return PMCheck(
        name="categorical_drift",
        passed=passed,
        severity="warning" if not passed else "info",
        summary="Categorical values stay within the source vocabulary."
        if passed
        else f"Novel categorical values detected in {len(drift)} column(s).",
        details={"drift": drift},
    )


def _fidelity_gate(fidelity: Mapping[str, Any]) -> PMCheck:
    passed = bool(fidelity.get("passed"))
    return PMCheck(
        name="statistical_fidelity",
        passed=passed,
        severity="warning" if not passed else "info",
        summary="Marginals and correlations are within MVP fidelity thresholds."
        if passed
        else "Statistical fidelity gate did not pass — review column deltas before production use.",
        details={
            "avg_numeric_mean_delta_pct": fidelity.get("avg_numeric_mean_delta_pct"),
            "avg_correlation_delta": fidelity.get("avg_correlation_delta"),
            "model_type": fidelity.get("model_type"),
            "columns": fidelity.get("columns"),
            "correlation_checks": fidelity.get("correlation_checks"),
        },
    )


def _readiness_check(checks: Sequence[PMCheck]) -> PMCheck:
    blockers = [c for c in checks if not c.passed and c.severity == "blocker"]
    warnings = [c for c in checks if not c.passed and c.severity == "warning"]
    passed = not blockers
    if blockers:
        summary = f"Not ready — {len(blockers)} blocker(s): " + ", ".join(c.name for c in blockers)
    elif warnings:
        summary = f"Review recommended — {len(warnings)} warning(s): " + ", ".join(c.name for c in warnings)
    else:
        summary = "Ready for PM sign-off — schema, privacy, and fidelity checks passed."
    return PMCheck(
        name="pm_readiness",
        passed=passed,
        severity="blocker" if blockers else ("warning" if warnings else "info"),
        summary=summary,
        details={
            "blockers": [c.name for c in blockers],
            "warnings": [c.name for c in warnings],
        },
    )


def compare_source_synthetic(
    source: Union[str, Path, pd.DataFrame],
    synthetic: Union[str, Path, pd.DataFrame],
    *,
    expected_rows: Optional[int] = None,
    table_name: str = "source_table",
) -> Dict[str, Any]:
    """
    Run PM-oriented checks comparing a production/source CSV to synthetic output.

    Returns a structured report suitable for JSON export or Streamlit display.
    """
    if isinstance(source, (str, Path)):
        source_df = pd.read_csv(source)
        source_path = str(source)
    else:
        source_df = source.copy()
        source_path = "<dataframe>"

    if isinstance(synthetic, (str, Path)):
        synthetic_df = pd.read_csv(synthetic)
        synthetic_path = str(synthetic)
    else:
        synthetic_df = synthetic.copy()
        synthetic_path = "<dataframe>"

    profile = SourceDataProfiler().profile(source_df, table_name=table_name)
    fidelity = build_fidelity_report(source_df, synthetic_df, profile)

    pii_columns = [
        c for c in source_df.columns if column_requires_fresh_generation(c, source_df[c])
    ]

    checks = [
        _schema_check(source_df, synthetic_df),
        _row_count_check(source_df, synthetic_df, expected_rows=expected_rows),
        _genericness_check(source_df, synthetic_df),
        _privacy_check(source_df, synthetic_df, profile),
        _null_and_empty_check(source_df, synthetic_df),
        _categorical_drift_check(source_df, synthetic_df),
        _fidelity_gate(fidelity),
    ]
    checks.append(_readiness_check(checks))

    return {
        "source_path": source_path,
        "synthetic_path": synthetic_path,
        "source_rows": len(source_df),
        "synthetic_rows": len(synthetic_df),
        "column_count": len(source_df.columns),
        "recommended_model": profile.recommended_model,
        "pii_columns_detected": pii_columns,
        "sdv_columns": fidelity.get("sdv_columns"),
        "fresh_columns": fidelity.get("fresh_columns"),
        "fidelity": fidelity,
        "checks": [check.to_dict() for check in checks],
        "pm_readiness": checks[-1].passed,
        "pm_summary": checks[-1].summary,
    }


def write_pm_compare_report(
    report: Mapping[str, Any],
    output_path: Union[str, Path],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
