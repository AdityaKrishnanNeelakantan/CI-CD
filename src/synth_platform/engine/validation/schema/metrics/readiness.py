"""Final readiness scoring across hard checks and review metrics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from synth_platform.engine.validation.schema.metrics.business import compute_business_fidelity
from synth_platform.engine.validation.schema.metrics.distributions import compare_schema_null_rates, compare_source_synthetic_distributions
from synth_platform.engine.validation.schema.metrics.privacy import compute_privacy_metrics
from synth_platform.engine.validation.schema.metrics.rules import compute_rule_pass_rate
from synth_platform.engine.validation.schema.metrics.schema_match import compare_schema_to_tables, compare_source_profile_schema
from synth_platform.engine.validation.schema.metrics.type_adherence import compute_type_adherence
from synth_platform.engine.inference.schema.schema import SchemaConfig


class FinalReadiness(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


METRIC_TARGETS = {
    "schema_match": 1.0,
    "type_adherence": 0.999,
    "ks_similarity": 0.85,
    "tv_similarity": 0.85,
    "missing_similarity": 0.90,
    "correlation_similarity": 0.80,
    "business_fidelity": 0.80,
    "pii_replay_count": 0,
    "exact_row_match": 0.0,
    "rule_pass_rate": 1.0,
}


def compute_final_readiness(
    metrics: Dict[str, Any],
    *,
    hard_validation_passed: bool = True,
    export_ready: bool = True,
) -> Dict[str, Any]:
    """Combine metric results into PASS / REVIEW / FAIL."""
    hard_fail_reasons: List[str] = []
    review_reasons: List[str] = []

    if not hard_validation_passed:
        hard_fail_reasons.append("hard validation failed")
    if not export_ready:
        hard_fail_reasons.append("export generation failed")

    schema = metrics.get("schema_match") or {}
    if schema and not schema.get("passed", True):
        hard_fail_reasons.append("schema mismatch")

    type_metric = metrics.get("type_adherence") or {}
    if type_metric and not type_metric.get("passed", True):
        hard_fail_reasons.append("type adherence below target")

    privacy = metrics.get("privacy") or {}
    replay = privacy.get("pii_replay_count") or {}
    if replay and not replay.get("passed", True):
        hard_fail_reasons.append("PII replay detected")

    exact = privacy.get("exact_row_match") or {}
    if exact and not exact.get("passed", True):
        review_reasons.append("exact row overlap with source")

    for key in ("ks_similarity", "tv_similarity", "missing_similarity", "correlation_similarity", "business_fidelity"):
        item = metrics.get(key) or {}
        if item.get("score") is None:
            continue
        if not item.get("passed", True):
            review_reasons.append(f"{key} below target")

    rules = metrics.get("rule_pass_rate") or {}
    if rules.get("total_rules", 0) > 0 and not rules.get("passed", True):
        review_reasons.append("distribution rules not fully met")

    if hard_fail_reasons:
        status = FinalReadiness.FAIL
    elif review_reasons:
        status = FinalReadiness.REVIEW
    else:
        status = FinalReadiness.PASS

    return {
        "metric": "final_readiness",
        "status": status.value,
        "passed": status == FinalReadiness.PASS,
        "hard_fail_reasons": hard_fail_reasons,
        "review_reasons": review_reasons,
    }


def build_metrics_report(
    synthetic_tables: Dict[str, pd.DataFrame],
    schema: SchemaConfig,
    *,
    source_tables: Optional[Dict[str, pd.DataFrame]] = None,
    source_profile: Any = None,
    rule_evidence: Optional[Sequence[Dict[str, Any]]] = None,
    hard_validation_passed: bool = True,
    export_ready: bool = True,
    generation_mode: str = "schema_driven",
) -> Dict[str, Any]:
    """Build unified metrics report for schema-driven or source-driven generation."""
    metrics: Dict[str, Any] = {
        "generation_mode": generation_mode,
        "targets": METRIC_TARGETS,
    }

    metrics["schema_match"] = compare_schema_to_tables(schema, synthetic_tables)
    metrics["type_adherence"] = compute_type_adherence(synthetic_tables, schema)
    metrics["rule_pass_rate"] = compute_rule_pass_rate(rule_evidence or [])

    if source_tables:
        # Use first shared table for source-vs-synthetic fidelity when single-table
        source_name = next(iter(source_tables))
        syn_name = source_name if source_name in synthetic_tables else next(iter(synthetic_tables))
        source_df = source_tables[source_name]
        synthetic_df = synthetic_tables[syn_name]

        metrics["schema_match"] = compare_source_profile_schema(
            list(source_df.columns),
            list(synthetic_df.columns),
        )
        dist = compare_source_synthetic_distributions(source_df, synthetic_df)
        metrics["ks_similarity"] = dist["ks_similarity"]
        metrics["tv_similarity"] = dist["tv_similarity"]
        metrics["missing_similarity"] = dist["missing_similarity"]
        metrics["correlation_similarity"] = dist["correlation_similarity"]
        metrics["distribution_columns"] = dist["columns"]
        metrics["business_fidelity"] = compute_business_fidelity(source_df, synthetic_df)

        fresh_cols = []
        if source_profile is not None:
            try:
                from synth_platform.engine.inference.schema.source_driven import pii_replay_column_names

                fresh_cols = pii_replay_column_names(source_profile)
            except Exception:
                fresh_cols = []
        if not fresh_cols:
            fresh_cols = [c for c in source_df.columns if c in synthetic_df.columns][:10]
        metrics["privacy"] = compute_privacy_metrics(source_df, synthetic_df, fresh_columns=fresh_cols)
    else:
        metrics["missing_similarity"] = compare_schema_null_rates(synthetic_tables, schema)
        metrics["privacy"] = {
            "pii_replay_count": {
                "metric": "pii_replay_count",
                "score": 0,
                "target": 0,
                "passed": True,
                "note": "No source table provided; replay check skipped.",
            },
            "exact_row_match": {
                "metric": "exact_row_match",
                "score": 0.0,
                "rate": 0.0,
                "target": 0.0,
                "passed": True,
                "note": "No source table provided; exact-row check skipped.",
            },
        }

    metrics["final_readiness"] = compute_final_readiness(
        metrics,
        hard_validation_passed=hard_validation_passed,
        export_ready=export_ready,
    )
    return metrics
