"""Tests for unified generation metrics."""

from __future__ import annotations

import pandas as pd

from synth_platform.engine.validation.schema.metrics.business import compute_business_fidelity
from synth_platform.engine.validation.schema.metrics.distributions import compare_source_synthetic_distributions
from synth_platform.engine.validation.schema.metrics.privacy import compute_privacy_metrics, exact_row_match_rate
from synth_platform.engine.validation.schema.metrics.readiness import METRIC_TARGETS, build_metrics_report, compute_final_readiness
from synth_platform.engine.validation.schema.metrics.rules import compute_rule_pass_rate
from synth_platform.engine.validation.schema.metrics.schema_match import compare_schema_to_tables
from synth_platform.engine.validation.schema.metrics.type_adherence import compute_type_adherence
from synth_platform.engine.inference.schema.schema import Column, SchemaConfig, Table


def _schema() -> SchemaConfig:
    return SchemaConfig(
        name="metrics demo",
        tables=[Table(name="records", row_count=100)],
        columns={
            "records": [
                Column(name="record_id", type="int", unique=True),
                Column(name="amount", type="float"),
                Column(name="status", type="categorical", distribution_params={"choices": ["open", "closed"]}),
                Column(name="notes", type="text", distribution_params={"text_type": "notes"}),
            ]
        },
    )


def test_schema_match_and_type_adherence():
    schema = _schema()
    tables = {
        "records": pd.DataFrame(
            {
                "record_id": [1, 2, 3],
                "amount": [10.5, 20.0, 30.0],
                "status": ["open", "closed", "open"],
                "notes": ["a", "b", "c"],
            }
        )
    }
    schema_metric = compare_schema_to_tables(schema, tables)
    type_metric = compute_type_adherence(tables, schema)
    assert schema_metric["passed"] is True
    assert type_metric["score"] >= 0.999


def test_distribution_similarity_scores():
    source = pd.DataFrame(
        {
            "amount": [10, 20, 30, 40, 50, 60, 70, 80],
            "status": ["open", "closed", "open", "closed", "open", "closed", "open", "closed"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "amount": [12, 18, 31, 39, 52, 58, 72, 78],
            "status": ["open", "closed", "open", "closed", "open", "closed", "open", "closed"],
        }
    )
    dist = compare_source_synthetic_distributions(source, synthetic)
    assert dist["ks_similarity"]["score"] >= 0.85
    assert dist["tv_similarity"]["score"] >= 0.85
    assert dist["missing_similarity"]["score"] >= 0.90


def test_business_fidelity_and_privacy_metrics():
    source = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B"],
            "amount": [100, 120, 200, 220],
            "email": ["a@example.com", "b@example.com", "c@example.com", "d@example.com"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B"],
            "amount": [102, 118, 198, 225],
            "email": ["x1@example.com", "x2@example.com", "x3@example.com", "x4@example.com"],
        }
    )
    business = compute_business_fidelity(source, synthetic)
    assert business["score"] >= 0.80
    privacy = compute_privacy_metrics(source, synthetic, fresh_columns=["email"])
    assert privacy["pii_replay_count"]["passed"] is True
    exact = exact_row_match_rate(source, synthetic)
    assert exact["passed"] is True


def test_business_fidelity_skips_missing_preview_categories():
    """Small previews should not compare absent categories as zero."""
    source = pd.DataFrame(
        {
            "Country": ["US"] * 40 + ["UK"] * 30 + ["CA"] * 20 + ["DE"] * 10,
            "Balance": [1000.0] * 40 + [2000.0] * 30 + [3000.0] * 20 + [4000.0] * 10,
        }
    )
    synthetic = pd.DataFrame(
        {
            "Country": ["US"] * 50 + ["UK"] * 30 + ["CA"] * 20,
            "Balance": [1000.0] * 50 + [2000.0] * 30 + [3000.0] * 20,
        }
    )
    business = compute_business_fidelity(source, synthetic)
    assert business["passed"] is True
    assert business["score"] >= 0.75
    assert business["preview_mode"] is True


def test_rule_pass_rate_and_final_readiness():
    rules = compute_rule_pass_rate(
        [
            {"rule": "30% status=open", "passed": True},
            {"rule": "20% amount>100", "passed": False},
        ]
    )
    assert rules["score"] == 0.5
    readiness = compute_final_readiness(
        {
            "schema_match": {"passed": True},
            "type_adherence": {"passed": True},
            "privacy": {"pii_replay_count": {"passed": True}, "exact_row_match": {"passed": True}},
            "ks_similarity": {"score": 0.7, "passed": False},
            "rule_pass_rate": rules,
        },
        hard_validation_passed=True,
    )
    assert readiness["status"] == "REVIEW"


def test_build_metrics_report_schema_driven():
    schema = _schema()
    tables = {
        "records": pd.DataFrame(
            {
                "record_id": [1, 2],
                "amount": [10.0, 20.0],
                "status": ["open", "closed"],
                "notes": ["note one", "note two"],
            }
        )
    }
    report = build_metrics_report(tables, schema, rule_evidence=[{"passed": True}])
    assert report["schema_match"]["passed"] is True
    assert report["final_readiness"]["status"] in {"PASS", "REVIEW", "FAIL"}
    assert "schema_match" in METRIC_TARGETS
