from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.validation.database.qa_report import build_qa_report

pytestmark = pytest.mark.unit


def _contract(primary_key):
    return {"tables": {"customers": {"primary_key": primary_key}}}


def _fk_validity(overall=1.0):
    return {"edges": {}, "overall_fk_validity": overall}


def test_unique_primary_key_passes_hard_check():
    df = pd.DataFrame({"customer_id": ["a", "b", "c"]})
    report = build_qa_report({"customers": df}, _contract(["customer_id"]), _fk_validity(), {})
    assert report["integrity"]["primary_key_checks"]["customers"]["is_unique"] is True
    assert report["hard_checks_passed"] is True


def test_duplicate_primary_key_fails_hard_check():
    df = pd.DataFrame({"customer_id": ["a", "a", "c"]})
    report = build_qa_report({"customers": df}, _contract(["customer_id"]), _fk_validity(), {})
    assert report["integrity"]["primary_key_checks"]["customers"]["is_unique"] is False
    assert report["integrity"]["primary_key_checks"]["customers"]["duplicate_count"] == 1
    assert report["hard_checks_passed"] is False


def test_fk_validity_below_one_fails_hard_check():
    df = pd.DataFrame({"customer_id": ["a", "b"]})
    report = build_qa_report({"customers": df}, _contract(["customer_id"]), _fk_validity(0.5), {})
    assert report["hard_checks_passed"] is False


def test_missing_primary_key_column_is_reported_but_not_checked():
    df = pd.DataFrame({"other_col": [1, 2]})
    report = build_qa_report({"customers": df}, _contract(["customer_id"]), _fk_validity(), {})
    check = report["integrity"]["primary_key_checks"]["customers"]
    assert check["checked"] is False


def test_fidelity_bounded_range_check_reports_within_bounds_rate():
    df = pd.DataFrame({"amount": [50.0, 60.0, 5000.0]})  # last value is outside the safe bound
    reference_profile = {
        "tables": {"customers": {"columns": {"amount": {"generation_lower_bound": 10.0, "generation_upper_bound": 100.0}}}}
    }
    report = build_qa_report(
        {"customers": df}, _contract([]), _fk_validity(), {}, reference_profile=reference_profile
    )
    check = report["fidelity"]["customers"]["amount"]
    assert check["check"] == "bounded_range"
    assert check["within_bounds_count"] == 2
    assert check["total_count"] == 3


def test_fidelity_category_check_reports_known_vocabulary_match_rate():
    df = pd.DataFrame({"status": ["active", "active", "__RARE__"]})
    reference_profile = {
        "tables": {"customers": {"columns": {"status": {"safe_category_frequencies": {"active": 100, "__RARE__": 3}}}}}
    }
    report = build_qa_report(
        {"customers": df}, _contract([]), _fk_validity(), {}, reference_profile=reference_profile
    )
    check = report["fidelity"]["customers"]["status"]
    assert check["check"] == "known_vocabulary"
    assert check["matched_count"] == 3


def test_fidelity_is_empty_when_no_reference_profile_supplied():
    df = pd.DataFrame({"amount": [1, 2, 3]})
    report = build_qa_report({"customers": df}, _contract([]), _fk_validity(), {}, reference_profile=None)
    assert report["fidelity"] == {}


def test_business_rules_section_passes_through_constraint_reports_unchanged():
    constraint_reports = {"orders": {"total_violations": 0, "compliance_rate": 1.0}}
    report = build_qa_report({"customers": pd.DataFrame()}, _contract([]), _fk_validity(), constraint_reports)
    assert report["business_rules"] == constraint_reports


def test_fidelity_never_reads_exact_minimum_maximum_or_category_frequencies():
    """The whole point of the sanitized-companion-fields design: even if
    the exact fields are present on the same profile object (they always
    are, in-pipeline - profiler.py keeps both), this module must never
    read them.
    """
    df = pd.DataFrame({"amount": [50.0]})
    reference_profile = {
        "tables": {
            "customers": {
                "columns": {
                    "amount": {
                        "minimum": 1.0,
                        "maximum": 999999.99,  # a real outlier - must never influence this check
                        "generation_lower_bound": 10.0,
                        "generation_upper_bound": 100.0,
                    }
                }
            }
        }
    }
    report = build_qa_report(
        {"customers": df}, _contract([]), _fk_validity(), {}, reference_profile=reference_profile
    )
    check = report["fidelity"]["customers"]["amount"]
    assert check["generation_upper_bound"] == 100.0
    assert "maximum" not in check
    assert 999999.99 not in check.values()
