from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.generation.database.constraint_engine import (
    UnsafeExpressionError,
    apply_derived_fields,
    check_constraints,
    evaluate_expression,
    repair_violations,
)

pytestmark = pytest.mark.unit


def test_evaluate_expression_arithmetic():
    assert evaluate_expression("credit_limit - balance", {"credit_limit": 1000, "balance": 250}) == 750


def test_evaluate_expression_comparison():
    assert evaluate_expression("end_date >= start_date", {"end_date": 10, "start_date": 5}) is True
    assert evaluate_expression("end_date >= start_date", {"end_date": 3, "start_date": 5}) is False


def test_evaluate_expression_boolean_combinator():
    row = {"a": 5, "b": 10}
    assert evaluate_expression("a > 0 and b > 0", row) is True
    assert evaluate_expression("a > 100 or b > 5", row) is True


def test_unknown_column_raises():
    with pytest.raises(UnsafeExpressionError):
        evaluate_expression("missing_column + 1", {"a": 1})


@pytest.mark.parametrize(
    "malicious_expression",
    [
        "__import__('os').system('echo pwned')",
        "open('secret.txt').read()",
        "().__class__.__bases__",
        "[x for x in range(10)]",
        "a.upper()",
        "lambda: 1",
        "a if True else b",
    ],
)
def test_unsafe_expressions_are_rejected_not_executed(malicious_expression: str):
    """The whole point of this module: a business rule is configuration,
    not trusted code. None of these must ever actually run.
    """
    with pytest.raises(UnsafeExpressionError):
        evaluate_expression(malicious_expression, {"a": 1, "b": 2})


def test_apply_derived_fields_recalculates_from_row_values():
    df = pd.DataFrame({"credit_limit": [1000, 2000], "balance": [250, 500]})
    result = apply_derived_fields(df, [{"target_column": "available_credit", "formula": "credit_limit - balance"}])
    assert list(result["available_credit"]) == [750, 1500]


def test_apply_derived_fields_with_no_rules_returns_unchanged_frame():
    df = pd.DataFrame({"a": [1, 2]})
    result = apply_derived_fields(df, [])
    assert result.equals(df)


def test_check_constraints_flags_violating_rows():
    df = pd.DataFrame({"start_date": [1, 5, 10], "end_date": [5, 3, 20]})
    report = check_constraints(df, [{"rule_id": "date_order", "expression": "end_date >= start_date"}])
    assert report["violations"]["date_order"] == [1]
    assert report["total_violations"] == 1
    assert report["compliance_rate"] == round(2 / 3, 6)


def test_check_constraints_all_pass_gives_full_compliance():
    df = pd.DataFrame({"start_date": [1, 2], "end_date": [5, 6]})
    report = check_constraints(df, [{"rule_id": "date_order", "expression": "end_date >= start_date"}])
    assert report["total_violations"] == 0
    assert report["compliance_rate"] == 1.0


def test_check_constraints_division_by_zero_counts_as_violation_not_a_crash():
    df = pd.DataFrame({"numerator": [10], "denominator": [0]})
    report = check_constraints(df, [{"rule_id": "r", "expression": "numerator / denominator > 1"}])
    assert report["violations"]["r"] == [0]


def test_repair_violations_drops_rejected_rows():
    df = pd.DataFrame({"start_date": [1, 5, 10], "end_date": [5, 3, 20]})
    constraints = [{"rule_id": "date_order", "expression": "end_date >= start_date", "on_violation": "reject_row"}]
    report = check_constraints(df, constraints)
    repaired = repair_violations(df, constraints, report)
    assert len(repaired) == 2
    assert 3 not in repaired["end_date"].values


def test_repair_violations_leaves_rows_alone_when_policy_is_not_reject_row():
    df = pd.DataFrame({"start_date": [1, 5], "end_date": [5, 3]})
    constraints = [{"rule_id": "date_order", "expression": "end_date >= start_date", "on_violation": "flag_only"}]
    report = check_constraints(df, constraints)
    repaired = repair_violations(df, constraints, report)
    assert len(repaired) == 2
