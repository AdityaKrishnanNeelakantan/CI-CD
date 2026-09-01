"""Safe, declarative business-rule evaluation over generated rows -
derived-field formulas (recalculate) and hard constraints (reject-row on
violation), the Constraints stage from this project's own reference
design: "models propose candidate values, constraints decide whether
they are allowed."

Deliberately not Python eval()/exec(): a business rule is user-approved
configuration, not trusted code, so the expression parser (stdlib ast)
only ever accepts a narrow whitelist of nodes - column-name lookups,
numeric/string/bool literals, arithmetic, comparisons, and boolean
combinators. Anything else (function calls, attribute access, imports,
comprehensions, subscripts, ...) is rejected before it is ever evaluated.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

import pandas as pd

REPAIR_REJECT_ROW = "reject_row"

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_COMPARE = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_ALLOWED_UNARY: dict[type, Callable[[Any], Any]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}


class UnsafeExpressionError(Exception):
    """Raised when a rule expression uses anything outside the safe, whitelisted grammar."""


def _validate_node_types(tree: ast.AST, expression: str) -> None:
    # ast.walk() yields every node, including the operator-marker nodes
    # themselves (Sub, Gt, And, ...) as separate nodes from the BinOp/
    # Compare/BoolOp that hold them - an unsupported operator is already
    # rejected at that parent-node check below, so the bare marker itself
    # (inert on its own - it carries no behaviour) is always safe to allow.
    for node in ast.walk(tree):
        if isinstance(node, (ast.operator, ast.cmpop, ast.unaryop, ast.boolop)):
            continue
        if isinstance(node, (ast.Expression, ast.Load, ast.Name)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            continue
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            continue
        if isinstance(node, ast.Compare) and all(type(op) in _ALLOWED_COMPARE for op in node.ops):
            continue
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            continue
        raise UnsafeExpressionError(
            f"expression {expression!r} contains unsupported syntax: {type(node).__name__}"
        )


def _parse(expression: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"invalid expression syntax: {expression!r}") from exc
    _validate_node_types(tree, expression)
    return tree


def _evaluate_node(node: ast.AST, row: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, row)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in row:
            raise UnsafeExpressionError(f"unknown column {node.id!r} in expression")
        return row[node.id]
    if isinstance(node, ast.BinOp):
        binop = _ALLOWED_BINOPS[type(node.op)]
        return binop(_evaluate_node(node.left, row), _evaluate_node(node.right, row))
    if isinstance(node, ast.UnaryOp):
        unaryop = _ALLOWED_UNARY[type(node.op)]
        return unaryop(_evaluate_node(node.operand, row))
    if isinstance(node, ast.Compare):
        left = _evaluate_node(node.left, row)
        result = True
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _evaluate_node(comparator, row)
            result = result and _ALLOWED_COMPARE[type(op_node)](left, right)
            left = right
        return result
    if isinstance(node, ast.BoolOp):
        values = [_evaluate_node(v, row) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    raise UnsafeExpressionError(f"cannot evaluate node type {type(node).__name__}")


def evaluate_expression(expression: str, row: dict[str, Any]) -> Any:
    """Evaluate a whitelisted arithmetic/comparison expression against one row's columns."""
    return _evaluate_node(_parse(expression), row)


def apply_derived_fields(df: pd.DataFrame, derived_fields: list[dict[str, Any]]) -> pd.DataFrame:
    """derived_fields: [{"target_column": "available_credit", "formula": "credit_limit - balance"}].

    Recalculates every derived column from the already-generated row -
    the "correct action: recalculate" repair from this project's
    reference design, applied unconditionally rather than only on
    mismatch, since a derived column has no independent generator of its
    own to disagree with in the first place.

    A row where the formula cannot be evaluated (missing column, type
    mismatch, division by zero) leaves that cell as NaN rather than
    crashing generation - matching check_constraints()'s per-row
    exception handling policy.
    """
    if not derived_fields:
        return df

    result = df.copy()
    for rule in derived_fields:
        target = rule["target_column"]
        formula = rule["formula"]
        computed: list[Any] = []
        for row in result.to_dict(orient="records"):
            try:
                computed.append(evaluate_expression(formula, row))
            except (KeyError, TypeError, ZeroDivisionError, UnsafeExpressionError):
                computed.append(pd.NA)
        result[target] = computed
    return result


def check_constraints(df: pd.DataFrame, constraints: list[dict[str, Any]]) -> dict[str, Any]:
    """constraints: [{"rule_id": ..., "expression": "end_date >= start_date"}].

    Reports which rows violate each rule; repair_violations() below
    decides what to do about it. A row where the expression cannot even
    be evaluated (missing column, type mismatch, division by zero) counts
    as a violation rather than silently passing.
    """
    violations: dict[str, list[int]] = {}
    for rule in constraints:
        rule_id = rule["rule_id"]
        expression = rule["expression"]
        violating_rows = []
        for idx, row in zip(df.index, df.to_dict(orient="records")):
            try:
                passed = bool(evaluate_expression(expression, row))
            except (KeyError, TypeError, ZeroDivisionError, UnsafeExpressionError):
                passed = False
            if not passed:
                violating_rows.append(idx)
        violations[rule_id] = violating_rows

    total_rows = len(df)
    total_checks = len(constraints) * total_rows
    total_violations = sum(len(v) for v in violations.values())

    return {
        "violations": violations,
        "total_rows": total_rows,
        "total_rules": len(constraints),
        "total_violations": total_violations,
        "compliance_rate": round(1 - (total_violations / total_checks), 6) if total_checks else 1.0,
    }


def repair_violations(
    df: pd.DataFrame, constraints: list[dict[str, Any]], violation_report: dict[str, Any]
) -> pd.DataFrame:
    """Only "reject_row" is implemented as an automatic repair here: drop
    every row that violates a rule configured with that policy.
    Resampling just the offending column in place would need to go back
    to that specific column's own generator/distribution, which a
    generic, column-agnostic constraint engine cannot do - a caller that
    needs that must regenerate the column itself and re-check.
    """
    policy_by_rule = {rule["rule_id"]: rule.get("on_violation", REPAIR_REJECT_ROW) for rule in constraints}
    rows_to_drop: set[Any] = set()
    for rule_id, violating_rows in violation_report["violations"].items():
        if policy_by_rule.get(rule_id) == REPAIR_REJECT_ROW:
            rows_to_drop.update(violating_rows)
    return df.drop(index=list(rows_to_drop)).reset_index(drop=True)
