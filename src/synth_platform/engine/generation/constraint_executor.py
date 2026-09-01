"""Generation-time enforcement for compiled row-level constraints."""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from synth_platform.domain.constraints.models import ConstraintKind, ConstraintSet

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
            ast.Div, ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Constant)


def _safe_expression(expression: str):
    tree = ast.parse(expression, mode="eval")
    if any(not isinstance(node, _ALLOWED) for node in ast.walk(tree)):
        raise ValueError(f"unsupported arithmetic expression: {expression!r}")
    return compile(tree, "<constraint>", "eval")


def apply_row_constraints(
    table_name: str,
    frame: pd.DataFrame,
    constraints: ConstraintSet,
    protected_columns: set[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    protected_columns = protected_columns or set()
    for rule in constraints.constraints:
        if rule.table != table_name:
            continue
        if rule.kind == ConstraintKind.TEMPORAL:
            if rule.earlier_column in frame and rule.later_column in frame:
                earlier = pd.to_datetime(frame[rule.earlier_column], errors="coerce", utc=True)
                later = pd.to_datetime(frame[rule.later_column], errors="coerce", utc=True)
                invalid = earlier.notna() & later.notna() & (later < earlier)
                frame.loc[invalid, rule.later_column] = frame.loc[invalid, rule.earlier_column]
        elif rule.kind == ConstraintKind.CONDITIONAL:
            when = rule.when_column
            required = rule.required_column
            if when not in frame.columns or required not in frame.columns:
                continue
            if rule.when_values:
                trigger = frame[when].astype(str).isin({str(v) for v in rule.when_values})
            else:
                trigger = frame[when].notna()
            missing = frame[required].isna() | frame[required].astype(str).str.strip().eq("")
            repair = trigger & missing
            if not repair.any():
                continue
            candidates = frame.loc[~missing, required]
            if len(candidates):
                fallback = candidates.mode(dropna=True).iloc[0]
            else:
                numeric = pd.to_numeric(frame[required], errors="coerce")
                fallback = 0 if numeric.notna().any() else "required"
            frame.loc[repair, required] = fallback
        elif rule.kind == ConstraintKind.UNIQUE:
            columns = [c for c in (rule.columns or ([rule.column] if rule.column else []))
                       if c in frame.columns]
            if len(columns) <= 1 or not frame.duplicated(columns).any():
                continue
            editable = next((c for c in reversed(columns) if c not in protected_columns), None)
            if editable is None:
                continue
            duplicate_rows = frame.index[frame.duplicated(columns, keep="first")]
            numeric = pd.to_numeric(frame[editable], errors="coerce")
            if numeric.notna().all():
                start = int(numeric.max()) + 1 if len(numeric) else 1
                frame.loc[duplicate_rows, editable] = np.arange(start, start + len(duplicate_rows))
            else:
                frame.loc[duplicate_rows, editable] = [
                    f"{frame.at[index, editable]}__{offset}"
                    for offset, index in enumerate(duplicate_rows, start=1)
                ]
        elif rule.kind == ConstraintKind.ARITHMETIC and rule.expression and "=" in rule.expression:
            lhs, rhs = [part.strip() for part in rule.expression.split("=", 1)]
            if lhs not in frame.columns:
                continue
            names = {node.id for node in ast.walk(ast.parse(rhs, mode="eval"))
                     if isinstance(node, ast.Name)}
            if not names.issubset(frame.columns):
                continue
            code = _safe_expression(rhs)
            values = {name: pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy(float)
                      for name in names}
            frame[lhs] = eval(code, {"__builtins__": {}}, values)
    return frame
