"""Concrete validation metrics (engine). Each is a Metric; the registry drives
them. This is the ONE home for per-check thresholding (A-05).

Metrics:
  StructuralIntegrityMetric   PK uniqueness/nulls, FK orphans        (critical)
  MarginalFidelityMetric      categorical TVD, numeric quantile RMSE (critical=False)
  CorrelationPreservationMetric  learned parent->child dependency strength is
      reproduced in synthetic data — the metric W2 makes meaningful    (critical)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synth_platform.domain.validation.metric import ValidationContext
from synth_platform.domain.validation.models import Capability, CheckResult, Status
from synth_platform.engine.inference.platform.service import _nmi  # canonical NMI (no dup)


def _key_columns(artifact, tname) -> set:
    cols = set()
    ts = artifact.schema_.tables.get(tname)
    if ts and ts.primary_key:
        cols.add(ts.primary_key)
    for fk in artifact.schema_.foreign_keys:
        if fk.child_table == tname:
            cols.add(fk.child_column)
    return cols


class StructuralIntegrityMetric:
    name = "structural"
    dimension = "structural"

    def evaluate(self, ctx: ValidationContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        schema, tables = ctx.artifact.schema_, ctx.tables
        for required_table in schema.tables:
            if required_table not in tables:
                out.append(CheckResult(name=f"table_present[{required_table}]", dimension="structural",
                                       critical=True, ran=True, status=Status.FAIL,
                                       detail="required generated table is missing"))
        for tname, ts in schema.tables.items():
            if ts.primary_key and tname in tables and ts.primary_key in tables[tname].columns:
                col = tables[tname][ts.primary_key]
                bad = int(col.duplicated().sum()) + int(col.isna().sum())
                out.append(CheckResult(name=f"pk_integrity[{tname}]", dimension="structural",
                                       critical=True, ran=True,
                                       status=Status.PASS if bad == 0 else Status.FAIL,
                                       metric=float(bad), threshold=0.0,
                                       detail="duplicate/null primary keys"))
        keys = {t: set(tables[t][s.primary_key]) for t, s in schema.tables.items()
                if s.primary_key and t in tables and s.primary_key in tables[t].columns}
        for fk in schema.foreign_keys:
            if fk.child_table in tables and fk.child_column in tables[fk.child_table].columns:
                child = tables[fk.child_table][fk.child_column].dropna()
                pool = keys.get(fk.parent_table, set())
                orphans = int((~child.isin(pool)).sum()) if pool else 0
                out.append(CheckResult(name=f"fk_integrity[{fk.child_table}.{fk.child_column}]",
                                       dimension="structural", critical=True, ran=True,
                                       status=Status.PASS if orphans == 0 else Status.FAIL,
                                       metric=float(orphans), threshold=0.0, detail="orphan FK values"))
        return out


class MarginalFidelityMetric:
    name = "marginal_fidelity"
    dimension = "fidelity"

    def evaluate(self, ctx: ValidationContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        art, tables, T = ctx.artifact, ctx.tables, ctx.thresholds
        for tname, tprof in art.data_profile.tables.items():
            if tname not in tables:
                continue
            df, keys = tables[tname], _key_columns(art, tname)
            for col, cprof in tprof.columns.items():
                if col not in df.columns or col in keys:
                    continue
                if cprof.physical_type == "categorical" and cprof.categorical:
                    exp = dict(zip(cprof.categorical.values, cprof.categorical.probabilities))
                    obs = df[col].dropna().astype(str).value_counts(normalize=True).to_dict()
                    tvd = 0.5 * sum(abs(exp.get(k, 0) - obs.get(k, 0)) for k in set(exp) | set(obs))
                    out.append(CheckResult(name=f"tvd[{tname}.{col}]", dimension="fidelity",
                                           critical=False, ran=True,
                                           status=Status.PASS if tvd <= T.categorical_tvd_max else Status.WARN,
                                           metric=float(tvd), threshold=T.categorical_tvd_max,
                                           detail="categorical total variation distance"))
                elif cprof.physical_type == "numeric" and cprof.numeric:
                    gv = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
                    if gv.size == 0:
                        continue
                    grid = np.linspace(0, 1, len(cprof.numeric.quantiles))
                    rmse = float(np.sqrt(np.mean(((np.quantile(gv, grid) -
                            np.array(cprof.numeric.quantiles)) /
                            ((cprof.numeric.maximum - cprof.numeric.minimum) or 1.0)) ** 2)))
                    out.append(CheckResult(name=f"quantile_rmse[{tname}.{col}]", dimension="fidelity",
                                           critical=False, ran=True,
                                           status=Status.PASS if rmse <= T.numeric_quantile_rmse_max else Status.WARN,
                                           metric=rmse, threshold=T.numeric_quantile_rmse_max,
                                           detail="normalized numeric quantile RMSE"))
        return out


class CorrelationPreservationMetric:
    """For each learned parent->child dependency edge, the correlation strength
    (NMI) in the synthetic data must track the source strength. This is only
    meaningful because generation is conditional (W2); it is the gate's proof
    that cross-column structure survived. Critical, applicable when the source
    exercises modellable attributes."""
    name = "correlation_preservation"
    dimension = "fidelity"

    def evaluate(self, ctx: ValidationContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        art, tables, T = ctx.artifact, ctx.tables, ctx.thresholds
        for tname, tprof in art.data_profile.tables.items():
            edges = tprof.dependencies.edges
            if not edges or tname not in tables:
                continue
            df = tables[tname]
            for e in edges:
                if e.parent not in df.columns or e.child not in df.columns:
                    continue
                syn_strength = _nmi(df[e.parent], df[e.child])
                # source strength was measured at compile time (e.strength).
                # require the synthetic edge to retain most of it.
                retained = syn_strength / e.strength if e.strength > 1e-9 else 1.0
                status = (Status.PASS if retained >= T.correlation_retention_min
                          else Status.WARN if retained >= 0.5 else Status.FAIL)
                out.append(CheckResult(
                    name=f"correlation[{tname}.{e.parent}->{e.child}]",
                    dimension="fidelity", critical=True, ran=True,
                    requires=[Capability.CATEGORICAL],  # applicable when structure exists
                    metric=float(retained), threshold=T.correlation_retention_min,
                    status=status,
                    detail=f"NMI retained syn/src = {syn_strength:.2f}/{e.strength:.2f}"))
        return out

class ConstraintComplianceMetric:
    name = "constraint_compliance"
    dimension = "business"

    def evaluate(self, ctx: ValidationContext) -> list[CheckResult]:
        from synth_platform.domain.constraints.models import ConstraintKind
        out: list[CheckResult] = []
        tables = ctx.tables
        constraint_set = getattr(ctx.artifact, "constraints", None)
        for rule in (constraint_set.constraints if constraint_set is not None else []):
            frame = tables.get(rule.table)
            if frame is None:
                out.append(CheckResult(
                    name=f"constraint[{rule.kind.value}:{rule.table}]",
                    dimension="business", critical=True, ran=True,
                    status=Status.FAIL, detail="required table missing"))
                continue
            violations = 0
            if rule.kind == ConstraintKind.RANGE and rule.column in frame:
                values = pd.to_numeric(frame[rule.column], errors="coerce").dropna()
                if rule.minimum is not None:
                    violations += int((values < rule.minimum).sum())
                if rule.maximum is not None:
                    violations += int((values > rule.maximum).sum())
            elif rule.kind == ConstraintKind.ENUM and rule.column in frame:
                allowed = set(map(str, rule.allowed_values))
                violations = int((~frame[rule.column].dropna().astype(str).isin(allowed)).sum())
            elif rule.kind == ConstraintKind.UNIQUE:
                columns = [c for c in rule.columns if c in frame.columns]
                if columns:
                    violations = int(frame.duplicated(columns).sum())
            elif rule.kind == ConstraintKind.MAX_CHILDREN:
                parent_name = rule.expression
                edge = next((e for e in ctx.artifact.relational_plan.edges
                             if e.child_table == rule.table and e.parent_table == parent_name), None)
                if edge and edge.child_column in frame and rule.limit is not None:
                    counts = frame.groupby(edge.child_column, dropna=True).size()
                    violations = int((counts > rule.limit).sum())
            elif rule.kind == ConstraintKind.TEMPORAL:
                if rule.earlier_column in frame and rule.later_column in frame:
                    earlier = pd.to_datetime(frame[rule.earlier_column], errors="coerce", utc=True)
                    later = pd.to_datetime(frame[rule.later_column], errors="coerce", utc=True)
                    violations = int((earlier > later).sum())
            elif rule.kind == ConstraintKind.CONDITIONAL:
                when = rule.when_column
                required = rule.required_column
                if when in frame and required in frame:
                    if rule.when_values:
                        trigger = frame[when].astype(str).isin({str(v) for v in rule.when_values})
                    else:
                        trigger = frame[when].notna()
                    missing = frame[required].isna() | frame[required].astype(str).str.strip().eq("")
                    violations = int((trigger & missing).sum())
                else:
                    violations = len(frame)
            elif rule.kind == ConstraintKind.ARITHMETIC and rule.expression and "=" in rule.expression:
                import ast
                lhs, rhs = [part.strip() for part in rule.expression.split("=", 1)]
                names = {node.id for node in ast.walk(ast.parse(rhs, mode="eval"))
                         if isinstance(node, ast.Name)}
                allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
                           ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Name,
                           ast.Load, ast.Constant)
                tree = ast.parse(rhs, mode="eval")
                if lhs in frame and names.issubset(frame.columns) and all(
                        isinstance(node, allowed) for node in ast.walk(tree)):
                    env = {name: pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy(float)
                           for name in names}
                    expected = eval(compile(tree, "<validation>", "eval"),
                                    {"__builtins__": {}}, env)
                    actual = pd.to_numeric(frame[lhs], errors="coerce").fillna(0).to_numpy(float)
                    violations = int((np.abs(actual - expected) > rule.tolerance).sum())
                else:
                    violations = len(frame)
            else:
                continue
            out.append(CheckResult(
                name=f"constraint[{rule.kind.value}:{rule.table}:{rule.column or ','.join(rule.columns)}]",
                dimension="business", critical=True, ran=True,
                status=Status.PASS if violations == 0 else Status.FAIL,
                metric=float(violations), threshold=0.0,
                detail=rule.source))
        return out
