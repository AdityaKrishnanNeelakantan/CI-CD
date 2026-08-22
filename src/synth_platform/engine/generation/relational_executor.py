"""Source-free relational executor using compiled cardinality and constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd

from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.domain.constraints.enforcement import max_children_for
from synth_platform.domain.constraints.models import ConstraintKind
from synth_platform.engine.generation.column_generator import ColumnContext, sample_column
from synth_platform.engine.generation.constraint_executor import apply_row_constraints


def _sample_counts(model, parent_frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    counts = np.zeros(len(parent_frame), dtype=int)
    base_p = np.asarray(model.probabilities, dtype=float)
    base_p = base_p / base_p.sum() if base_p.sum() else np.ones(len(model.values)) / len(model.values)
    for i in range(len(parent_frame)):
        values, probs = model.values, base_p
        if model.parent_context_column and model.parent_context_column in parent_frame.columns:
            state = str(parent_frame.iloc[i][model.parent_context_column])
            if state in model.conditional_values:
                values = model.conditional_values[state]
                probs = np.asarray(model.conditional_probabilities[state], dtype=float)
                probs = probs / probs.sum() if probs.sum() else np.ones(len(values)) / len(values)
        counts[i] = int(rng.choice(values, p=probs))
    return counts


def _planned_size_and_fk(
    artifact: SynthArtifact,
    table_name: str,
    root_rows: dict[str, int],
    scale: float,
    tables: dict[str, pd.DataFrame],
    rng: np.random.Generator,
):
    tschema = artifact.schema_.tables[table_name]
    inbound = artifact.relational_plan.edges_into(table_name)
    if table_name in root_rows:
        return max(0, int(root_rows[table_name])), None, None
    if not inbound:
        return max(0, int(round(tschema.row_count * max(scale, 0.0)))), None, None

    primary_edge = inbound[0]
    parent = tables.get(primary_edge.parent_table)
    if parent is None or parent.empty:
        return 0, primary_edge, np.array([], dtype=object)
    key = f"{table_name}->{primary_edge.parent_table}"
    model = artifact.relational_plan.cardinality_models.get(key)
    if model is not None:
        counts = _sample_counts(model, parent, rng)
    else:
        ratio = artifact.relational_plan.child_per_parent.get(key, 0.0)
        counts = np.full(len(parent), max(0, int(round(ratio))), dtype=int)
    limit = max_children_for(artifact.constraints, table_name)
    if limit is not None:
        counts = np.minimum(counts, limit)
    parent_keys = parent[primary_edge.parent_column].to_numpy()
    fk_values = np.repeat(parent_keys, counts)
    return len(fk_values), primary_edge, fk_values


def _apply_column_constraints(artifact, table_name: str, cols: dict[str, np.ndarray]):
    for rule in artifact.constraints.constraints:
        if rule.table != table_name:
            continue
        if rule.kind == ConstraintKind.RANGE and rule.column in cols:
            arr = pd.to_numeric(pd.Series(cols[rule.column]), errors="coerce").to_numpy(float)
            if rule.minimum is not None:
                arr = np.maximum(arr, rule.minimum)
            if rule.maximum is not None:
                arr = np.minimum(arr, rule.maximum)
            cols[rule.column] = arr
        elif rule.kind == ConstraintKind.ENUM and rule.column in cols and rule.allowed_values:
            allowed = set(str(v) for v in rule.allowed_values)
            vals = cols[rule.column].astype(object)
            fallback = rule.allowed_values[0]
            cols[rule.column] = np.asarray([
                value if str(value) in allowed else fallback for value in vals
            ], dtype=object)
        elif rule.kind == ConstraintKind.UNIQUE:
            columns = rule.columns or ([rule.column] if rule.column else [])
            if len(columns) == 1 and columns[0] in cols:
                col = columns[0]
                if len(set(map(str, cols[col]))) != len(cols[col]):
                    cols[col] = np.arange(1, len(cols[col]) + 1)


def execute(artifact: SynthArtifact, root_rows: dict[str, int], scale: float,
            seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    schema = artifact.schema_
    profile = artifact.data_profile
    rel = artifact.relational_plan
    policy = {(c.table, c.column): c.action.value for c in artifact.privacy_policy.columns}
    tables: dict[str, pd.DataFrame] = {}

    for tname in rel.order:
        tschema = schema.tables[tname]
        tprof = profile.tables[tname]
        n, primary_edge, primary_fk_values = _planned_size_and_fk(
            artifact, tname, root_rows, scale, tables, rng)
        pk = tschema.primary_key
        fk_cols = {e.child_column: e for e in rel.edges_into(tname)}
        cols: dict[str, np.ndarray] = {}

        for col in tschema.column_names:
            if col == pk:
                cols[col] = np.arange(1, n + 1)
            elif col in fk_cols:
                edge = fk_cols[col]
                if primary_edge is not None and edge == primary_edge and primary_fk_values is not None:
                    cols[col] = primary_fk_values
                else:
                    parent = tables.get(edge.parent_table)
                    if parent is None or parent.empty:
                        cols[col] = np.array([None] * n, dtype=object)
                    else:
                        pool = parent[edge.parent_column].to_numpy()
                        cols[col] = rng.choice(pool, size=n) if n else np.array([], dtype=pool.dtype)

        cross_models = {
            model.child_attribute: model
            for model in rel.cross_table_conditionals
            if model.child_table == tname
        }
        for child_attribute, cross in cross_models.items():
            parent = tables.get(cross.parent_table)
            fk_values = cols.get(cross.child_fk_column)
            if parent is None or fk_values is None or cross.parent_attribute not in parent:
                continue
            mapping = dict(zip(parent[cross.parent_key_column], parent[cross.parent_attribute]))
            cols[cross.context_name] = np.asarray(
                [mapping.get(value) for value in fk_values], dtype=object)

        structural = {c for c in cols if c in tschema.column_names}
        attribute_cols = [c for c in tschema.column_names if c not in structural]
        for col in tprof.dependencies.topological_order(attribute_cols):
            cprof = tprof.columns.get(col)
            if cprof is None:
                cols[col] = np.array([None] * n, dtype=object)
                continue
            vals = sample_column(ColumnContext(
                n=n, rng=rng, profile=cprof,
                conditional=(cross_models[col].model if col in cross_models
                             else tprof.conditionals.get(col)), generated=cols,
                action=policy.get((tname, col), "preserve")))
            if cprof.physical_type == "datetime":
                converted = pd.to_datetime(vals, unit="s", errors="coerce", utc=True)
                if (cprof.semantic_type or "").lower() == "date":
                    vals = np.asarray([x.date().isoformat() if not pd.isna(x) else None
                                       for x in converted], dtype=object)
                else:
                    vals = np.asarray([x.isoformat() if not pd.isna(x) else None
                                       for x in converted], dtype=object)
            schema_column = next(
                (column for column in tschema.columns if column.name == col), None)
            logical_type = schema_column.logical_type if schema_column is not None else None
            schema_physical = (schema_column.physical_type.lower()
                               if schema_column is not None else "")
            if logical_type == "integer" or "int" in schema_physical:
                numeric = pd.to_numeric(pd.Series(vals), errors="coerce")
                vals = np.asarray([int(round(value)) if not pd.isna(value) else None
                                   for value in numeric], dtype=object)
            elif logical_type == "boolean":
                vals = np.asarray([
                    value if isinstance(value, (bool, np.bool_))
                    else str(value).strip().lower() in {"true", "1", "yes", "y", "x", "✓", "✔"}
                    if value is not None else None
                    for value in vals
                ], dtype=object)
            if n and cprof.missing_rate > 0 and cprof.physical_type != "identifier":
                mask = rng.random(n) < cprof.missing_rate
                out = pd.array(vals, dtype=object)
                out[mask] = None
                vals = np.asarray(out, dtype=object)
            cols[col] = vals
        _apply_column_constraints(artifact, tname, cols)
        frame = pd.DataFrame(cols, columns=list(tschema.column_names))
        protected = {edge.child_column for edge in rel.edges_into(tname)}
        tables[tname] = apply_row_constraints(
            tname, frame, artifact.constraints, protected_columns=protected)
    return tables
