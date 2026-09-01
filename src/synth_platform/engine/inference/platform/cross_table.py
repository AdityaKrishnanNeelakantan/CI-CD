"""Compile source-generic parent-attribute -> child-attribute conditionals."""
from __future__ import annotations

import pandas as pd

from synth_platform.domain.profiling.models import DataProfile, SensitivityClass
from synth_platform.domain.relational.models import CrossTableConditional
from synth_platform.domain.schema.models import DatabaseSchema
from synth_platform.engine.inference.platform.service import _nmi, compile_conditional


def compile_cross_table_conditionals(
    schema: DatabaseSchema,
    frames: dict[str, pd.DataFrame],
    profile: DataProfile,
    threshold: float = 0.20,
) -> list[CrossTableConditional]:
    compiled: list[CrossTableConditional] = []
    for fk in schema.foreign_keys:
        if fk.parent_table == fk.child_table:
            continue
        parent = frames[fk.parent_table]
        child = frames[fk.child_table]
        if fk.parent_column not in parent or fk.child_column not in child:
            continue
        pprof = profile.tables[fk.parent_table]
        cprof = profile.tables[fk.child_table]
        parent_candidates = [
            name for name, col in pprof.columns.items()
            if name != fk.parent_column
            and col.sensitivity == SensitivityClass.PUBLIC
            and col.physical_type in {"categorical", "numeric", "datetime"}
            and name in parent
        ]
        if not parent_candidates:
            continue
        merged = child.merge(
            parent[[fk.parent_column, *parent_candidates]],
            left_on=fk.child_column, right_on=fk.parent_column,
            how="inner", suffixes=("", "__parent"))
        if len(merged) < 8:
            continue
        for child_attribute, child_profile in cprof.columns.items():
            if child_attribute in {fk.child_column, schema.tables[fk.child_table].primary_key}:
                continue
            if child_profile.physical_type not in {"categorical", "numeric", "datetime"}:
                continue
            if child_attribute not in merged:
                continue
            best_attr, best_strength = None, threshold
            for parent_attribute in parent_candidates:
                score = _nmi(merged[parent_attribute], merged[child_attribute])
                if score > best_strength:
                    best_attr, best_strength = parent_attribute, score
            if best_attr is None:
                continue
            context_name = f"__parent__{fk.parent_table}__{best_attr}"
            temp = merged.copy()
            temp[context_name] = temp[best_attr]
            local_parents = [p for p in cprof.dependencies.parents_of(child_attribute)
                             if p in temp]
            parents = [context_name, *local_parents]
            model = compile_conditional(
                temp, child_attribute, parents,
                child_profile.physical_type in {"numeric", "datetime"})
            compiled.append(CrossTableConditional(
                parent_table=fk.parent_table,
                parent_key_column=fk.parent_column,
                parent_attribute=best_attr,
                child_table=fk.child_table,
                child_fk_column=fk.child_column,
                child_attribute=child_attribute,
                context_name=context_name,
                strength=best_strength,
                model=model))
    return compiled
