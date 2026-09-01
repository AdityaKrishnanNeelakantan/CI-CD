"""Compile declared schema and profile evidence into executable constraints."""
from __future__ import annotations

from synth_platform.domain.constraints.models import (
    ConstraintDefinition, ConstraintKind, ConstraintSet,
)
from synth_platform.domain.profiling.models import DataProfile
from synth_platform.domain.schema.models import DatabaseSchema


def compile_constraints(
    schema: DatabaseSchema,
    profile: DataProfile,
    confirmed_rules: list[ConstraintDefinition] | None = None,
) -> ConstraintSet:
    out: list[ConstraintDefinition] = []
    for table_name, table in schema.tables.items():
        if table.primary_key:
            out.append(ConstraintDefinition(
                kind=ConstraintKind.UNIQUE, table=table_name,
                columns=[table.primary_key], source="declared_primary_key"))
        if table.primary_key_columns:
            out.append(ConstraintDefinition(
                kind=ConstraintKind.UNIQUE, table=table_name,
                columns=list(table.primary_key_columns), source="declared_primary_key"))
        for unique in table.unique_constraints:
            out.append(ConstraintDefinition(
                kind=ConstraintKind.UNIQUE, table=table_name,
                columns=list(unique.columns), source="declared_unique"))
        names = set(table.column_names)
        starts = [name for name in names if "start" in name.lower() or "begin" in name.lower()]
        ends = [name for name in names if "end" in name.lower() or "finish" in name.lower()]
        if starts and ends:
            out.append(ConstraintDefinition(
                kind=ConstraintKind.TEMPORAL, table=table_name,
                earlier_column=sorted(starts)[0], later_column=sorted(ends)[0],
                inferred=True, confidence=0.85, source="name_based_temporal_order"))
        tprof = profile.tables.get(table_name)
        if not tprof:
            continue
        structural_columns = set(table.primary_key_columns)
        if table.primary_key:
            structural_columns.add(table.primary_key)
        structural_columns.update(
            fk.child_column for fk in schema.foreign_keys if fk.child_table == table_name
        )
        for col_name, col in tprof.columns.items():
            if col_name in structural_columns:
                continue
            if col.numeric is not None and col.physical_type != "datetime":
                out.append(ConstraintDefinition(
                    kind=ConstraintKind.RANGE, table=table_name, column=col_name,
                    minimum=col.numeric.minimum, maximum=col.numeric.maximum,
                    inferred=True, confidence=0.95, source="profile_range"))
            if col.categorical is not None and col.categorical.values:
                out.append(ConstraintDefinition(
                    kind=ConstraintKind.ENUM, table=table_name, column=col_name,
                    allowed_values=list(col.categorical.values), inferred=True,
                    confidence=0.95, source="profile_enum"))
    for edge_key, model in profile.cardinality_models.items():
        child, parent = edge_key.split("->", 1)
        if model.maximum is not None:
            out.append(ConstraintDefinition(
                kind=ConstraintKind.MAX_CHILDREN, table=child,
                limit=model.maximum, expression=parent,
                inferred=True, confidence=0.9, source="observed_cardinality"))
    out.extend(confirmed_rules or [])
    return ConstraintSet(constraints=out)
