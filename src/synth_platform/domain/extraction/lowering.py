"""Pure lowering: TargetSchema -> canonical DatabaseSchema (the IR type).

Deterministic and side-effect-free. Field types map to physical types; the
parent link becomes a declared ForeignKey with explicit columns (never guessed).
"""
from __future__ import annotations

from synth_platform.domain.extraction.target_schema import (
    FieldType, TargetSchema,
)
from synth_platform.domain.schema.models import (
    ColumnSchema, DatabaseSchema, ForeignKey, TableSchema,
)

_PHYSICAL = {
    FieldType.INTEGER: "integer",
    FieldType.NUMBER: "real",
    FieldType.BOOLEAN: "boolean",
    FieldType.DATE: "date",
    FieldType.IDENTIFIER: "text",
}


def _physical(ft: FieldType) -> str:
    return _PHYSICAL.get(ft, "text")


def lower_target_schema(target: TargetSchema, source_kind: str) -> DatabaseSchema:
    tables: dict[str, TableSchema] = {}
    fks: list[ForeignKey] = []
    for ent in target.entities:
        cols = [ColumnSchema(name=f.name, physical_type=_physical(f.type),
                             logical_type=f.type.value, nullable=not f.required) for f in ent.fields]
        # ensure PK column exists even if not declared among fields
        if ent.primary_key and ent.primary_key not in [c.name for c in cols]:
            cols.insert(0, ColumnSchema(name=ent.primary_key,
                                        physical_type="integer", nullable=False))
        # ensure FK column exists
        if ent.parent and ent.parent.fk_field not in [c.name for c in cols]:
            cols.append(ColumnSchema(name=ent.parent.fk_field,
                                     physical_type="integer", nullable=False))
        tables[ent.name] = TableSchema(
            name=ent.name, primary_key=ent.primary_key,
            primary_key_confirmed=ent.primary_key is not None,
            columns=cols, row_count=0)
        if ent.parent:
            parent = target.entity(ent.parent.entity)
            fks.append(ForeignKey(
                parent_table=ent.parent.entity,
                parent_column=(parent.primary_key if parent and parent.primary_key else "id"),
                child_table=ent.name, child_column=ent.parent.fk_field,
                confirmed=True, evidence="declared_target_schema"))
    return DatabaseSchema(source_kind=source_kind, tables=tables, foreign_keys=fks)
