"""Adapters from existing schema-domain models to the canonical contract."""
from __future__ import annotations

from synth_platform.domain.contracts.models import (
    CanonicalContract,
    CanonicalEntity,
    CanonicalField,
    CanonicalProvenance,
    CanonicalRelationship,
    SourceDescriptor,
)
from synth_platform.domain.schema.models import DatabaseSchema, ForeignKey


def _table_entity_id(table_name: str) -> str:
    return f"table:{table_name}"


def _field_id(table_name: str, column_name: str) -> str:
    return f"field:{table_name}.{column_name}"


def database_schema_to_contract(
    schema: DatabaseSchema,
    *,
    contract_id: str = "",
    source_fingerprint: str = "",
    produced_by: str = "database_schema_to_contract",
) -> CanonicalContract:
    source = SourceDescriptor(
        source_type="database",
        source_fingerprint=source_fingerprint,
        format_version=schema.format_version,
    )
    provenance = CanonicalProvenance(produced_by=produced_by, source=source, warnings=list(schema.warnings))
    entities: list[CanonicalEntity] = []
    fields: list[CanonicalField] = []

    for table_name in sorted(schema.tables):
        table = schema.tables[table_name]
        pk_columns = set(table.primary_key_columns or ([table.primary_key] if table.primary_key else []))
        field_ids: list[str] = []
        for column in table.columns:
            fid = _field_id(table_name, column.name)
            field_ids.append(fid)
            fields.append(
                CanonicalField(
                    field_id=fid,
                    name=column.name,
                    entity_id=_table_entity_id(table_name),
                    physical_type=column.physical_type,
                    logical_type=column.logical_type,
                    nullable=column.nullable,
                    primary_key=column.name in pk_columns,
                    unique=column.name in pk_columns,
                    constraints={"default": column.default} if column.default is not None else {},
                    provenance=provenance,
                )
            )
        entities.append(
            CanonicalEntity(
                entity_id=_table_entity_id(table_name),
                name=table.name,
                entity_type="table",
                fields=field_ids,
                metadata={
                    "schema": table.schema_name,
                    "row_count": table.row_count,
                    "estimated_row_count": table.estimated_row_count,
                },
            )
        )

    return CanonicalContract(
        contract_id=contract_id,
        source_type="database",
        entities=entities,
        fields=fields,
        relationships=[_foreign_key_to_relationship(fk) for fk in schema.foreign_keys],
        provenance=provenance,
    )


def _foreign_key_to_relationship(fk: ForeignKey) -> CanonicalRelationship:
    child_columns = fk.child_columns or [fk.child_column]
    parent_columns = fk.parent_columns or [fk.parent_column]
    return CanonicalRelationship(
        relationship_id=f"fk:{fk.child_table}->{fk.parent_table}:{','.join(child_columns)}",
        relationship_type="foreign_key",
        from_entity_id=_table_entity_id(fk.child_table),
        to_entity_id=_table_entity_id(fk.parent_table),
        from_field_ids=[_field_id(fk.child_table, col) for col in child_columns],
        to_field_ids=[_field_id(fk.parent_table, col) for col in parent_columns],
        cardinality="many_to_one",
        confidence=1.0 if fk.confirmed else 0.5,
        provenance=CanonicalProvenance(produced_by="database_schema_to_contract", evidence_refs=[fk.evidence]),
    )
