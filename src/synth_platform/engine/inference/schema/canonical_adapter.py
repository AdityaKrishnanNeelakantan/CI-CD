"""Adapt Schema Mode's SchemaConfig into the canonical domain contract."""
from __future__ import annotations

from synth_platform.domain.contracts.models import (
    CanonicalContract,
    CanonicalEntity,
    CanonicalField,
    CanonicalProvenance,
    CanonicalRelationship,
    SourceDescriptor,
)
from synth_platform.engine.inference.schema.schema import SchemaConfig


def schema_config_to_canonical_contract(schema: SchemaConfig) -> CanonicalContract:
    provenance = CanonicalProvenance(
        produced_by="schema_config_to_canonical_contract",
        source=SourceDescriptor(source_type="schema", source_id=schema.name),
    )
    entities: list[CanonicalEntity] = []
    fields: list[CanonicalField] = []

    for table in sorted(schema.tables, key=lambda item: item.name):
        entity_id = f"table:{table.name}"
        field_ids: list[str] = []
        for column in sorted(schema.columns.get(table.name, []), key=lambda item: item.name):
            field_id = f"field:{table.name}.{column.name}"
            field_ids.append(field_id)
            fields.append(
                CanonicalField(
                    field_id=field_id,
                    name=column.name,
                    entity_id=entity_id,
                    physical_type=column.type,
                    logical_type=column.type,
                    semantic_type=_semantic_type_for_schema_column(column.type),
                    nullable=column.nullable,
                    primary_key=column.unique,
                    unique=column.unique,
                    constraints={
                        "distribution_params": column.distribution_params,
                        "description": column.description,
                    },
                    provenance=provenance,
                )
            )
        entities.append(
            CanonicalEntity(
                entity_id=entity_id,
                name=table.name,
                entity_type="table",
                fields=field_ids,
                metadata={
                    "row_count": table.row_count,
                    "description": table.description,
                    "is_reference": table.is_reference,
                },
            )
        )

    return CanonicalContract(
        contract_id=schema.name,
        source_type="schema",
        entities=entities,
        fields=fields,
        events=[],
        relationships=[
            CanonicalRelationship(
                relationship_id=f"fk:{rel.child_table}->{rel.parent_table}:{rel.child_key}",
                relationship_type="foreign_key",
                from_entity_id=f"table:{rel.child_table}",
                to_entity_id=f"table:{rel.parent_table}",
                from_field_ids=[f"field:{rel.child_table}.{rel.child_key}"],
                to_field_ids=[f"field:{rel.parent_table}.{rel.parent_key}"],
                cardinality="many_to_one",
                confidence=1.0,
                provenance=provenance,
            )
            for rel in schema.relationships
        ],
        generation_policy={"seed": schema.seed, "domain": schema.domain},
        provenance=provenance,
    )


def _semantic_type_for_schema_column(column_type: str) -> str:
    if column_type in {"int", "float", "decimal", "money", "currency"}:
        return "numerical"
    if column_type in {"date", "time", "datetime"}:
        return "datetime"
    if column_type in {"boolean"}:
        return "boolean"
    if column_type in {"uuid", "foreign_key", "bank_account", "account_number", "routing_number", "ssn",
                       "aadhaar", "iban", "swift_bic", "ifsc_code", "credit_card"}:
        return "identifier"
    if column_type in {"email", "phone", "url", "address"}:
        return column_type
    if column_type == "categorical":
        return "category"
    return "free_text" if column_type == "text" else "unknown"
