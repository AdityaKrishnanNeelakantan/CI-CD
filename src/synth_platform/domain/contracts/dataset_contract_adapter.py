"""Adapter from legacy approved dataset_contract.json to CanonicalContract."""
from __future__ import annotations

from typing import Any

from synth_platform.domain.contracts.models import (
    CanonicalContract,
    CanonicalEntity,
    CanonicalField,
    CanonicalProvenance,
    CanonicalRelationship,
    SourceDescriptor,
)


def dataset_contract_to_canonical_contract(
    contract: dict[str, Any],
    *,
    contract_id: str = "",
    produced_by: str = "dataset_contract_to_canonical_contract",
) -> CanonicalContract:
    source_fingerprint = str(contract.get("source_fingerprint") or "")
    provenance = CanonicalProvenance(
        produced_by=produced_by,
        source=SourceDescriptor(
            source_type="database",
            source_id=str(contract.get("dataset_id") or ""),
            source_fingerprint=source_fingerprint,
            format_version=str(contract.get("contract_version") or ""),
        ),
    )
    entities: list[CanonicalEntity] = []
    fields: list[CanonicalField] = []
    relationships: list[CanonicalRelationship] = []

    for table_name in sorted((contract.get("tables") or {})):
        table = contract["tables"][table_name]
        entity_id = f"table:{table_name}"
        field_ids: list[str] = []
        primary_keys = set(_as_list(table.get("primary_key")))
        for column_name in sorted((table.get("columns") or {})):
            column = table["columns"][column_name]
            field_id = f"field:{table_name}.{column_name}"
            field_ids.append(field_id)
            semantic_type = column.get("semantic_type")
            fields.append(
                CanonicalField(
                    field_id=field_id,
                    name=column_name,
                    entity_id=entity_id,
                    physical_type=str(column.get("physical_type") or "unknown"),
                    semantic_type=str(semantic_type) if semantic_type is not None else None,
                    nullable=bool(column.get("nullable", True)),
                    primary_key=column_name in primary_keys,
                    unique=column_name in primary_keys,
                    constraints={
                        "sensitive": bool(column.get("sensitive", False)),
                        "inference_status": column.get("inference_status"),
                        "confidence": column.get("confidence"),
                        "alternatives": column.get("alternatives") or [],
                    },
                    provenance=CanonicalProvenance(
                        produced_by=produced_by,
                        source=provenance.source,
                        evidence_refs=[str(item) for item in (column.get("evidence") or [])],
                    ),
                )
            )
        entities.append(CanonicalEntity(entity_id=entity_id, name=table_name, fields=field_ids))
        relationships.extend(_relationships_from_table(table_name, table, produced_by))

    return CanonicalContract(
        contract_id=contract_id or str(contract.get("dataset_id") or ""),
        source_type="database",
        entities=entities,
        fields=fields,
        relationships=relationships,
        provenance=provenance,
    )


def _relationships_from_table(
    table_name: str,
    table: dict[str, Any],
    produced_by: str,
) -> list[CanonicalRelationship]:
    relationships: list[CanonicalRelationship] = []
    for index, raw_fk in enumerate(table.get("foreign_keys") or []):
        if not isinstance(raw_fk, dict):
            continue
        child_table = str(raw_fk.get("child_table") or table_name)
        parent_table = str(raw_fk.get("parent_table") or raw_fk.get("table") or "")
        child_columns = _as_list(raw_fk.get("child_columns") or raw_fk.get("child_column") or raw_fk.get("column"))
        parent_columns = _as_list(
            raw_fk.get("parent_columns") or raw_fk.get("parent_column") or raw_fk.get("references_column")
        )
        if not parent_table or not child_columns or not parent_columns:
            continue
        relationships.append(
            CanonicalRelationship(
                relationship_id=f"fk:{child_table}->{parent_table}:{index}",
                relationship_type="foreign_key",
                from_entity_id=f"table:{child_table}",
                to_entity_id=f"table:{parent_table}",
                from_field_ids=[f"field:{child_table}.{column}" for column in child_columns],
                to_field_ids=[f"field:{parent_table}.{column}" for column in parent_columns],
                cardinality="many_to_one",
                confidence=1.0,
                provenance=CanonicalProvenance(produced_by=produced_by),
            )
        )
    return relationships


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
