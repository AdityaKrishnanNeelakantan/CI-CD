"""Build shared entity graph views from canonical contracts."""
from __future__ import annotations

from synth_platform.domain.contracts.models import CanonicalContract
from synth_platform.domain.entities.graph import EntityGraph, EntityNode, EntityRelationship


def entity_graph_from_contract(contract: CanonicalContract) -> EntityGraph:
    return EntityGraph(
        nodes=[
            EntityNode(
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
                label=entity.name,
                field_ids=list(entity.fields),
                metadata={key: str(value) for key, value in entity.metadata.items() if value is not None},
            )
            for entity in contract.entities
        ],
        relationships=[
            EntityRelationship(
                relationship_id=rel.relationship_id,
                relationship_type=rel.relationship_type,
                from_entity_id=rel.from_entity_id,
                to_entity_id=rel.to_entity_id,
                from_field_ids=list(rel.from_field_ids),
                to_field_ids=list(rel.to_field_ids),
            )
            for rel in contract.relationships
        ],
    )
