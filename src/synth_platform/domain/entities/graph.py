"""Shared synthetic entity graph across database, document, and future modes."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntityNode(_Base):
    entity_id: str
    entity_type: str
    label: str
    field_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class EntityRelationship(_Base):
    relationship_id: str
    relationship_type: str
    from_entity_id: str
    to_entity_id: str
    from_field_ids: list[str] = Field(default_factory=list)
    to_field_ids: list[str] = Field(default_factory=list)


class EntityGraph(_Base):
    format_version: str = "1.0.0"
    nodes: list[EntityNode] = Field(default_factory=list)
    relationships: list[EntityRelationship] = Field(default_factory=list)

    def node(self, entity_id: str) -> EntityNode | None:
        return next((node for node in self.nodes if node.entity_id == entity_id), None)

    def outgoing(self, entity_id: str) -> list[EntityRelationship]:
        return [rel for rel in self.relationships if rel.from_entity_id == entity_id]

    def incoming(self, entity_id: str) -> list[EntityRelationship]:
        return [rel for rel in self.relationships if rel.to_entity_id == entity_id]
