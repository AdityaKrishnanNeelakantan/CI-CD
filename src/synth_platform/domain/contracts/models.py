"""Canonical platform contract models.

These models describe the shared synthetic truth across schema, database,
document, transcript, and future event representations. They are intentionally
compact and pure so engines can adapt existing modality-specific contracts into
one stable handoff without rewriting working pipelines all at once.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from synth_platform.domain.contracts.versioning import CANONICAL_CONTRACT_VERSION


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


SourceType = Literal["schema", "database", "document", "transcript", "event", "api", "file"]


class SourceDescriptor(_Base):
    source_type: SourceType
    source_id: str = ""
    source_fingerprint: str = ""
    format_version: str = ""


class CanonicalProvenance(_Base):
    produced_by: str = ""
    source: SourceDescriptor | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CanonicalField(_Base):
    field_id: str
    name: str
    entity_id: str | None = None
    physical_type: str = "unknown"
    logical_type: str | None = None
    semantic_type: str | None = None
    context_role: str | None = None
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    constraints: dict[str, Any] = Field(default_factory=dict)
    provenance: CanonicalProvenance | None = None


class CanonicalEntity(_Base):
    entity_id: str
    name: str
    entity_type: str = "table"
    fields: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalEvent(_Base):
    event_id: str
    name: str
    entity_ids: list[str] = Field(default_factory=list)
    timestamp_field_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalRelationship(_Base):
    relationship_id: str
    relationship_type: str
    from_entity_id: str
    to_entity_id: str
    from_field_ids: list[str] = Field(default_factory=list)
    to_field_ids: list[str] = Field(default_factory=list)
    cardinality: str | None = None
    confidence: float | None = None
    provenance: CanonicalProvenance | None = None


class DependencyEdge(_Base):
    dependency_id: str
    dependency_type: str
    source_field_ids: list[str] = Field(default_factory=list)
    target_field_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    provenance: CanonicalProvenance | None = None


class CanonicalContract(_Base):
    contract_version: str = CANONICAL_CONTRACT_VERSION
    contract_id: str = ""
    source_type: SourceType
    entities: list[CanonicalEntity] = Field(default_factory=list)
    fields: list[CanonicalField] = Field(default_factory=list)
    events: list[CanonicalEvent] = Field(default_factory=list)
    relationships: list[CanonicalRelationship] = Field(default_factory=list)
    dependencies: list[DependencyEdge] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    privacy_policy: dict[str, Any] = Field(default_factory=dict)
    generation_policy: dict[str, Any] = Field(default_factory=dict)
    validation_policy: dict[str, Any] = Field(default_factory=dict)
    provenance: CanonicalProvenance = Field(default_factory=CanonicalProvenance)
