"""Artifact manifest + component descriptors (pure)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synth_platform.domain.artifacts.versions import ARTIFACT_FORMAT_VERSION
from synth_platform.domain.runs.models import SourceKind


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelComponent(_Base):
    """A trained per-column/table component descriptor."""
    table: str
    column: str | None = None
    backend: str = "statistical"
    payload_ref: str = ""
    tensor_ref: str | None = None


class ArtifactManifest(_Base):
    format_version: str = ARTIFACT_FORMAT_VERSION
    package_version: str = "3.1.0"
    artifact_id: str = ""
    created_at: str = ""
    generation_order: list[str] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    privacy_claim: str = "no_formal_dp"

    # Explicit source identity. Defaults preserve legacy package readability.
    source_kind: SourceKind = SourceKind.DATABASE
    source_run_id: str = ""
    source_schema_fingerprint: str = ""
    contract_version: str = ""
    contract_fingerprint: str = ""
    artifact_role: Literal["relational", "document_relational"] = "relational"

    # Deprecated compatibility alias; contains only a schema/document digest.
    source_fingerprint: str = ""

    @model_validator(mode="after")
    def _reject_source_locations(self) -> "ArtifactManifest":
        values = [self.source_run_id, self.source_schema_fingerprint,
                  self.source_fingerprint, self.contract_fingerprint,
                  *self.capabilities, *self.limitations]
        joined = " ".join(str(value) for value in values).lower()
        forbidden = ("://", "password=", "passwd=", "pwd=", "api_key=", "token=")
        if any(marker in joined for marker in forbidden):
            raise ValueError("artifact manifest must not contain source locations or credentials")
        return self
