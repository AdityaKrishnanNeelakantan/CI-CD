"""Load a verified artifact, generate source-free tables, and validate them."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from synth_platform.domain.generation.models import GenerationRequest
from synth_platform.domain.runs.models import SourceKind
from synth_platform.domain.artifacts.guards import require_source_kind


class GenerateArtifactCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_path: str
    request: GenerationRequest
    expected_source_kind: SourceKind | None = None


class GeneratedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    artifact: object
    tables: dict
    report: object


def generate_from_artifact(cmd: GenerateArtifactCommand, deps) -> GeneratedDataset:
    reader = getattr(deps.store, "read_and_verify", None) or deps.store.read
    artifact = reader(cmd.artifact_path)
    if cmd.expected_source_kind is not None:
        require_source_kind(artifact, cmd.expected_source_kind)
    tables = deps.generator.run(artifact, cmd.request)
    report = deps.validator(artifact, tables)
    return GeneratedDataset(artifact=artifact, tables=tables, report=report)
