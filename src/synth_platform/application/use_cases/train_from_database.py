"""Database-track training use case with guaranteed source closure."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from synth_platform.application.use_cases.materialize_connector import materialize_connector
from synth_platform.application.use_cases.train_model import fingerprint_schema, train_relational_dataset
from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.domain.runs.models import SourceKind


class TrainDatabaseCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_secret_ref: str
    allowed_schemas: tuple[str, ...] = ("public",)
    allowed_tables: tuple[str, ...] | None = None
    tables: list[str] | None = None
    sample_size: int = 50_000
    seed: int = 42
    rare_threshold: int = 10
    artifact_path: str
    source_run_id: str | None = None


class SourceFactory(Protocol):
    def create(self, secret_ref: str, allowlist: dict[str, Any]): ...


class ArtifactWriter(Protocol):
    def write_signed(self, artifact: SynthArtifact, path: str | Path) -> Path: ...


@dataclass
class DatabaseDeps:
    source_factory: SourceFactory
    artifact_store: ArtifactWriter


class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    artifact: SynthArtifact
    artifact_path: str
    source_health: dict[str, Any] = Field(default_factory=dict)
    source_disconnected: bool = True


def train_from_database(cmd: TrainDatabaseCommand, deps: DatabaseDeps) -> TrainingResult:
    source = deps.source_factory.create(cmd.source_secret_ref, {
        "schemas": cmd.allowed_schemas, "tables": cmd.allowed_tables,
    })
    try:
        health = source.health_check()
        source.assert_read_only()
        dataset, sampling = materialize_connector(
            source, cmd.sample_size, cmd.seed, selected_tables=cmd.tables)
        artifact = train_relational_dataset(
            dataset, source_kind=SourceKind.DATABASE,
            source_fingerprint=fingerprint_schema(dataset.schema),
            sample_records=sampling, seed=cmd.seed,
            rare_threshold=cmd.rare_threshold,
            source_run_id=cmd.source_run_id)
        path = deps.artifact_store.write_signed(artifact, cmd.artifact_path)
        health_data = health.model_dump(mode="json") if hasattr(health, "model_dump") else {}
        return TrainingResult(artifact=artifact, artifact_path=str(path),
                              source_health=health_data, source_disconnected=True)
    finally:
        source.close()
