"""Export the bundle to a .synthpkg via the artifact store port."""
from __future__ import annotations

from pathlib import Path

from synth_platform.application.ports.artifact_store import ArtifactStore
from synth_platform.domain.artifacts.bundle import SynthArtifact


def export_artifact(store: ArtifactStore, artifact: SynthArtifact, path: str | Path) -> Path:
    return store.write(artifact, path)
