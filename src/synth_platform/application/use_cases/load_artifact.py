"""Load a .synthpkg (hostile-input-safe) via the artifact store port."""
from __future__ import annotations

from pathlib import Path

from synth_platform.application.ports.artifact_store import ArtifactStore
from synth_platform.domain.artifacts.bundle import SynthArtifact


def load_artifact(store: ArtifactStore, path: str | Path) -> SynthArtifact:
    return store.read(path, verify=True)
