"""ArtifactStore port: persist/load a .synthpkg bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from synth_platform.domain.artifacts.bundle import SynthArtifact


class ArtifactStore(Protocol):
    def write(self, artifact: SynthArtifact, path: str | Path) -> Path: ...

    def read(self, path: str | Path, verify: bool = True) -> SynthArtifact: ...
