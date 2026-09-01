"""ArtifactStore adapter binding writer+reader (implements the port)."""
from __future__ import annotations

from pathlib import Path

from synth_platform.infrastructure.artifacts.package_reader import read_package
from synth_platform.infrastructure.artifacts.package_writer import write_package
from synth_platform.domain.artifacts.bundle import SynthArtifact


class SynthpkgStore:
    def write(self, artifact: SynthArtifact, path: str | Path) -> Path:
        return write_package(artifact, path)

    def read(self, path: str | Path, verify: bool = True) -> SynthArtifact:
        return read_package(path, verify=verify)

    def write_signed(self, artifact: SynthArtifact, path: str | Path) -> Path:
        return self.write(artifact, path)

    def read_and_verify(self, path: str | Path) -> SynthArtifact:
        return self.read(path, verify=True)
