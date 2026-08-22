"""Guards preventing cross-track artifact misuse."""
from __future__ import annotations

from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.domain.runs.models import SourceKind


def require_source_kind(artifact: SynthArtifact, expected: SourceKind) -> None:
    actual = artifact.manifest.source_kind
    if actual != expected:
        raise ValueError(
            f"artifact source kind mismatch: expected {expected.value}, got {actual.value}"
        )
