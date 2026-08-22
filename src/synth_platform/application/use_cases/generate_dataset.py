"""Generate relational data from a bundle (engine executor). Source-free."""
from __future__ import annotations

import pandas as pd

from synth_platform.application.dto.commands import GenerationRequest
from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.engine.generation.service import GenerationService


def generate_dataset(artifact: SynthArtifact, request: GenerationRequest) -> dict[str, pd.DataFrame]:
    return GenerationService().run(artifact, request)
