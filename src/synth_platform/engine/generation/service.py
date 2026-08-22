"""Generation service facade over the relational executor."""
from __future__ import annotations

import pandas as pd

from synth_platform.domain.generation.models import GenerationRequest
from synth_platform.domain.artifacts.bundle import SynthArtifact
from synth_platform.engine.generation.relational_executor import execute


class GenerationService:
    def run(self, artifact: SynthArtifact, request: GenerationRequest) -> dict[str, pd.DataFrame]:
        if request.output_format not in {"csv", "sqlite", "parquet", "postgresql", "postgres"}:
            raise ValueError(f"unsupported output format: {request.output_format!r}")
        return execute(artifact, request.root_table_rows, request.scale, request.seed)
