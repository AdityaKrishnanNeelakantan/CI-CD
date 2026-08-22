"""Profile the source (delegates to the engine profiling service)."""
from __future__ import annotations

from synth_platform.application.ports.source_connector import SourceConnector
from synth_platform.domain.profiling.models import DataProfile
from synth_platform.domain.schema.models import DatabaseSchema
from synth_platform.engine.profiling.service import ProfilingService


def profile_source(connector: SourceConnector, schema: DatabaseSchema,
                   max_rows: int, seed: int) -> tuple[DataProfile, dict]:
    """Returns (profile, raw_sensitive_values). The raw map is transient — used
    only by the compile-time privacy verifier, never serialized (A-07)."""
    return ProfilingService().run(connector, schema, max_rows=max_rows, seed=seed)
