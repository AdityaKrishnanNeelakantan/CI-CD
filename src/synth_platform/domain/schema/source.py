"""SourceConnector port — the ingestion boundary contract (pure, inward).

Defined in the domain so the engine and application both depend inward on it,
never on an outer layer (ARCHITECTURE_AUDIT A-06 / RC-10). The materialized
sample is typed as `Any` here deliberately: the concrete dataframe type is an
outer-layer concern, and the domain must import no dataframe library. Adapters
implement this Protocol and return a real DataFrame.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from synth_platform.domain.schema.models import DatabaseSchema


@runtime_checkable
class SourceConnector(Protocol):
    kind: str

    def health_check(self) -> None: ...

    def discover_schema(self) -> DatabaseSchema: ...

    def count_rows(self, table: str) -> int: ...

    def sample_table(self, table: str, max_rows: int, seed: int = 0) -> Any:
        """Deterministic bounded sample (a dataframe in outer layers)."""
        ...

    def close(self) -> None: ...
