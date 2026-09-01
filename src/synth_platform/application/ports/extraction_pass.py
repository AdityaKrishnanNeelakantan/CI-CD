"""ExtractionPass: the single front-end port (compiler front-end).

An ExtractionPass lifts a source into the canonical RelationalDataset IR. This
is the ONLY ingestion contract. Structured sources implement the trivial case
(schema is self-describing); unstructured sources implement the schema-guided
case (a caller-supplied TargetSchema drives the mapping). The back-end depends
on the produced IR, never on the pass.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from synth_platform.application.dto.dataset import RelationalDataset


@runtime_checkable
class ExtractionPass(Protocol):
    kind: str

    def extract(self) -> RelationalDataset:
        """Lift the source into the canonical relational IR (bounded/materialized)."""
        ...
