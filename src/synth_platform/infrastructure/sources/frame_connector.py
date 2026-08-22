"""FrameConnector: present a materialized RelationalDataset as a SourceConnector.

The bridge between the compiler front-end (any ExtractionPass -> RelationalDataset)
and the connector-based back-end (profiling/training). Because every source —
SQLite, CSV, or a PDF via DocumentExtractionPass — lifts into the same IR, this
one adapter lets all of them train through the identical back-end. Generic: it
knows nothing about the source's origin.
"""
from __future__ import annotations

import pandas as pd

from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.domain.schema.models import DatabaseSchema


class FrameConnector:
    kind = "frames"

    def __init__(self, dataset: RelationalDataset):
        dataset.validate()
        self._dataset = dataset

    def health_check(self) -> None:
        if not self._dataset.tables:
            raise ValueError("empty dataset")

    def discover_schema(self) -> DatabaseSchema:
        return self._dataset.schema

    def count_rows(self, table: str) -> int:
        return int(len(self._dataset.tables[table]))

    def sample_table(self, table: str, max_rows: int, seed: int = 0) -> pd.DataFrame:
        return self._dataset.tables[table].head(max_rows).reset_index(drop=True)

    def close(self) -> None:
        pass
