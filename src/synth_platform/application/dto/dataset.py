"""RelationalDataset: the ONE canonical ingestion IR (schema + materialized rows).

Every source, structured or unstructured, is lifted into this by an
ExtractionPass. The back-end (profiling -> planning -> training) consumes only
this — it never knows whether the rows came from SQLite, a CSV, or a PDF.

Holds pandas frames, so it lives in the application layer, not domain (which
stays pandas-free). The pure typed schema is the domain DatabaseSchema.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from synth_platform.domain.schema.models import DatabaseSchema


@dataclass
class RelationalDataset:
    schema: DatabaseSchema
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)

    def finalize_counts(self) -> "RelationalDataset":
        """Stamp row counts from the materialized frames onto the schema."""
        for name, ts in self.schema.tables.items():
            ts.row_count = int(len(self.tables.get(name, [])))
        return self

    def validate(self) -> None:
        """Every schema table must have a frame; declared PK/FK columns present."""
        for name, ts in self.schema.tables.items():
            if name not in self.tables:
                raise ValueError(f"dataset missing frame for table '{name}'")
            cols = set(self.tables[name].columns)
            if ts.primary_key and ts.primary_key not in cols:
                raise ValueError(f"table '{name}' missing PK column '{ts.primary_key}'")
        for fk in self.schema.foreign_keys:
            child = self.tables.get(fk.child_table)
            if child is not None and fk.child_column not in child.columns:
                raise ValueError(
                    f"table '{fk.child_table}' missing FK column '{fk.child_column}'")
