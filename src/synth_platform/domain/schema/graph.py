"""Schema-derived helpers (pure)."""
from __future__ import annotations

from synth_platform.domain.schema.models import DatabaseSchema


def parent_child_ratios(schema: DatabaseSchema) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for fk in schema.foreign_keys:
        p = schema.tables[fk.parent_table].row_count or 1
        c = schema.tables[fk.child_table].row_count
        ratios[f"{fk.child_table}->{fk.parent_table}"] = c / p
    return ratios
