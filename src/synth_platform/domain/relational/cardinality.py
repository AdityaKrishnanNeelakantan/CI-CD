"""Parent->child cardinality models (pure, deterministic)."""
from __future__ import annotations


def children_for(parent_rows: int, ratio: float) -> int:
    """Deterministic expected child-row count for a parent table."""
    return max(0, int(round(parent_rows * max(ratio, 0.0))))
