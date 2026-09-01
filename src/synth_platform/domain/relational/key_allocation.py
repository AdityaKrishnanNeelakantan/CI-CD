"""Deterministic key strategies (pure)."""
from __future__ import annotations


def sequential_pk(n: int, start: int = 1) -> list[int]:
    return list(range(start, start + n))
