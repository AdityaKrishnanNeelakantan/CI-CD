"""Synthetic PII in reserved namespaces (never reuses source values)."""
from __future__ import annotations

import numpy as np

_FIRST = ["Alex", "Sam", "Jordan", "Riley", "Casey", "Morgan", "Taylor", "Jamie"]
_LAST = ["Reed", "Vance", "Cole", "Frost", "Nash", "Pike", "Wren", "Ash"]


def _email(rng, n): return [f"user{int(rng.integers(0,10**8)):08d}@example.com" for _ in range(n)]
def _phone(rng, n): return [f"+1-555-01{int(rng.integers(0,100)):02d}-{int(rng.integers(0,10000)):04d}" for _ in range(n)]
def _ssn(rng, n): return [f"900-{int(rng.integers(0,100)):02d}-{int(rng.integers(0,10000)):04d}" for _ in range(n)]
def _card(rng, n): return [f"4000{int(rng.integers(0,10**12)):012d}" for _ in range(n)]
def _name(rng, n): return [f"{_FIRST[int(rng.integers(0,8))]} {_LAST[int(rng.integers(0,8))]}" for _ in range(n)]
def _addr(rng, n): return [f"{int(rng.integers(1,9999))} Example St, Springfield" for _ in range(n)]

_GEN = {"email": _email, "phone": _phone, "ssn": _ssn, "card_number": _card,
        "person_name": _name, "address": _addr}


def generate_pii(semantic_type: str, rng: np.random.Generator, n: int):
    return _GEN.get(semantic_type, _name)(rng, n)
