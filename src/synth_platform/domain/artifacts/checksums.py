"""Checksum value objects (pure)."""
from __future__ import annotations

import hashlib


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_checksum_file(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "  " not in line:
            continue
        digest, name = line.split("  ", 1)
        out[name] = digest
    return out


def render_checksum_file(mapping: dict[str, str]) -> str:
    return "".join(f"{d}  {n}\n" for n, d in sorted(mapping.items()))
