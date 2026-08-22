"""Shared source helpers: strict identifier validation (injection choke point)."""
from __future__ import annotations

import re

from synth_platform.errors import UnsafeIdentifierError

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str) -> str:
    if not isinstance(name, str) or not _IDENT.match(name):
        raise UnsafeIdentifierError(f"unsafe SQL identifier: {name!r}")
    return name
