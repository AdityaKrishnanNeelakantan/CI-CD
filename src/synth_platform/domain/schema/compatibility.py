"""Format-version compatibility rules (pure)."""
from __future__ import annotations

SUPPORTED_PREFIX = "1."


def is_supported(format_version: str) -> bool:
    return str(format_version).startswith(SUPPORTED_PREFIX)
