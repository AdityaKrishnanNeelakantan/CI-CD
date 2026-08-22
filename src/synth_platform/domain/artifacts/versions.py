"""Supported artifact format versions (pure)."""
from __future__ import annotations

ARTIFACT_FORMAT_VERSION = "2.0.0"
SUPPORTED_PREFIX = "2."


def is_supported(version: str) -> bool:
    return str(version).startswith(SUPPORTED_PREFIX)
