"""Canonical contract version compatibility."""
from __future__ import annotations

CANONICAL_CONTRACT_VERSION = "1.0.0"
SUPPORTED_CONTRACT_VERSIONS = frozenset({CANONICAL_CONTRACT_VERSION})


class UnsupportedContractVersion(ValueError):
    """Raised when a contract version cannot be read by this package."""


def is_supported_contract_version(version: str) -> bool:
    return version in SUPPORTED_CONTRACT_VERSIONS


def assert_supported_contract_version(version: str) -> None:
    if not is_supported_contract_version(version):
        raise UnsupportedContractVersion(f"unsupported canonical contract version: {version}")
