"""Hostile-input verification for .synthpkg members (before any construction)."""
from __future__ import annotations

import posixpath

from synth_platform.errors import (
    ArtifactIntegrityError, PickleRejectedError, UnsafePathError,
)

_ALLOWED_SUFFIX = (".json", ".sha256", ".safetensors", ".ed25519")
_MAX_MEMBER_BYTES = 64 * 1024 * 1024   # 64 MB per member, uncompressed


def check_member_name(name: str) -> None:
    if name.startswith("/") or name.startswith("\\"):
        raise UnsafePathError(f"absolute member path: {name!r}")
    norm = posixpath.normpath(name)
    if norm.startswith("..") or "/../" in ("/" + norm):
        raise UnsafePathError(f"path traversal in member: {name!r}")
    if name.endswith((".pkl", ".pickle", ".pt", ".bin", ".joblib")):
        raise PickleRejectedError(f"refused executable/pickle member: {name!r}")
    if not name.endswith(_ALLOWED_SUFFIX) and "/" not in name:
        raise ArtifactIntegrityError(f"disallowed member suffix: {name!r}")


def check_member_size(name: str, size: int) -> None:
    if size > _MAX_MEMBER_BYTES:
        raise ArtifactIntegrityError(f"member too large: {name} ({size} bytes)")


def reject_pickle_bytes(blob: bytes) -> None:
    # pickle protocol 2+ opcode PROTO (0x80) or classic '(c' / 'c__' signatures
    if blob[:1] == b"\x80" or blob[:2] in (b"(c", b"c_"):
        raise PickleRejectedError("pickle stream detected in member payload")
