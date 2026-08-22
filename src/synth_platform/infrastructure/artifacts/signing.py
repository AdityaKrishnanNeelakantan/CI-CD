"""Ed25519 signing helpers with a compiled trusted demo public key."""
from __future__ import annotations

import hashlib
import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

# Local-demo key. Production deployments must inject SYNTH_SIGNING_PRIVATE_KEY_HEX
# and SYNTH_TRUSTED_PUBLIC_KEY_HEX from a secret manager/configuration boundary.
_DEMO_SEED = hashlib.sha256(b"synth-platform-local-demo-signing-key-v1").digest()
_DEMO_PRIVATE = Ed25519PrivateKey.from_private_bytes(_DEMO_SEED)
_DEMO_PUBLIC_BYTES = _DEMO_PRIVATE.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw,
)


def private_key_from_env() -> Ed25519PrivateKey:
    raw = os.getenv("SYNTH_SIGNING_PRIVATE_KEY_HEX")
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(raw)) if raw else _DEMO_PRIVATE


def trusted_public_key_from_env() -> Ed25519PublicKey:
    raw = os.getenv("SYNTH_TRUSTED_PUBLIC_KEY_HEX")
    key_bytes = bytes.fromhex(raw) if raw else _DEMO_PUBLIC_BYTES
    return Ed25519PublicKey.from_public_bytes(key_bytes)


def sign(payload: bytes) -> bytes:
    return private_key_from_env().sign(payload)


def verify(signature: bytes, payload: bytes) -> None:
    trusted_public_key_from_env().verify(signature, payload)
