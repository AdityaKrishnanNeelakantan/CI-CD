"""Archetype registry: config-driven document type routing.

Adding a new document type means registering a new DocumentArchetype
subclass here rather than editing src/documents/classifier.py's fixed
DOCUMENT_TYPES dict.
"""

from __future__ import annotations

from typing import Any

from synth_platform.engine.documents.pdf.archetypes.bank_statement import BankStatementArchetype
from synth_platform.engine.documents.pdf.archetypes.base import DocumentArchetype

_REGISTRY: dict[str, DocumentArchetype] = {}


def _ensure_registered() -> None:
    if _REGISTRY:
        return
    register_archetype(BankStatementArchetype())


def register_archetype(archetype: DocumentArchetype) -> None:
    _REGISTRY[archetype.archetype_id] = archetype


def get_archetype(archetype_id: str) -> DocumentArchetype | None:
    _ensure_registered()
    return _REGISTRY.get(archetype_id)


def list_archetypes() -> list[DocumentArchetype]:
    _ensure_registered()
    return list(_REGISTRY.values())


def resolve_archetype(text: str, classification: dict[str, Any]) -> dict[str, Any]:
    """Select the best-matching archetype or fall back to generic_shape_fallback."""
    _ensure_registered()

    best_id = "generic_shape_fallback"
    best_score = 0.0

    for archetype in _REGISTRY.values():
        score = archetype.match_score(text, classification)
        if score > best_score:
            best_score = score
            best_id = archetype.archetype_id

    if best_id == "generic_shape_fallback":
        return {
            "archetype_id": best_id,
            "display_name": "Generic Shape Fallback",
            "match_score": best_score,
            "config": None,
        }

    archetype = _REGISTRY[best_id]
    return {
        "archetype_id": best_id,
        "display_name": archetype.display_name,
        "match_score": round(best_score, 6),
        "config": archetype.to_config(),
    }
