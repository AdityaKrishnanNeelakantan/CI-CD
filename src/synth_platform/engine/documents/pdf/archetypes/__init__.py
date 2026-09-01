"""Document archetypes package."""

from synth_platform.engine.documents.pdf.archetypes.registry import (
    get_archetype,
    list_archetypes,
    register_archetype,
    resolve_archetype,
)

__all__ = ["get_archetype", "list_archetypes", "register_archetype", "resolve_archetype"]
