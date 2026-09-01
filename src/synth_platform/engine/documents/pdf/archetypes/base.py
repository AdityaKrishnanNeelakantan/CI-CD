"""Base contract for document archetypes.

An archetype encapsulates document-type-specific template logic: field
labels, expected regions, generator priors, and validation rules.  Each
archetype is registered in the archetype registry so new document types
can be added without editing the classifier's fixed DOCUMENT_TYPES dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DocumentArchetype(ABC):
    archetype_id: str = ""
    document_types: frozenset[str] = frozenset()
    display_name: str = ""

    @abstractmethod
    def expected_field_labels(self) -> list[str]:
        """Labels this archetype expects to find in a source document."""

    @abstractmethod
    def generator_priors(self) -> dict[str, Any]:
        """Default generation priors (value ranges, patterns) for this archetype."""

    @abstractmethod
    def validation_rules(self) -> list[dict[str, Any]]:
        """Post-render validation rules specific to this archetype."""

    def match_score(self, text: str, classification: dict[str, Any]) -> float:
        """Return a 0-1 match score for routing to this archetype."""
        if classification.get("document_type") in self.document_types:
            return classification.get("confidence", 0.0)
        return 0.0

    def to_config(self) -> dict[str, Any]:
        return {
            "archetype_id": self.archetype_id,
            "document_types": sorted(self.document_types),
            "display_name": self.display_name,
            "expected_field_labels": self.expected_field_labels(),
            "generator_priors": self.generator_priors(),
            "validation_rules": self.validation_rules(),
        }
