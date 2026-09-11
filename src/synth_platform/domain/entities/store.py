"""Synthetic entity identity map."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EntityIdentityMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identities: dict[str, dict[str, str]] = Field(default_factory=dict)

    def set(self, entity_type: str, source_key: str, synthetic_id: str) -> None:
        self.identities.setdefault(entity_type, {})[source_key] = synthetic_id

    def get(self, entity_type: str, source_key: str) -> str | None:
        return self.identities.get(entity_type, {}).get(source_key)
