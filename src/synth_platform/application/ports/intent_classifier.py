"""Port for workflow intent classification."""

from __future__ import annotations

from typing import Any, Protocol


class IntentClassifier(Protocol):
    def classify(self, request: Any) -> Any:
        """Return a structured workflow routing decision."""
