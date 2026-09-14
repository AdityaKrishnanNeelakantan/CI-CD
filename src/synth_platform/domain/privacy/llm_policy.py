"""Central policy for optional LLM provider egress."""

from __future__ import annotations

import os
from dataclasses import dataclass


EXTERNAL_LLM_OPT_IN_ENV = "SYNTH_ALLOW_EXTERNAL_LLM"
LOCAL_PROVIDERS = frozenset({"ollama", "local"})
OFF_PROVIDERS = frozenset({"", "off", "none", "disabled"})
EXTERNAL_PROVIDERS = frozenset({"openai", "groq"})


class LlmPolicyError(RuntimeError):
    """Raised when an LLM provider is blocked by egress policy."""


@dataclass(frozen=True)
class LlmProviderDecision:
    provider: str
    allowed: bool
    is_external: bool
    reason: str = ""


class LlmPolicy:
    """Single source of truth for optional LLM provider egress decisions."""

    def __init__(self, *, allow_external: bool | None = None) -> None:
        self.allow_external = self._external_opt_in_enabled() if allow_external is None else bool(allow_external)

    @classmethod
    def from_environment(cls) -> "LlmPolicy":
        return cls()

    def require_provider(self, provider: str | None) -> LlmProviderDecision:
        decision = self.evaluate(provider)
        if not decision.allowed:
            raise LlmPolicyError(decision.reason)
        return decision

    def evaluate(self, provider: str | None) -> LlmProviderDecision:
        normalized = self.normalize_provider(provider)
        if normalized in OFF_PROVIDERS:
            return LlmProviderDecision(
                provider=normalized or "off",
                allowed=True,
                is_external=False,
                reason="LLM generation is disabled.",
            )
        if normalized in LOCAL_PROVIDERS:
            return LlmProviderDecision(provider=normalized, allowed=True, is_external=False)
        if normalized in EXTERNAL_PROVIDERS:
            if self.allow_external:
                return LlmProviderDecision(provider=normalized, allowed=True, is_external=True)
            return LlmProviderDecision(
                provider=normalized,
                allowed=False,
                is_external=True,
                reason=(
                    f"External LLM provider '{normalized}' is disabled in air-gapped mode. "
                    f"Enable local Ollama or set {EXTERNAL_LLM_OPT_IN_ENV}=true to use OpenAI/Groq."
                ),
            )
        raise LlmPolicyError(f"Unsupported LLM provider: {normalized}")

    @staticmethod
    def normalize_provider(provider: str | None) -> str:
        return str(provider or "off").strip().lower()

    @staticmethod
    def _external_opt_in_enabled() -> bool:
        return os.getenv(EXTERNAL_LLM_OPT_IN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
