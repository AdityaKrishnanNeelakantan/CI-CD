"""Backward-compatible facade over the generic text_generation package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from synth_platform.engine.generation.text.eligibility import ALLOWED_TEXT_ROLES
from synth_platform.engine.generation.text.generator import TextGenerationConfig, generate_text_column
from synth_platform.engine.generation.text.validators import sanitize_text, validate_generated_text

SERVICE_CASE_NOTE_ALIASES = frozenset(
    role for role in ALLOWED_TEXT_ROLES if "note" in role or "message" in role
)


@dataclass(frozen=True)
class LLMTextResult:
    values: List[str]
    used_llm: bool
    fallback_count: int
    provider_error: Optional[str] = None


def is_safe_banking_note(text: Any) -> bool:
    result = validate_generated_text(text, max_words=80, min_words=3)
    return result.passed


def sanitize_banking_note(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = sanitize_text(text)
    return cleaned if is_safe_banking_note(cleaned) else None


is_safe_service_note = is_safe_banking_note
sanitize_service_note = sanitize_banking_note


class SafeBankingTextGenerator:
    """Legacy wrapper — delegates to TextGenerationEngine."""

    def __init__(
        self,
        *,
        country: str = "GLOBAL",
        locale: Optional[str] = None,
        seed: Optional[int] = None,
        llm_enabled: bool = True,
        model: Optional[str] = None,
        provider: str = "ollama",
        timeout: float = 20.0,
    ) -> None:
        domain = "banking" if str(country).upper() in {"US", "IN"} else "generic"
        self._config = TextGenerationConfig(
            llm_enabled=llm_enabled,
            is_preview=True,
            provider=provider,
            model=model,
            locale=locale or "en_US",
            domain=domain,
            timeout=timeout,
            seed=seed,
        )

    def generate(
        self,
        *,
        size: int,
        table_name: str,
        column_name: str,
        context: Optional[str] = None,
        max_words: int = 42,
    ) -> LLMTextResult:
        from synth_platform.engine.inference.schema.schema import Column

        column = Column(
            name=column_name,
            type="text",
            distribution_params={
                "text_type": "case_note",
                "llm_text": True,
                "llm_enabled": self._config.llm_enabled,
                "max_words": max_words,
                "context": context,
                "domain": self._config.domain,
            },
            description=context,
        )
        result = generate_text_column(
            table_name=table_name,
            column=column,
            size=size,
            config=self._config,
        )
        values = [v if v is not None else "" for v in result.values]
        return LLMTextResult(
            values=values,
            used_llm=result.evidence.llm_used,
            fallback_count=result.evidence.fallback_count,
            provider_error=result.provider_error,
        )


SafeServiceTextGenerator = SafeBankingTextGenerator


def generate_safe_banking_notes(
    *,
    size: int,
    country: str = "GLOBAL",
    locale: Optional[str] = None,
    seed: Optional[int] = None,
    llm_enabled: bool = True,
    model: Optional[str] = None,
    provider: str = "ollama",
    table_name: str = "records",
    column_name: str = "notes",
    context: Optional[str] = None,
    max_words: int = 42,
) -> LLMTextResult:
    generator = SafeBankingTextGenerator(
        country=country,
        locale=locale,
        seed=seed,
        llm_enabled=llm_enabled,
        model=model,
        provider=provider,
    )
    return generator.generate(
        size=size,
        table_name=table_name,
        column_name=column_name,
        context=context,
        max_words=max_words,
    )


generate_safe_service_notes = generate_safe_banking_notes
