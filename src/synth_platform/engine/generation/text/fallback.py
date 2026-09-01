"""Deterministic fallback text when LLM output is unavailable or unsafe."""

from __future__ import annotations

from typing import Any, Dict, Optional

from synth_platform.engine.generation.text.faker_fallback import context_aware_fallback_text
from synth_platform.engine.generation.text.validators import sanitize_text, validate_generated_text
from synth_platform.engine.generation.text.vocabulary import DomainVocabulary, get_domain_vocabulary


def fallback_text(
    *,
    index: int,
    text_role: str,
    safe_context: Optional[Dict[str, str]] = None,
    domain: str = "generic",
    locale: str = "en_US",
    tone: str = "neutral professional",
    min_words: int = 8,
    max_words: int = 42,
    seed: Optional[int] = None,
    nullable: bool = False,
    null_probability: Optional[float] = None,
    rng_value: float = 0.0,
) -> Optional[str]:
    """Return context-aware Faker fallback text, optional null, or neutral placeholder."""
    if nullable and null_probability is not None and rng_value < float(null_probability):
        return None

    candidate = context_aware_fallback_text(
        index=index,
        text_role=text_role,
        safe_context=safe_context,
        domain=domain,
        locale=locale,
        tone=tone,
        min_words=min_words,
        max_words=max_words,
        seed=seed,
    )
    if validate_generated_text(
        candidate,
        safe_context=safe_context or {},
        min_words=min(3, min_words),
        max_words=max_words,
    ).passed:
        return candidate

    vocab = get_domain_vocabulary(domain)
    templates = list(vocab.templates) or ["Synthetic narrative text for generated tabular data."]
    base = sanitize_text(templates[index % len(templates)])
    if validate_generated_text(base, safe_context=safe_context or {}, min_words=3).passed:
        return base
    return base or sanitize_text(templates[0])


def domain_template(
    vocab: DomainVocabulary,
    *,
    index: int,
    safe_context: Optional[Dict[str, str]] = None,
) -> str:
    return fallback_text(
        index=index,
        text_role="narrative",
        safe_context=safe_context,
        domain=vocab.name,
    ) or "Synthetic narrative text."
