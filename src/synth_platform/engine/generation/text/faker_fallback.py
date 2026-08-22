"""Context-aware Faker fallback for LLM text generation."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

import numpy as np

from synth_platform.engine.generation.schema.realism import RealisticTextGenerator
from synth_platform.engine.generation.text.validators import sanitize_text, validate_generated_text
from synth_platform.engine.generation.text.vocabulary import get_domain_vocabulary


_PERSONA_TONE = {
    "high": "urgent",
    "urgent": "urgent",
    "critical": "urgent",
    "low": "calm",
    "medium": "neutral",
    "open": "active",
    "closed": "resolved",
    "resolved": "resolved",
    "pending": "pending",
    "delayed": "pending",
}

_ROLE_TO_SEMANTIC = {
    "notes": "support_ticket",
    "case_note": "support_ticket",
    "service_case_note": "support_ticket",
    "support_case_note": "support_ticket",
    "support_message": "support_ticket",
    "ticket_message": "support_ticket",
    "review": "review",
    "comment": "comment_body",
    "comments": "comment_body",
    "feedback": "review",
    "description": "comment_body",
    "summary": "comment_body",
    "narrative": "support_ticket",
    "long_text": "comment_body",
}


def _seed_from_parts(*parts: Any) -> int:
    blob = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _infer_persona(safe_context: Dict[str, str]) -> str:
    for key in ("priority", "status", "state", "stage", "segment", "tier"):
        value = str(safe_context.get(key, "")).strip().lower()
        if value in _PERSONA_TONE:
            return _PERSONA_TONE[value]
    return "neutral"


def _context_phrase(safe_context: Dict[str, str], *, persona: str) -> str:
    bits: List[str] = []
    for key in sorted(safe_context.keys())[:4]:
        value = safe_context[key]
        if value and value != "[redacted]":
            bits.append(f"{key} is {value.lower()}")
    if not bits:
        return ""
    joined = ", ".join(bits)
    if persona == "urgent":
        return f"Given {joined}, the update should reflect urgency without naming people or identifiers."
    if persona == "resolved":
        return f"Given {joined}, the update should describe a resolved workflow step without private details."
    if persona == "pending":
        return f"Given {joined}, the update should note a pending follow-up without unsupported facts."
    return f"Given {joined}, the update should stay aligned with the provided context."


class ContextAwareFakerFallback:
    """Generate privacy-safe, context-aware fallback text using Faker + domain templates."""

    def __init__(self, *, locale: str = "en_US", domain: str = "generic", seed: Optional[int] = None) -> None:
        self.locale = locale
        self.domain = domain
        self.seed = seed or 0
        self._cache: Dict[str, str] = {}

    def _engine(self, *, index: int, text_role: str, safe_context: Dict[str, str]) -> RealisticTextGenerator:
        seed = _seed_from_parts(self.seed, index, text_role, self.domain, tuple(sorted(safe_context.items())))
        return RealisticTextGenerator(rng=np.random.default_rng(seed), locale=self.locale)

    def _faker(self):
        try:
            from synth_platform.engine.generation.schema.locales.registry import LocaleRegistry

            return LocaleRegistry.global_instance().get_faker(self.locale)
        except Exception:
            try:
                from faker import Faker

                fake = Faker(self.locale)
                Faker.seed(self.seed)
                return fake
            except Exception:
                return None

    def generate(
        self,
        *,
        index: int,
        text_role: str,
        safe_context: Optional[Dict[str, str]] = None,
        tone: str = "neutral professional",
        min_words: int = 8,
        max_words: int = 42,
    ) -> str:
        ctx = dict(safe_context or {})
        cache_key = f"{index}|{text_role}|{self.domain}|{tone}|{tuple(sorted(ctx.items()))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        persona = _infer_persona(ctx)
        semantic = _ROLE_TO_SEMANTIC.get(text_role, "comment_body")
        engine = self._engine(index=index, text_role=text_role, safe_context=ctx)
        base = str(engine.generate("text", "records", 1, semantic_type=semantic)[0])

        vocab = get_domain_vocabulary(self.domain)
        template = vocab.templates[index % max(len(vocab.templates), 1)] if vocab.templates else base
        context_hint = _context_phrase(ctx, persona=persona)
        faker = self._faker()
        filler = ""
        if faker is not None:
            try:
                filler = faker.paragraph(nb_sentences=1) if persona == "neutral" else faker.sentence(nb_words=8)
            except Exception:
                filler = faker.sentence(nb_words=8)

        parts = [template, base, context_hint, filler]
        candidate = sanitize_text(" ".join(p for p in parts if p))
        candidate = self._trim_words(candidate, min_words=min_words, max_words=max_words)

        if not validate_generated_text(candidate, safe_context=ctx, min_words=min(3, min_words), max_words=max_words).passed:
            candidate = sanitize_text(f"{template} {context_hint}".strip() or template)

        self._cache[cache_key] = candidate
        return candidate

    @staticmethod
    def _trim_words(text: str, *, min_words: int, max_words: int) -> str:
        words = text.split()
        if len(words) > max_words:
            words = words[:max_words]
            text = " ".join(words)
            if not text.endswith("."):
                text += "."
        if len(words) < min_words and words:
            text = text + " " + " ".join(["Review"] * (min_words - len(words)))
        return text.strip()


def context_aware_fallback_text(
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
) -> str:
    generator = ContextAwareFakerFallback(locale=locale, domain=domain, seed=seed)
    return generator.generate(
        index=index,
        text_role=text_role,
        safe_context=safe_context,
        tone=tone,
        min_words=min_words,
        max_words=max_words,
    )
