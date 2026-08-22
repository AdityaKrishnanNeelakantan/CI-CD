"""Generic LLM text column generator with validation, fallback, and cost controls."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from synth_platform.engine.inference.schema.schema import Column
from synth_platform.engine.generation.text.context import build_batch_contexts, context_cache_key
from synth_platform.engine.generation.text.eligibility import assess_column_eligibility
from synth_platform.engine.generation.text.evidence import TextGenerationEvidence
from synth_platform.engine.generation.text.fallback import fallback_text
from synth_platform.engine.generation.text.prompts import build_system_message, build_text_generation_prompt
from synth_platform.engine.generation.text.validators import sanitize_text, validate_generated_text
from synth_platform.engine.generation.text.vocabulary import get_domain_vocabulary

DEFAULT_MAX_LLM_ROWS = 50
DEFAULT_BATCH_SIZE = 25
FULL_GENERATION_LLM_ROW_CAP = 200


@dataclass
class TextGenerationConfig:
    llm_enabled: bool = False
    is_preview: bool = True
    max_llm_rows: int = DEFAULT_MAX_LLM_ROWS
    batch_size: int = DEFAULT_BATCH_SIZE
    provider: str = "openai"
    model: Optional[str] = None
    locale: str = "en_US"
    domain: str = "generic"
    timeout: float = 20.0
    seed: Optional[int] = None
    allow_nulls: bool = True


@dataclass
class TextGenerationResult:
    values: List[Optional[str]]
    evidence: TextGenerationEvidence
    provider_error: Optional[str] = None


class TextGenerationEngine:
    """Generate one text-heavy column with LLM + safe fallback."""

    def __init__(self, config: Optional[TextGenerationConfig] = None) -> None:
        self.config = config or TextGenerationConfig()
        self._context_cache: Dict[str, str] = {}
        self._source_value_set: Set[str] = set()

    def generate(
        self,
        *,
        table_name: str,
        column: Column,
        size: int,
        table_data: Optional[pd.DataFrame] = None,
        source_series: Optional[pd.Series] = None,
    ) -> TextGenerationResult:
        start = time.perf_counter()
        params = column.distribution_params or {}
        eligibility = assess_column_eligibility(column, table_name=table_name)
        evidence = TextGenerationEvidence(table=table_name, column=column.name, rows_requested=size)

        if not eligibility.eligible:
            evidence.warnings.append(f"ineligible: {', '.join(eligibility.reasons)}")
            values = [
                fallback_text(
                    index=i,
                    text_role=eligibility.text_role or "narrative",
                    domain=self.config.domain,
                    locale=self.config.locale,
                    nullable=column.nullable,
                    null_probability=params.get("null_probability"),
                    rng_value=null_rng.random(),
                    seed=self.config.seed,
                )
                for i in range(size)
            ]
            evidence.fallback_count = size
            evidence.rows_generated = sum(1 for v in values if v is not None)
            evidence.generation_time_seconds = time.perf_counter() - start
            return TextGenerationResult(values=values, evidence=evidence)

        if source_series is not None:
            self._source_value_set = set(source_series.dropna().astype(str).str.strip())

        contexts = build_batch_contexts(table_data, target_column=column.name)
        while len(contexts) < size:
            contexts.append({})

        max_words = int(params.get("max_words", 42))
        min_words = int(params.get("min_words", 8))
        tone = str(params.get("tone", "neutral professional"))
        output_format = str(params.get("output_format", "plain text sentence"))
        domain = str(params.get("domain") or params.get("domain_hint") or self.config.domain)
        vocab = get_domain_vocabulary(domain)

        llm_cap = self._effective_llm_cap(size)
        llm_enabled = self.config.llm_enabled and llm_cap > 0
        if self.config.llm_enabled and llm_cap == 0:
            evidence.warnings.append("LLM disabled for large full generation; using templates.")

        llm_values: Dict[int, str] = {}
        provider_error: Optional[str] = None
        null_rng = np.random.default_rng(
            (self.config.seed or 0) + hash(f"{table_name}.{column.name}") % (2**32)
        )
        if llm_enabled:
            try:
                llm_values, cache_hits, attempted = self._generate_llm_batch(
                    table_name=table_name,
                    column=column,
                    contexts=contexts[:llm_cap],
                    text_role=eligibility.text_role or "narrative",
                    max_words=max_words,
                    min_words=min_words,
                    tone=tone,
                    output_format=output_format,
                    domain=domain,
                    vocab_words=list(vocab.topic_words),
                    column_purpose=column.description or params.get("context"),
                )
                evidence.llm_used = bool(llm_values)
                evidence.llm_rows_attempted = attempted
                evidence.cache_hits = cache_hits
            except Exception as exc:
                provider_error = str(exc)
                evidence.warnings.append(f"provider error: {provider_error}")

        values: List[Optional[str]] = []
        word_counts: List[int] = []
        for idx in range(size):
            null_prob = params.get("null_probability")
            if column.nullable and null_prob is not None:
                if null_rng.random() < float(null_prob):
                    values.append(None)
                    continue

            candidate = llm_values.get(idx)
            if candidate is not None:
                cleaned = sanitize_text(candidate)
                validation = validate_generated_text(
                    cleaned,
                    source_values=self._source_value_set,
                    max_words=max_words,
                    min_words=min_words,
                    safe_context=contexts[idx] if idx < len(contexts) else {},
                )
                if validation.passed:
                    values.append(cleaned)
                    word_counts.append(len(cleaned.split()))
                    continue
                if "source replay" in validation.reasons:
                    evidence.source_replay_count += 1
                if any(r in validation.reasons for r in ("email or url", "phone-like", "sensitive token", "id-like string")):
                    evidence.privacy_fail_count += 1
                else:
                    evidence.format_fail_count += 1

            fb = fallback_text(
                index=idx,
                text_role=eligibility.text_role or "narrative",
                safe_context=contexts[idx] if idx < len(contexts) else {},
                domain=domain,
                locale=self.config.locale,
                tone=tone,
                min_words=min_words,
                max_words=max_words,
                seed=self.config.seed,
                nullable=False,
            )
            values.append(fb)
            evidence.fallback_count += 1
            if fb:
                word_counts.append(len(str(fb).split()))

        evidence.rows_generated = sum(1 for v in values if v is not None)
        evidence.average_word_count = float(sum(word_counts) / len(word_counts)) if word_counts else 0.0
        evidence.generation_time_seconds = round(time.perf_counter() - start, 4)
        return TextGenerationResult(values=values, evidence=evidence, provider_error=provider_error)

    def _effective_llm_cap(self, size: int) -> int:
        cap = min(self.config.max_llm_rows, self.config.batch_size * 2, size)
        if not self.config.is_preview:
            cap = min(cap, FULL_GENERATION_LLM_ROW_CAP)
            if size > FULL_GENERATION_LLM_ROW_CAP and not self.config.llm_enabled:
                return 0
            if size > 10_000 and not self.config.llm_enabled:
                return 0
        if not self.config.llm_enabled:
            return 0
        return max(0, cap)

    def _generate_llm_batch(
        self,
        *,
        table_name: str,
        column: Column,
        contexts: List[Dict[str, str]],
        text_role: str,
        max_words: int,
        min_words: int,
        tone: str,
        output_format: str,
        domain: str,
        vocab_words: List[str],
        column_purpose: Optional[str],
    ) -> tuple[Dict[int, str], int, int]:
        if not contexts:
            return {}, 0, 0
        unique: Dict[str, List[int]] = {}
        cache_hits = 0
        for idx, ctx in enumerate(contexts):
            key = context_cache_key(ctx)
            if key in self._context_cache:
                cache_hits += 1
            unique.setdefault(key, []).append(idx)

        results: Dict[int, str] = {}
        pending_keys = [k for k in unique if k not in self._context_cache]
        if pending_keys:
            batch_context = contexts[unique[pending_keys[0]][0]]
            generated = self._call_provider(
                table_name=table_name,
                column_name=column.name,
                text_role=text_role,
                rows=len(pending_keys),
                safe_context=batch_context,
                max_words=max_words,
                min_words=min_words,
                tone=tone,
                output_format=output_format,
                domain=domain,
                vocab_words=vocab_words,
                column_purpose=column_purpose,
            )
            for i, key in enumerate(pending_keys):
                if i < len(generated):
                    self._context_cache[key] = generated[i]

        for key, indices in unique.items():
            text = self._context_cache.get(key, "")
            for idx in indices:
                results[idx] = text
        return results, cache_hits, len(contexts)

    def _call_provider(
        self,
        *,
        table_name: str,
        column_name: str,
        text_role: str,
        rows: int,
        safe_context: Dict[str, str],
        max_words: int,
        min_words: int,
        tone: str,
        output_format: str,
        domain: str,
        vocab_words: List[str],
        column_purpose: Optional[str],
    ) -> List[str]:
        provider = str(self.config.provider or "openai").lower()
        if provider not in {"openai", "groq"}:
            raise RuntimeError(f"Unsupported LLM provider: {provider}")
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(f"{provider.upper()} API key is not configured")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is not installed; install mvp[llm]") from exc

        base_url = os.getenv("OPENAI_BASE_URL") if provider == "openai" else os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        model = self.config.model or os.getenv("MVP_LLM_TEXT_MODEL") or ("gpt-4o-mini" if provider == "openai" else "llama-3.1-8b-instant")
        client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=self.config.timeout)
        prompt = build_text_generation_prompt(
            table_name=table_name,
            column_name=column_name,
            text_role=text_role,
            rows=rows,
            safe_context=safe_context,
            tone=tone,
            output_format=output_format,
            min_words=min_words,
            max_words=max_words,
            locale=self.config.locale,
            domain_hint=domain,
            column_purpose=column_purpose,
            domain_vocabulary=vocab_words,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": build_system_message()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content or "[]"
        return self._parse_payload(content)

    @staticmethod
    def _parse_payload(payload: str) -> List[str]:
        text = payload.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("notes", "values", "rows", "data"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        if not isinstance(parsed, list):
            raise ValueError("LLM response was not a JSON array")
        return [str(item) for item in parsed]


def generate_text_column(
    *,
    table_name: str,
    column: Column,
    size: int,
    table_data: Optional[pd.DataFrame] = None,
    config: Optional[TextGenerationConfig] = None,
    source_series: Optional[pd.Series] = None,
) -> TextGenerationResult:
    engine = TextGenerationEngine(config=config)
    return engine.generate(
        table_name=table_name,
        column=column,
        size=size,
        table_data=table_data,
        source_series=source_series,
    )
