from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from synth_platform.engine.generation.schema.smart_values import SmartValueGenerator
from synth_platform.engine.generation.text.generator import TextGenerationConfig, TextGenerationEngine
from synth_platform.engine.inference.schema.schema import Column
from synth_platform.domain.privacy.llm_policy import EXTERNAL_LLM_OPT_IN_ENV, LlmPolicyError


def _eligible_text_column() -> Column:
    return Column(name="support_notes", type="text", distribution_params={"text_type": "notes", "llm_text": True})


def test_text_generation_blocks_external_provider_without_explicit_opt_in(monkeypatch):
    monkeypatch.delenv(EXTERNAL_LLM_OPT_IN_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "present-but-not-enough")
    engine = TextGenerationEngine(TextGenerationConfig(llm_enabled=True, provider="openai", max_llm_rows=1))

    with pytest.raises(LlmPolicyError, match=EXTERNAL_LLM_OPT_IN_ENV):
        engine.generate(table_name="records", column=_eligible_text_column(), size=1)


def test_smart_value_generation_blocks_external_provider_without_explicit_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv(EXTERNAL_LLM_OPT_IN_ENV, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "present-but-not-enough")
    generator = SmartValueGenerator(provider="groq", cache_dir=str(tmp_path))

    with pytest.raises(LlmPolicyError, match=EXTERNAL_LLM_OPT_IN_ENV):
        generator.generate_pool_with_llm("disease", "clinic", 2)


def test_text_generation_external_provider_proceeds_with_explicit_opt_in(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='["A careful support note with realistic detail"]')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv(EXTERNAL_LLM_OPT_IN_ENV, "true")
    monkeypatch.setenv("OPENAI_API_KEY", "allowed")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    engine = TextGenerationEngine(TextGenerationConfig(llm_enabled=True, provider="openai", max_llm_rows=1))
    result = engine.generate(table_name="records", column=_eligible_text_column(), size=1)

    assert calls
    assert result.provider_error is None


def test_smart_value_external_provider_proceeds_with_explicit_opt_in(monkeypatch, tmp_path):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='["Alpha", "Beta"]')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeGroq:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv(EXTERNAL_LLM_OPT_IN_ENV, "true")
    monkeypatch.setenv("GROQ_API_KEY", "allowed")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))

    generator = SmartValueGenerator(provider="groq", cache_dir=str(tmp_path))
    assert generator.generate_pool_with_llm("disease", "clinic", 2) == ["Alpha", "Beta"]
    assert calls


def test_text_generation_allows_local_ollama_without_external_opt_in(monkeypatch):
    monkeypatch.delenv(EXTERNAL_LLM_OPT_IN_ENV, raising=False)

    def fake_complete(*, model, host, system, user, timeout):
        return '["A local-only narrative note with useful detail"]'

    monkeypatch.setattr("synth_platform.engine.generation.text.generator._call_ollama_chat", fake_complete)
    engine = TextGenerationEngine(TextGenerationConfig(llm_enabled=True, provider="ollama", max_llm_rows=1))

    result = engine.generate(table_name="records", column=_eligible_text_column(), size=1)

    assert result.provider_error is None
    assert result.values[0]


def test_smart_value_disk_cache_is_local_json_only(tmp_path):
    generator = SmartValueGenerator(provider="ollama", cache_dir=str(tmp_path))
    cache_key = generator._get_cache_key("disease", "clinic", 2)
    generator._save_pool_to_cache(cache_key, ["Alpha", "Beta"])

    assert generator._load_cached_pool(cache_key) == ["Alpha", "Beta"]
    assert (tmp_path / f"{cache_key}.json").is_file()
