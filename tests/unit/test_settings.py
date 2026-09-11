from __future__ import annotations

from synth_platform.settings import Settings


def test_settings_have_local_ollama_demo_defaults() -> None:
    settings = Settings()

    assert settings.ollama_host == "http://localhost:11434"
    assert settings.ollama_model == "qwen3.5:9b"


def test_settings_allow_single_ollama_model_override(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "demo-model:latest")

    assert Settings.from_env().ollama_model == "demo-model:latest"
