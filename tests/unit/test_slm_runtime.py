from __future__ import annotations

import os

from synth_platform.infrastructure.slm_runtime import (
    apply_local_endpoint_network_policy,
    data_designer_max_parallel_requests,
    data_designer_max_tokens,
    data_designer_provider_api_key,
    data_designer_timeout,
    resolve_platform_slm_runtime,
)


def test_platform_slm_runtime_is_single_source_of_truth(monkeypatch):
    monkeypatch.delenv("INTERNAL_LLM_API_KEY", raising=False)
    monkeypatch.setenv("SP_PLATFORM_SLM_ENDPOINT", "http://localhost:11434/v1")
    monkeypatch.setenv("SP_PLATFORM_SLM_PROVIDER", "internal")
    monkeypatch.setenv("SP_PLATFORM_SLM_MODEL", "synth-platform-slm")
    monkeypatch.setenv("SP_PLATFORM_SLM_API_KEY_ENV", "INTERNAL_LLM_API_KEY")
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MODEL", "legacy/model")

    runtime = resolve_platform_slm_runtime()

    assert runtime.endpoint == "http://127.0.0.1:11434/v1"
    assert runtime.provider == "internal"
    assert runtime.model_id == "synth-platform-slm"
    assert runtime.api_key_env == "INTERNAL_LLM_API_KEY"
    assert data_designer_provider_api_key(runtime) is None


def test_local_slm_runtime_bypasses_proxy_envs(monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    apply_local_endpoint_network_policy("http://127.0.0.1:8000/v1")

    assert "127.0.0.1" in os.environ["NO_PROXY"]
    assert "localhost" in os.environ["NO_PROXY"]
    assert "::1" in os.environ["no_proxy"]


def test_default_platform_slm_runtime_uses_single_local_alias(monkeypatch):
    for name in (
        "SP_PLATFORM_SLM_MODEL",
        "SP_DIGITAL_TWIN_SLM_MODEL",
        "SP_NEMO_DATA_DESIGNER_MODEL",
        "SP_PLATFORM_SLM_ENDPOINT",
        "SP_DIGITAL_TWIN_SLM_ENDPOINT",
        "SP_NEMO_DATA_DESIGNER_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

    runtime = resolve_platform_slm_runtime()

    assert runtime.model_id == "synth-platform-slm"
    assert runtime.endpoint == "http://127.0.0.1:11434/v1"


def test_legacy_data_designer_model_env_is_ignored(monkeypatch):
    monkeypatch.delenv("SP_PLATFORM_SLM_MODEL", raising=False)
    monkeypatch.delenv("SP_DIGITAL_TWIN_SLM_MODEL", raising=False)
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MODEL", "legacy/model")

    runtime = resolve_platform_slm_runtime()

    assert runtime.model_id == "synth-platform-slm"


def test_data_designer_slm_limits_use_platform_env_first(monkeypatch):
    monkeypatch.setenv("SP_PLATFORM_SLM_TIMEOUT", "360")
    monkeypatch.setenv("SP_PLATFORM_SLM_MAX_PARALLEL_REQUESTS", "1")
    monkeypatch.setenv("SP_PLATFORM_SLM_MAX_TOKENS", "1024")
    monkeypatch.setenv("SP_TRANSCRIPT_DATA_DESIGNER_TIMEOUT", "120")

    assert data_designer_timeout(workflow_env="SP_TRANSCRIPT_DATA_DESIGNER_TIMEOUT") == 360
    assert data_designer_max_parallel_requests() == 1
    assert data_designer_max_tokens(2048) == 1024


def test_data_designer_slm_limits_allow_workflow_fallback(monkeypatch):
    monkeypatch.delenv("SP_PLATFORM_SLM_TIMEOUT", raising=False)
    monkeypatch.setenv("SP_TRANSCRIPT_SSOT_TIMEOUT", "240")

    assert data_designer_timeout(workflow_env="SP_TRANSCRIPT_SSOT_TIMEOUT") == 240
