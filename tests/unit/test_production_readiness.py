"""Production readiness checks for client-facing deployments."""

from __future__ import annotations

from synth_platform.interfaces.streamlit import ui_config
from synth_platform.production import build_production_readiness_report, is_production_profile


def test_development_readiness_allows_local_defaults(monkeypatch):
    monkeypatch.delenv("SP_DEPLOYMENT_PROFILE", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.setenv("SP_PLATFORM_SLM_ENDPOINT", "http://localhost:11434/v1")
    monkeypatch.setenv("SP_PLATFORM_SLM_MODEL", "synth-platform-slm")
    ui_config.load_ui_config.cache_clear()

    report = build_production_readiness_report(ui_config.load_ui_config())

    assert report.profile == "development"
    assert report.ready is True


def test_production_readiness_blocks_mock_mode(monkeypatch):
    monkeypatch.setenv("SP_DEPLOYMENT_PROFILE", "production")
    monkeypatch.setenv("SP_PRODUCTION_ALLOW_LOCAL_SLM", "1")
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MOCK", "1")
    monkeypatch.setenv("SP_PLATFORM_SLM_ENDPOINT", "http://localhost:11434/v1")
    ui_config.load_ui_config.cache_clear()

    report = build_production_readiness_report(ui_config.load_ui_config())

    assert report.ready is False
    assert any(check.name == "mock_generation" and check.status == "fail" for check in report.checks)


def test_production_readiness_requires_remote_api_key(monkeypatch):
    monkeypatch.setenv("SP_DEPLOYMENT_PROFILE", "production")
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_MOCK", raising=False)
    monkeypatch.delenv("SP_NEMO_DATA_DESIGNER_API_KEY", raising=False)
    monkeypatch.setenv("SP_PLATFORM_SLM_ENDPOINT", "https://llm.example.test/v1")
    monkeypatch.setenv("SP_PLATFORM_SLM_API_KEY_ENV", "CLIENT_LLM_KEY")
    monkeypatch.delenv("CLIENT_LLM_KEY", raising=False)
    ui_config.load_ui_config.cache_clear()

    report = build_production_readiness_report(ui_config.load_ui_config())

    assert report.ready is False
    assert any(check.name == "slm_api_key" and check.status == "fail" for check in report.checks)


def test_ui_config_marks_pdf_as_production_wip():
    ui_config.load_ui_config.cache_clear()

    modes = ui_config.ui_value("home", "chat", "modes")
    pdf = next(mode for mode in modes if mode["key"] == "pdf")

    assert pdf["production_status"] == "work_in_progress"
    assert pdf["available_in_production"] is False


def test_production_profile_aliases():
    assert is_production_profile("production")
    assert is_production_profile("prod")
    assert not is_production_profile("development")
