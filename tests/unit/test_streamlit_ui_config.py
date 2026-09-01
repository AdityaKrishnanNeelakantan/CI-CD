"""Tests for user-editable Streamlit UI configuration."""

from __future__ import annotations

from synth_platform.interfaces.streamlit import ui_config


def test_default_ui_config_loads_navigation_and_home():
    ui_config.load_ui_config.cache_clear()

    config = ui_config.load_ui_config()

    assert config["app"]["page_title"] == "Synthetic Data Twin Platform"
    assert config["home"]["workflows"]
    assert config["database"]["model_type_labels"]["safe_gaussian_copula"] == "Standard ML twin"


def test_partial_external_ui_config_deep_merges(monkeypatch, tmp_path):
    config_path = tmp_path / "ui.yaml"
    config_path.write_text(
        """
app:
  page_title: Custom Twin
database:
  steps:
    train:
      title: Learn portable twin
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("SP_STREAMLIT_UI_CONFIG", str(config_path))
    ui_config.load_ui_config.cache_clear()

    config = ui_config.load_ui_config()

    assert config["app"]["page_title"] == "Custom Twin"
    assert config["app"]["page_icon"] == ":material/data_object:"
    assert ui_config.ui_step("database", "train")["title"] == "Learn portable twin"
    assert config["database"]["intent_options"][0] == "Development & testing"


def test_missing_external_config_uses_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("SP_STREAMLIT_UI_CONFIG", str(tmp_path / "missing.yaml"))
    ui_config.load_ui_config.cache_clear()

    assert ui_config.ui_value("schema", "default_intent") == "Development & testing"
