"""Intent presets are data-driven from Streamlit YAML config."""

from __future__ import annotations

from synth_platform.interfaces.streamlit import ui_config
from synth_platform.interfaces.streamlit.components.common.intent_presets import (
    database_intent_preset,
    pdf_intent_preset,
    schema_intent_preset,
)


def test_schema_intent_preset_uses_yaml_overlay(monkeypatch, tmp_path):
    path = tmp_path / "ui.yaml"
    path.write_text(
        """
schema:
  default_intent: Client pilot
  intent_presets:
    Client pilot:
      rows_per_table: 17
      locale: fr_FR
      prefer_llm_text: true
      caption: Client-specific schema preset.
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("SP_STREAMLIT_UI_CONFIG", str(path))
    ui_config.load_ui_config.cache_clear()

    preset = schema_intent_preset("Client pilot")

    assert preset.rows_per_table == 17
    assert preset.locale == "fr_FR"
    assert preset.prefer_llm_text is True
    assert preset.caption == "Client-specific schema preset."


def test_database_intent_preset_uses_yaml_overlay(monkeypatch, tmp_path):
    path = tmp_path / "ui.yaml"
    path.write_text(
        """
database:
  default_intent: Partner share
  intent_presets:
    Partner share:
      preferred_model_type: dp_gaussian_copula
      dp_epsilon: 0.5
      sample_entity_default: 125
      locale: de_DE
      caption: Client-specific database preset.
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("SP_STREAMLIT_UI_CONFIG", str(path))
    ui_config.load_ui_config.cache_clear()

    preset = database_intent_preset("Partner share")

    assert preset.preferred_model_type == "dp_gaussian_copula"
    assert preset.dp_epsilon == 0.5
    assert preset.sample_entity_default == 125
    assert preset.locale == "de_DE"


def test_pdf_intent_preset_uses_yaml_overlay(monkeypatch, tmp_path):
    path = tmp_path / "ui.yaml"
    path.write_text(
        """
pdf:
  default_intent: Regulated document
  intent_presets:
    Regulated document:
      locale: en_GB
      prefer_llm_narrative: true
      caption: Client-specific PDF preset.
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("SP_STREAMLIT_UI_CONFIG", str(path))
    ui_config.load_ui_config.cache_clear()

    preset = pdf_intent_preset("Regulated document")

    assert preset.locale == "en_GB"
    assert preset.prefer_llm_narrative is True
    assert preset.caption == "Client-specific PDF preset."
