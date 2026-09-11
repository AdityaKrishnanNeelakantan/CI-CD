"""Intent -> generation presets loaded from Streamlit YAML config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from synth_platform.interfaces.streamlit.ui_config import ui_value


@dataclass(frozen=True)
class SchemaIntentPreset:
    rows_per_table: int
    locale: str
    prefer_llm_text: bool
    caption: str


@dataclass(frozen=True)
class DatabaseIntentPreset:
    preferred_model_type: str  # safe_gaussian_copula | dp_gaussian_copula
    dp_epsilon: float
    sample_entity_default: int
    locale: str
    caption: str


@dataclass(frozen=True)
class PdfIntentPreset:
    locale: str
    prefer_llm_narrative: bool
    caption: str


def schema_intent_preset(intent: str) -> SchemaIntentPreset:
    data = _preset("schema", intent)
    return SchemaIntentPreset(
        rows_per_table=_int(data.get("rows_per_table"), 100),
        locale=str(data.get("locale") or "en_US"),
        prefer_llm_text=_bool(data.get("prefer_llm_text")),
        caption=str(data.get("caption") or ""),
    )


def database_intent_preset(intent: str) -> DatabaseIntentPreset:
    data = _preset("database", intent)
    return DatabaseIntentPreset(
        preferred_model_type=str(data.get("preferred_model_type") or "safe_gaussian_copula"),
        dp_epsilon=_float(data.get("dp_epsilon"), 1.0),
        sample_entity_default=_int(data.get("sample_entity_default"), 400),
        locale=str(data.get("locale") or "en_US"),
        caption=str(data.get("caption") or ""),
    )


def pdf_intent_preset(intent: str) -> PdfIntentPreset:
    data = _preset("pdf", intent)
    return PdfIntentPreset(
        locale=str(data.get("locale") or "en_US"),
        prefer_llm_narrative=_bool(data.get("prefer_llm_narrative")),
        caption=str(data.get("caption") or ""),
    )


def _preset(section: str, intent: str) -> Mapping[str, Any]:
    presets = ui_value(section, "intent_presets", default={})
    if not isinstance(presets, Mapping):
        return {}
    value = presets.get(intent)
    if isinstance(value, Mapping):
        return value
    default_intent = ui_value(section, "default_intent", default="")
    fallback = presets.get(default_intent)
    return fallback if isinstance(fallback, Mapping) else {}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
