"""User-facing Streamlit UI configuration.

The default source is ``configs/streamlit_ui.yaml`` at the repository root.
Set ``SP_STREAMLIT_UI_CONFIG`` to point at a different YAML file.
"""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_UI_CONFIG: dict[str, Any] = {
    "app": {
        "page_title": "",
        "page_icon": "",
        "layout": "wide",
        "intro": "",
        "nav": {},
    },
    "home": {
        "chat": {
            "modes": [],
            "quick_prompts": [],
            "file_types": [],
            "submit_mode": "disable",
        },
        "workflows": [],
    },
    "production": {
        "profiles": {
            "production_aliases": ["prod", "production", "client"],
            "known": ["development", "demo", "staging", "prod", "production", "client"],
        },
        "statuses": {"ready": "ready", "unavailable": "work_in_progress"},
        "checks": {},
    },
    "schema": {
        "intent_options": [],
        "locale_options": [],
        "export_formats": [],
        "intent_presets": {},
    },
    "database": {
        "intent_options": [],
        "source_options": [],
        "model_type_labels": {},
        "intent_presets": {},
    },
    "pdf": {
        "intent_options": [],
        "extraction_options": [],
        "intent_presets": {},
    },
    "transcript": {
    },
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[4]


def default_ui_config_path() -> Path:
    return _repo_root() / "configs" / "streamlit_ui.yaml"


@lru_cache(maxsize=1)
def load_ui_config() -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_UI_CONFIG)
    config = _merge_yaml_file(config, default_ui_config_path())
    raw_path = os.environ.get("SP_STREAMLIT_UI_CONFIG")
    path = Path(raw_path).expanduser() if raw_path else None
    if path is None or path == default_ui_config_path():
        return config
    return _merge_yaml_file(config, path)


def _merge_yaml_file(config: dict[str, Any], path: Path) -> dict[str, Any]:
    if not path.is_file():
        return config
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Streamlit UI config must be a mapping: {path}")
    return _deep_merge(config, data)


def ui_value(*path: str, default: Any = None) -> Any:
    value: Any = load_ui_config()
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def ui_step(page: str, key: str) -> dict[str, str]:
    value = ui_value(page, "steps", key, default={})
    return dict(value) if isinstance(value, Mapping) else {}
