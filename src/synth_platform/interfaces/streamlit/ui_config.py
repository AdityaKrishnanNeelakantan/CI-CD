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
        "page_title": "Synthetic Data Twin Platform",
        "page_icon": ":material/data_object:",
        "layout": "wide",
        "intro": "One synthetic-data platform - multiple SSOT input workflows",
        "nav": {
            "Platform": [
                {"key": "home", "file": "home.py", "title": "Home", "icon": ":material/home:", "default": True},
            ],
            "Create": [
                {"key": "schema", "file": "schema_twin.py", "title": "Schema Mode", "icon": ":material/schema:"},
                {"key": "database", "file": "database_twin.py", "title": "Database Twin", "icon": ":material/database:"},
                {"key": "pdf", "file": "pdf_twin.py", "title": "PDF Twin", "icon": ":material/description:"},
                {
                    "key": "transcript",
                    "file": "transcript_twin.py",
                    "title": "Customer Interactions Twin",
                    "icon": ":material/support_agent:",
                },
            ],
        },
    },
    "home": {
        "title": "Synthetic Data Twin Platform",
        "chooser_heading": "What would you like to create?",
        "chooser_copy": "One platform. Multiple SSOT input paths. Pick the source type you have.",
        "workflows": [
            {
                "key": "schema",
                "title": "Schema Mode",
                "body": "Create synthetic data from structure alone.",
                "caption": "SQL DDL -> review -> set row counts -> generate -> validate -> download",
                "page": "schema_twin.py",
                "link_label": "Open Schema Mode",
                "icon": ":material/schema:",
            },
            {
                "key": "database",
                "title": "Database Twin",
                "body": "Train a portable ML twin from an existing database.",
                "caption": "Connect -> profile -> train ML twin -> disconnect -> generate -> validate -> export",
                "page": "database_twin.py",
                "link_label": "Open Database Twin",
                "icon": ":material/database:",
            },
            {
                "key": "pdf",
                "title": "PDF Twin",
                "body": "Create synthetic documents while preserving useful document structure.",
                "caption": "Upload -> understand -> generate -> validate -> download",
                "page": "pdf_twin.py",
                "link_label": "Open PDF Twin",
                "icon": ":material/description:",
            },
            {
                "key": "transcript",
                "title": "Customer Interactions Twin",
                "body": "Create synthetic customer conversations from real interaction examples.",
                "caption": "Paste/upload -> preview -> build twin -> create interactions -> validate",
                "page": "transcript_twin.py",
                "link_label": "Open Customer Interactions Twin",
                "icon": ":material/support_agent:",
            },
        ],
        "controls_heading": "Shared platform controls",
        "controls": [
            {"label": "SSOT paths", "value": "4"},
            {"label": "Output volume", "value": "user-controlled"},
            {"label": "Validation", "value": "common evidence"},
            {"label": "Privacy", "value": "tested safeguards"},
        ],
        "technical_details": (
            "Shared stages behind every input path: discovery, understanding, canonical contract, "
            "entity/dependency representation, generation, validation, and export."
        ),
    },
    "schema": {
        "title": "Schema Mode",
        "caption": "You have the structure of the data you need - generate synthetic relational data without production records.",
        "default_intent": "Development & testing",
        "intent_options": [
            "Development & testing",
            "QA / automated tests",
            "Demonstrations",
            "Data pipeline development",
            "Early project environments",
        ],
        "locale_options": ["en_US", "en_GB", "en_IN", "de_DE", "fr_FR"],
        "export_formats": ["csv", "parquet"],
    },
    "database": {
        "title": "Database Twin",
        "caption": (
            "Learn from an existing database, download a portable trained twin, disconnect the source, "
            "then generate and validate synthetic data from the artifact alone."
        ),
        "default_intent": "Development & testing",
        "intent_options": [
            "Development & testing",
            "QA / automated tests",
            "Demonstrations",
            "Analytics prototyping",
            "Safe sharing with partners",
        ],
        "source_options": ["Use a generated sample database", "Upload a SQLite file"],
        "model_type_labels": {
            "safe_gaussian_copula": "Standard ML twin",
            "dp_gaussian_copula": "ML twin with differential privacy",
            "sdv_gaussian_copula": "SDV Gaussian copula",
        },
    },
    "pdf": {
        "title": "PDF Twin",
        "caption": (
            "Upload a document and create a synthetic view that preserves useful document structure. "
            "Fields are replaced with generated values."
        ),
        "default_intent": "Document testing",
        "default_extraction_engine": "Advanced document extraction",
        "intent_options": ["Document testing", "QA / automation", "Demonstrations", "Privacy-safe sharing"],
        "extraction_options": ["Advanced document extraction", "Automatic (native/OCR)"],
    },
    "transcript": {
        "title": "Customer Interactions Twin",
        "caption": "Create synthetic customer interactions from a real conversation example.",
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
    raw_path = os.environ.get("SP_STREAMLIT_UI_CONFIG")
    path = Path(raw_path).expanduser() if raw_path else default_ui_config_path()
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
