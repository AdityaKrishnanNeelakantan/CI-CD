"""Product-generation defaults resolved from the persisted settings API."""

from __future__ import annotations

from typing import Any, Protocol


SETTING_KEYS = (
    "generation_mode",
    "default_record_count",
    "privacy_level",
    "default_output_format",
)

DEFAULT_PRODUCT_SETTINGS: dict[str, Any] = {
    "generation_mode": "schema_driven",
    "default_record_count": 100,
    "privacy_level": "standard",
    "default_output_format": "csv",
}

SUPPORTED_GENERATION_MODES = {"schema_driven", "source_driven"}
SUPPORTED_PRIVACY_LEVELS = {"standard", "restricted", "strict"}
SUPPORTED_OUTPUT_FORMATS = {"csv", "parquet", "sqlite", "pdf"}


class ProductSettingsReader(Protocol):
    def read_settings(self, keys: tuple[str, ...] | None = None) -> dict[str, Any]:
        ...


def read_generation_defaults(settings: ProductSettingsReader | None = None) -> dict[str, Any]:
    """Read product defaults from persisted settings, falling back safely."""

    stored: dict[str, Any] = {}
    if settings is not None:
        try:
            stored = settings.read_settings(SETTING_KEYS)
        except Exception:
            stored = {}

    generation_mode = str(stored.get("generation_mode") or DEFAULT_PRODUCT_SETTINGS["generation_mode"])
    if generation_mode not in SUPPORTED_GENERATION_MODES:
        generation_mode = DEFAULT_PRODUCT_SETTINGS["generation_mode"]

    privacy_level = str(stored.get("privacy_level") or DEFAULT_PRODUCT_SETTINGS["privacy_level"])
    if privacy_level not in SUPPORTED_PRIVACY_LEVELS:
        privacy_level = DEFAULT_PRODUCT_SETTINGS["privacy_level"]

    output_format = str(stored.get("default_output_format") or DEFAULT_PRODUCT_SETTINGS["default_output_format"])
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        output_format = DEFAULT_PRODUCT_SETTINGS["default_output_format"]

    try:
        record_count = int(stored.get("default_record_count") or DEFAULT_PRODUCT_SETTINGS["default_record_count"])
    except (TypeError, ValueError):
        record_count = int(DEFAULT_PRODUCT_SETTINGS["default_record_count"])

    return {
        "generation_mode": generation_mode,
        "default_record_count": max(1, record_count),
        "privacy_level": privacy_level,
        "default_output_format": output_format,
    }


def privacy_level_removes_sensitive_information(privacy_level: str) -> bool:
    """Conservative Phase 9 privacy mapping for transcript generation."""

    return str(privacy_level or "standard").lower() in {"standard", "restricted", "strict"}
