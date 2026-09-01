"""Intent -> generation presets for Schema / Database / PDF twins.

Presentation helpers return concrete defaults the UI applies. Algorithms stay
in backend packages; this module only maps user intent to knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


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


_SCHEMA: Mapping[str, SchemaIntentPreset] = {
    "Development & testing": SchemaIntentPreset(
        100, "en_US", False, "Balanced row counts for local development."
    ),
    "QA / automated tests": SchemaIntentPreset(
        50, "en_US", False, "Smaller, fast suites for automated QA."
    ),
    "Demonstrations": SchemaIntentPreset(
        200, "en_US", False, "Larger samples that look good in demos."
    ),
    "Data pipeline development": SchemaIntentPreset(
        500, "en_US", False, "Higher volume for pipeline soak tests."
    ),
    "Early project environments": SchemaIntentPreset(
        100, "en_US", False, "Starter volumes for new environments."
    ),
}

_DATABASE: Mapping[str, DatabaseIntentPreset] = {
    "Development & testing": DatabaseIntentPreset(
        "safe_gaussian_copula", 1.0, 400, "en_US", "Fidelity-focused twin for engineering work."
    ),
    "QA / automated tests": DatabaseIntentPreset(
        "safe_gaussian_copula", 1.0, 200, "en_US", "Faster sample DB defaults for QA loops."
    ),
    "Demonstrations": DatabaseIntentPreset(
        "safe_gaussian_copula", 1.0, 600, "en_US", "Richer sample defaults for demos."
    ),
    "Analytics prototyping": DatabaseIntentPreset(
        "safe_gaussian_copula", 1.0, 800, "en_US", "Larger samples for analytics sketches."
    ),
    "Safe sharing with partners": DatabaseIntentPreset(
        "dp_gaussian_copula", 1.0, 400, "en_US", "Prefer differential privacy when sharing externally."
    ),
}

_PDF: Mapping[str, PdfIntentPreset] = {
    "Document testing": PdfIntentPreset("en_US", False, "Deterministic local values for test docs."),
    "QA / automation": PdfIntentPreset("en_US", False, "Stable seeds/locale for automation."),
    "Demonstrations": PdfIntentPreset("en_US", False, "Readable synthetic documents for demos."),
    "Privacy-safe sharing": PdfIntentPreset(
        "en_US", False, "Privacy-first defaults; LLM narrative stays off unless you enable it."
    ),
}


def schema_intent_preset(intent: str) -> SchemaIntentPreset:
    return _SCHEMA.get(intent, _SCHEMA["Development & testing"])


def database_intent_preset(intent: str) -> DatabaseIntentPreset:
    return _DATABASE.get(intent, _DATABASE["Development & testing"])


def pdf_intent_preset(intent: str) -> PdfIntentPreset:
    return _PDF.get(intent, _PDF["Document testing"])
