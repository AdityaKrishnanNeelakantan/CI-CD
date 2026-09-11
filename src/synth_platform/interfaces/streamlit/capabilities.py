"""Streamlit hand-off mapping for application capability IDs."""

from __future__ import annotations

from pathlib import Path

from synth_platform.application.coordinator import CapabilityId

_PAGES_DIR = Path(__file__).resolve().parent / "pages"
_CAPABILITY_PAGES = {
    CapabilityId.SCHEMA_TWIN: _PAGES_DIR / "schema_twin.py",
    CapabilityId.DATABASE_TWIN: _PAGES_DIR / "database_twin.py",
    CapabilityId.DOCUMENT_TWIN: _PAGES_DIR / "pdf_twin.py",
    CapabilityId.INTERACTION_TWIN: _PAGES_DIR / "interaction_twin.py",
}


def capability_page_path(capability_id: CapabilityId) -> str | None:
    """Return an existing Streamlit workflow page for an available capability."""

    page = _CAPABILITY_PAGES.get(capability_id)
    return str(page) if page is not None else None
