"""Streamlit session state contains identifiers and paths, never credentials."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DemoState:
    db_run_id: str | None = None
    db_artifact_path: str | None = None
    db_dataset_id: str | None = None
    db_report_id: str | None = None
    pdf_run_id: str | None = None
    pdf_artifact_path: str | None = None
    pdf_dataset_id: str | None = None
    pdf_report_id: str | None = None
    pdf_package_path: str | None = None


def load_state(session_state) -> DemoState:
    raw = session_state.get("demo_state") or {}
    return DemoState(**{k: raw.get(k) for k in DemoState.__dataclass_fields__})


def save_state(session_state, state: DemoState) -> None:
    session_state["demo_state"] = asdict(state)
