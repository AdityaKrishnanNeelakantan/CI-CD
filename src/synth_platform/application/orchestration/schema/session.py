"""Streamlit session memory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


LARGE_SESSION_KEYS = (
    "source_df",
    "source_full_tables",
    "full_tables",
    "sample_tables",
    "source_sample_tables",
)


def clear_large_session_objects(session_state: Any, *, keep: Optional[Iterable[str]] = None) -> None:
    """Drop large DataFrames from Streamlit session state."""
    keep_set = set(keep or ())
    for key in LARGE_SESSION_KEYS:
        if key in keep_set:
            continue
        value = session_state.get(key)
        if isinstance(value, dict) and value and isinstance(next(iter(value.values()), None), pd.DataFrame):
            session_state.pop(key, None)
        elif isinstance(value, pd.DataFrame):
            session_state.pop(key, None)
    session_state.pop("source_df", None)


def store_artifact_reference(session_state: Any, *, prefix: str, artifact_dir: Path) -> None:
    session_state[f"{prefix}_artifact_dir"] = str(artifact_dir)


def get_artifact_for_export(session_state: Any, *, prefix: str) -> Optional[Path]:
    raw = session_state.get(f"{prefix}_artifact_dir")
    return Path(raw) if raw else None


def store_preview_sample_only(
    session_state: Any,
    *,
    key: str,
    tables: Dict[str, pd.DataFrame],
    max_rows: int = 500,
) -> None:
    """Keep only a bounded preview sample in session state."""
    session_state[key] = {name: df.head(max_rows).copy() for name, df in tables.items()}


def render_preview_sample_only(tables: Dict[str, pd.DataFrame], *, max_rows: int = 200) -> Dict[str, pd.DataFrame]:
    return {name: df.head(max_rows) for name, df in tables.items()}
