"""Source profile and fit-config caching."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


def _sample_fingerprint(df: pd.DataFrame, *, max_rows: int = 256) -> bytes:
    if df.empty:
        return b"empty"
    sample = df.head(max_rows) if len(df) <= max_rows else df.sample(n=max_rows, random_state=0)
    payload = {
        "columns": list(sample.columns),
        "dtypes": {col: str(sample[col].dtype) for col in sample.columns},
        "rows": int(len(df)),
        "values": sample.fillna("__NULL__").astype(str).values.tolist(),
    }
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def source_dataframe_fingerprint(
    df: pd.DataFrame,
    *,
    profile_max_rows: int,
    fit_max_rows: int,
    model_type: Optional[str],
    seed: Optional[int],
    table_name: str,
) -> str:
    """Stable hash for source profile / fit cache keys."""
    hasher = hashlib.sha256()
    hasher.update(table_name.encode("utf-8"))
    hasher.update(str(profile_max_rows).encode())
    hasher.update(str(fit_max_rows).encode())
    hasher.update(str(model_type or "auto").encode())
    hasher.update(str(seed if seed is not None else "none").encode())
    hasher.update(_sample_fingerprint(df))
    return hasher.hexdigest()


class SourceProfileCache:
    """Disk cache for serialized source profiles keyed by fingerprint."""

    def __init__(self, root: Optional[Union[str, Path]] = None) -> None:
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        self.root = Path(root or cache_home / "synth-platform" / "source_profiles")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.root = Path(tempfile.gettempdir()) / "synth-platform" / "source_profiles"
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, profile_dict: Dict[str, Any]) -> Path:
        path = self._path(key)
        path.write_text(json.dumps(profile_dict, indent=2), encoding="utf-8")
        return path
