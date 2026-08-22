"""Deterministic fingerprints for schema/config content.

Fingerprints must be computed only from structural metadata (schema shape,
config values) and must never include secrets or raw data rows - see
SourceAdapter.discover(), which builds the dict passed to
compute_fingerprint() here from column/table/key metadata only.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def compute_fingerprint(payload: dict[str, Any]) -> str:
    """Return a stable "sha256:<hex>" fingerprint for a JSON-serialisable dict.

    Keys are sorted recursively so the result is independent of dict
    construction order, and separators are fixed so formatting whitespace
    cannot change the hash.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def compute_dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Content fingerprint of a DataFrame's actual values.

    Used to make a training run traceable to the exact data it was trained
    on (the "immutable training-data fingerprint") without persisting the
    data itself - this is a one-way hash, not a copy of the data.
    """
    import pandas as pd

    if df.empty:
        payload = f"empty:{','.join(str(c) for c in df.columns)}".encode()
    else:
        row_hashes = pd.util.hash_pandas_object(df, index=False)
        payload = row_hashes.to_numpy().tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"
