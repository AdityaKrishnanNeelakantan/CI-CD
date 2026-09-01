"""Export helpers for pipeline artifacts."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import pandas as pd


def make_zip_from_paths(
    table_paths: Mapping[str, Union[str, Path]],
    *,
    report_path: Optional[Union[str, Path]] = None,
    report_payload: Optional[Dict[str, Any]] = None,
    report_name: str = "validation_report.json",
) -> bytes:
    """Package on-disk table exports and optional report into an in-memory ZIP."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for table_name, raw_path in table_paths.items():
            path = Path(raw_path)
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            archive_name = f"{table_name}{suffix or '.csv'}"
            zf.write(path, arcname=archive_name)
        if report_path is not None:
            path = Path(report_path)
            if path.exists():
                zf.write(path, arcname=report_name)
        elif report_payload is not None:
            zf.writestr(report_name, json.dumps(report_payload, indent=2, default=str))
    return buffer.getvalue()


def read_preview_from_export(path: Union[str, Path], *, max_rows: int = 500) -> pd.DataFrame:
    """Load a bounded preview sample from a CSV or Parquet export."""
    file_path = Path(path)
    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path).head(max_rows)
    return pd.read_csv(file_path, nrows=max_rows)
