"""Build a SQLite database from uploaded CSV table files.

Each CSV becomes one table (filename stem = table name). Used by Database Twin
so CSV folders are a first-class connect path without requiring a pre-built DB.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import pandas as pd

_SAFE_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _table_name_from_filename(name: str) -> str:
    stem = Path(name).stem.strip().replace("-", "_").replace(" ", "_")
    stem = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    if not stem or not _SAFE_TABLE.match(stem):
        stem = f"table_{abs(hash(name)) % 10_000_000}"
    return stem.lower()


def build_sqlite_from_csv_uploads(
    destination: Path,
    uploads: Sequence[tuple[str, bytes | BinaryIO]],
) -> list[str]:
    """Write ``destination`` SQLite with one table per CSV upload.

    Returns the list of created table names.
    """
    if not uploads:
        raise ValueError("At least one CSV file is required.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    tables: list[str] = []
    conn = sqlite3.connect(str(destination))
    try:
        used: set[str] = set()
        for filename, payload in uploads:
            raw = payload.read() if hasattr(payload, "read") else payload  # type: ignore[union-attr]
            assert isinstance(raw, (bytes, bytearray))
            df = pd.read_csv(pd.io.common.BytesIO(raw))
            if df.empty and len(df.columns) == 0:
                raise ValueError(f"{filename}: CSV has no columns.")
            base = _table_name_from_filename(filename)
            table = base
            n = 2
            while table in used:
                table = f"{base}_{n}"
                n += 1
            used.add(table)
            df.to_sql(table, conn, index=False, if_exists="replace")
            tables.append(table)
        conn.commit()
    finally:
        conn.close()
    return tables
