"""Transactional SQLite output sink."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


class SqliteSink:
    def __init__(self, destination: str | Path):
        self.destination = Path(destination)
        self._tmp = self.destination.with_suffix(".tmp.db")
        self._conn = None

    def begin(self, dataset_id: str) -> None:
        if self._tmp.exists():
            self._tmp.unlink()
        self._conn = sqlite3.connect(self._tmp)

    def write_table(self, table_name: str, frame: pd.DataFrame) -> None:
        frame.to_sql(table_name, self._conn, index=False, if_exists="replace")

    def commit(self) -> None:
        self._conn.commit(); self._conn.close()
        self._tmp.replace(self.destination)

    def rollback(self) -> None:
        if self._conn:
            self._conn.close()
        if self._tmp.exists():
            self._tmp.unlink()
