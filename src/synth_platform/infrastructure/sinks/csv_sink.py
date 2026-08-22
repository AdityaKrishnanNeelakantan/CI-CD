"""Transactional CSV sink: stage then atomically publish."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


class CsvSink:
    def __init__(self, destination: str | Path):
        self.destination = Path(destination)
        self._staging = self.destination.with_suffix(".staging")

    def begin(self, dataset_id: str) -> None:
        if self._staging.exists():
            shutil.rmtree(self._staging)
        self._staging.mkdir(parents=True, exist_ok=True)

    def write_table(self, table_name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(self._staging / f"{table_name}.csv", index=False)

    def commit(self) -> None:
        self.destination.mkdir(parents=True, exist_ok=True)
        for f in self._staging.glob("*.csv"):
            shutil.copy2(f, self.destination / f.name)
        shutil.rmtree(self._staging, ignore_errors=True)

    def rollback(self) -> None:
        shutil.rmtree(self._staging, ignore_errors=True)
