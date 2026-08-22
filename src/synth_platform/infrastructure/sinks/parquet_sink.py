"""Transactional directory-based Parquet sink."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


class ParquetSink:
    def __init__(self, destination: str | Path):
        self.destination = Path(destination)
        self.staging = self.destination.with_name(self.destination.name + ".staging")

    def begin(self, dataset_id: str) -> None:
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.mkdir(parents=True, exist_ok=True)

    def write_table(self, table_name: str, frame: pd.DataFrame) -> None:
        try:
            frame.to_parquet(self.staging / f"{table_name}.parquet", index=False,
                             engine="pyarrow")
        except ImportError as exc:
            raise RuntimeError("Parquet output requires the pyarrow optional dependency") from exc

    def commit(self) -> None:
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.staging.replace(self.destination)

    def rollback(self) -> None:
        shutil.rmtree(self.staging, ignore_errors=True)
