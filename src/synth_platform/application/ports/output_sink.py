"""OutputSink port: transactional table output (staging -> commit)."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class OutputSink(Protocol):
    def begin(self, dataset_id: str) -> None: ...

    def write_table(self, table_name: str, frame: pd.DataFrame) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...
