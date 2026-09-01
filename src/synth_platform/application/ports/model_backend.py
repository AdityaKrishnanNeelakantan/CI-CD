"""ModelBackend port. Statistical backend implemented; others are scaffold."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from synth_platform.domain.planning.models import LearningTask


class ModelBackend(Protocol):
    backend_name: str

    def supports(self, task: LearningTask) -> bool: ...

    def generate_column(self, task: LearningTask, n: int, seed: int) -> pd.Series: ...
