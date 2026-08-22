"""DataFrameExtractionPass: lift in-memory frames + a target schema into the IR."""
from __future__ import annotations

import pandas as pd

from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.domain.extraction.lowering import lower_target_schema
from synth_platform.domain.extraction.target_schema import TargetSchema


class DataFrameExtractionPass:
    kind = "dataframe"

    def __init__(self, frames: dict[str, pd.DataFrame], target: TargetSchema):
        self.frames = frames
        self.target = target

    def extract(self) -> RelationalDataset:
        schema = lower_target_schema(self.target, self.kind)
        return RelationalDataset(schema=schema, tables=dict(self.frames)).finalize_counts()
