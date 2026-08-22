"""Mixed-type row -> float-vector encoder, fit on the real train split."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _as_float(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.view("int64").astype(float).to_numpy()
    return pd.to_numeric(s, errors="coerce").astype(float).to_numpy()


@dataclass
class FeatureEncoder:
    numeric_cols: list = field(default_factory=list)
    numeric_mean: dict = field(default_factory=dict)
    numeric_std: dict = field(default_factory=dict)
    cat_cols: list = field(default_factory=list)
    cat_levels: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, exclude: set) -> "FeatureEncoder":
        enc = cls()
        for col in df.columns:
            if col in exclude:
                continue
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                v = _as_float(s)
                enc.numeric_cols.append(col)
                enc.numeric_mean[col] = float(np.nanmean(v)) if v.size else 0.0
                std = float(np.nanstd(v)) if v.size else 0.0
                enc.numeric_std[col] = std if std > 1e-9 else 1.0
            else:
                levels = sorted(s.dropna().astype(str).unique().tolist())
                if len(levels) > 32:
                    levels = sorted(s.dropna().astype(str).value_counts().head(32).index.tolist())
                enc.cat_cols.append(col)
                enc.cat_levels[col] = levels
        return enc

    @property
    def width(self) -> int:
        return len(self.numeric_cols) + sum(len(v) for v in self.cat_levels.values())

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        blocks = []
        for col in self.numeric_cols:
            v = (_as_float(df[col]) if col in df.columns
                 else np.full(n, self.numeric_mean[col]))
            v = np.nan_to_num(v, nan=self.numeric_mean[col])
            blocks.append(((v - self.numeric_mean[col]) / self.numeric_std[col]).reshape(-1, 1))
        for col in self.cat_cols:
            levels = self.cat_levels[col]
            oh = np.zeros((n, len(levels)))
            if col in df.columns and levels:
                idx = {lv: i for i, lv in enumerate(levels)}
                for r, val in enumerate(df[col].astype(str).to_numpy()):
                    j = idx.get(val)
                    if j is not None:
                        oh[r, j] = 1.0
            blocks.append(oh)
        return np.hstack(blocks) if blocks else np.zeros((n, 0))
