"""Default validation thresholds (pure, tunable)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    categorical_tvd_max: float = 0.15
    numeric_quantile_rmse_max: float = 0.10
    near_dup_max: float = 0.05
    dcr_ratio_min: float = 0.90
    exact_match_max: float = 0.001
    mia_auc_max: float = 0.60
    tstr_ratio_min: float = 0.80
    correlation_retention_min: float = 0.70
