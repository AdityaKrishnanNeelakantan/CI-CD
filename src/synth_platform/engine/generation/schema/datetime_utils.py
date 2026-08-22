"""Shared datetime sniffing and coercion helpers."""

from __future__ import annotations

import re
import warnings

import pandas as pd

_DATELIKE_RE = re.compile(
    r"^("
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r"|\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"
    r").*$"
)


def looks_like_date_series(
    series: pd.Series,
    *,
    threshold: float = 0.7,
    sample_size: int = 50,
) -> bool:
    """Return True when a series plausibly contains date/datetime values."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    sample = non_null.astype(str).head(sample_size)
    if float(sample.str.match(_DATELIKE_RE).mean()) < threshold:
        return False

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= threshold


def coerce_datetime_series(series: pd.Series) -> pd.Series:
    """Parse a series to datetimes without pandas format-inference warnings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
        return pd.to_datetime(series, errors="coerce", format="mixed")


_COMMON_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
)


def infer_datetime_format(series: pd.Series) -> str:
    """Infer a strftime pattern suitable for SDV datetime metadata."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series.dropna()
        if parsed.empty:
            return "%Y-%m-%d"
        if (parsed.dt.time == pd.Timestamp(0).time()).all():
            return "%Y-%m-%d"
        if (parsed.dt.second != 0).any():
            return "%Y-%m-%d %H:%M:%S"
        return "%Y-%m-%d %H:%M"

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return "%Y-%m-%d"

    sample = non_null.head(200)
    best_fmt = "%Y-%m-%d"
    best_score = -1.0
    for fmt in _COMMON_DATETIME_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        score = float(parsed.notna().mean())
        if score > best_score:
            best_score = score
            best_fmt = fmt
        if score >= 0.99:
            break

    if best_score < 0.7:
        parsed = coerce_datetime_series(series).dropna()
        if parsed.empty:
            return "%Y-%m-%d"
        if (parsed.dt.time == pd.Timestamp(0).time()).all():
            return "%Y-%m-%d"
        return "%Y-%m-%d %H:%M:%S"

    return best_fmt
