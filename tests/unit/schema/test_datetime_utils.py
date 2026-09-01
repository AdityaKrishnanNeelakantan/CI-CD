"""Tests for datetime sniffing helpers."""

from __future__ import annotations

import warnings

import pandas as pd

from synth_platform.engine.generation.schema.datetime_utils import coerce_datetime_series, looks_like_date_series


def test_looks_like_date_series_rejects_numeric_ids():
    series = pd.Series([1, 2, 3, 4, 5])
    assert looks_like_date_series(series) is False


def test_looks_like_date_series_accepts_iso_dates():
    series = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])
    assert looks_like_date_series(series) is True


def test_coerce_datetime_series_without_inference_warning():
    series = pd.Series(["2024-06-01", "2024-06-02"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        parsed = coerce_datetime_series(series)
    assert parsed.notna().all()
