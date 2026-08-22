"""Scaling layer: chunked and partitioned processing for large datasets.

Profiling and synthesis currently operate on sampled, in-memory data.
This module provides chunked iteration utilities so callers can process
very large datasets (e.g. 10M+ rows) without holding everything in
memory at once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import pandas as pd

T = TypeVar("T")

DEFAULT_CHUNK_SIZE = 10_000


def iter_dataframe_chunks(
    df: pd.DataFrame, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Iterator[pd.DataFrame]:
    """Yield successive chunks of a DataFrame."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]


def process_in_chunks(  # noqa: UP047 - PEP 695 generic syntax is a typing-style
    # modernization, not a correctness fix; left as TypeVar-based generics to
    # avoid an unrelated signature-syntax change during a CI/CD lint rollout.
    df: pd.DataFrame,
    processor: Callable[[pd.DataFrame], T],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    aggregator: Callable[[list[T]], T] | None = None,
) -> T | list[T]:
    """Apply a processor function to DataFrame chunks.

    If ``aggregator`` is provided, chunk results are combined into a
    single return value; otherwise a list of per-chunk results is returned.
    """
    results: list[T] = []
    for chunk in iter_dataframe_chunks(df, chunk_size):
        results.append(processor(chunk))

    if aggregator is not None:
        return aggregator(results)
    return results


def merge_chunk_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-chunk table profiles (as produced by
    ``StructuredProfiler.profile_table`` on each chunk) into one table-level
    profile.

    Exact across chunks: row/null counts, min/max, category frequencies,
    and mean (reconstructed via a count-weighted average, which is exact
    for a linear statistic like the mean - unlike median/quantiles below).

    Approximate or omitted, and always flagged with a warning rather than
    silently wrong:
    - ``distinct_count`` is exact only when the merged ``category_frequencies``
      is available (bounded-cardinality columns); for high-cardinality
      columns it is a row_count-capped approximation (sum of per-chunk
      distinct counts, which double-counts values recurring across
      chunks) - exact cross-chunk cardinality would require holding every
      distinct value in memory, which is exactly what chunking avoids.
    - ``median``, ``quantiles``, ``generation_lower_bound`` and
      ``generation_upper_bound`` cannot be reconstructed from per-chunk
      quantiles without the full data (unlike the mean, a quantile is not
      a linear statistic), so they are set to None with an
      ``quantile_stats_skipped_in_chunked_mode`` warning rather than
      silently returning only the first chunk's values.
    """
    if not profiles:
        return {}

    merged: dict[str, Any] = {"row_count": 0, "columns": {}}
    approx_distinct_sums: dict[str, int] = {}
    weighted_mean_sums: dict[str, float] = {}
    weighted_string_len_sums: dict[str, float] = {}
    weighted_parse_rate_sums: dict[str, float] = {}
    non_null_sums: dict[str, int] = {}
    had_quantile_stats: dict[str, bool] = {}
    had_string_lengths: dict[str, bool] = {}
    had_parse_rate: dict[str, bool] = {}

    for profile in profiles:
        merged["row_count"] += profile.get("row_count", 0)
        for col_name, col_stats in profile.get("columns", {}).items():
            non_null = col_stats.get("row_count", 0) - col_stats.get("null_count", 0)
            non_null_sums[col_name] = non_null_sums.get(col_name, 0) + non_null
            approx_distinct_sums[col_name] = approx_distinct_sums.get(
                col_name, 0
            ) + col_stats.get("distinct_count", 0)
            if col_stats.get("mean") is not None:
                weighted_mean_sums[col_name] = weighted_mean_sums.get(col_name, 0.0) + (
                    col_stats["mean"] * non_null
                )
            if col_stats.get("string_lengths") is not None:
                had_string_lengths[col_name] = True
                weighted_string_len_sums[col_name] = weighted_string_len_sums.get(
                    col_name, 0.0
                ) + (col_stats["string_lengths"]["mean"] * non_null)
            if col_stats.get("date_parse_success_rate") is not None:
                had_parse_rate[col_name] = True
                weighted_parse_rate_sums[col_name] = weighted_parse_rate_sums.get(
                    col_name, 0.0
                ) + (col_stats["date_parse_success_rate"] * non_null)
            if col_stats.get("median") is not None or col_stats.get("quantiles") is not None:
                had_quantile_stats[col_name] = True

            if col_name not in merged["columns"]:
                merged["columns"][col_name] = dict(col_stats)
                continue
            existing = merged["columns"][col_name]
            existing["null_count"] = existing.get("null_count", 0) + col_stats.get("null_count", 0)
            if col_stats.get("minimum") is not None:
                existing["minimum"] = (
                    col_stats["minimum"]
                    if existing.get("minimum") is None
                    else min(existing["minimum"], col_stats["minimum"])
                )
            if col_stats.get("maximum") is not None:
                existing["maximum"] = (
                    col_stats["maximum"]
                    if existing.get("maximum") is None
                    else max(existing["maximum"], col_stats["maximum"])
                )
            if col_stats.get("string_lengths") is not None:
                existing_lengths = existing.get("string_lengths") or dict(col_stats["string_lengths"])
                existing_lengths["min"] = min(existing_lengths["min"], col_stats["string_lengths"]["min"])
                existing_lengths["max"] = max(existing_lengths["max"], col_stats["string_lengths"]["max"])
                existing["string_lengths"] = existing_lengths
            for cat, freq in (col_stats.get("category_frequencies") or {}).items():
                existing.setdefault("category_frequencies", {})
                existing["category_frequencies"][cat] = (
                    existing["category_frequencies"].get(cat, 0) + freq
                )

    row_count = merged["row_count"]
    for col_name, existing in merged["columns"].items():
        existing["warnings"] = list(existing.get("warnings") or [])
        if existing.get("category_frequencies") is not None:
            existing["distinct_count"] = len(existing["category_frequencies"])
        else:
            existing["distinct_count"] = min(approx_distinct_sums.get(col_name, 0), row_count)
            if "approximate_distinct_count_chunked" not in existing["warnings"]:
                existing["warnings"].append("approximate_distinct_count_chunked")
        existing["distinct_ratio"] = round(existing["distinct_count"] / row_count, 6) if row_count else 0.0
        existing["null_percentage"] = (
            round((existing["null_count"] / row_count) * 100.0, 6) if row_count else 0.0
        )
        existing["row_count"] = row_count

        non_null = non_null_sums.get(col_name, 0)
        existing["mean"] = (
            weighted_mean_sums[col_name] / non_null if col_name in weighted_mean_sums and non_null else None
        )
        if had_string_lengths.get(col_name) and non_null:
            lengths = existing.get("string_lengths") or {}
            existing["string_lengths"] = {
                "min": lengths.get("min"),
                "max": lengths.get("max"),
                "mean": weighted_string_len_sums[col_name] / non_null,
            }
        if had_parse_rate.get(col_name) and non_null:
            existing["date_parse_success_rate"] = round(weighted_parse_rate_sums[col_name] / non_null, 6)

        if had_quantile_stats.get(col_name):
            existing["median"] = None
            existing["quantiles"] = None
            existing["generation_lower_bound"] = None
            existing["generation_upper_bound"] = None
            if "quantile_stats_skipped_in_chunked_mode" not in existing["warnings"]:
                existing["warnings"].append("quantile_stats_skipped_in_chunked_mode")

    return merged


def estimate_chunk_count(total_rows: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    """Return the number of chunks needed for a given row count."""
    if total_rows <= 0:
        return 0
    return (total_rows + chunk_size - 1) // chunk_size
