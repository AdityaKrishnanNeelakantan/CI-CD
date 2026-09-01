"""Checkpoint 2: factual, measurable statistics about sampled data.

StructuredProfiler deliberately knows nothing about business meaning - it
never assigns a semantic_type. It records evidence (counts, ranges,
distributions, parse-success rates); Checkpoint 3's SemanticInferenceEngine
is the only place that interprets this evidence into a semantic role. This
module has no Streamlit and no model dependency, so it is usable and
testable in complete isolation from either.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from synth_platform.engine.common.database.core.scaling import merge_chunk_profiles
from synth_platform.engine.common.database.privacy.rare_category import DEFAULT_MINIMUM_SUPPORT, suppress_rare_categories

PROFILER_VERSION = "1.0"


@dataclass(frozen=True)
class ProfilerConfig:
    near_unique_threshold: float = 0.95
    missingness_threshold: float = 0.5
    imbalance_threshold: float = 0.95
    max_category_values: int = 50
    example_count: int = 3
    date_parse_threshold: float = 0.5
    # Export-safe companions to the exact minimum/maximum/category_frequencies
    # below (see their own comments) - not used by anything in the protected
    # training zone, only by src/artifact/builder.py when it decides what a
    # portable generator ZIP is allowed to contain.
    extreme_value_lower_percentile: float = 0.01
    extreme_value_upper_percentile: float = 0.99
    rare_category_minimum_support: int = DEFAULT_MINIMUM_SUPPORT


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    f = float(value)
    return None if math.isnan(f) else f


def _mask_value(value: Any) -> str:
    """Mask a value for display as a representative example.

    Category frequency tables keep raw values (downstream stages need the
    exact category), but individual example values are masked since they
    may echo raw, potentially sensitive, source data into evidence files.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<null>"
    s = str(value)
    if len(s) <= 2:
        return "*" * len(s)
    return s[0] + "*" * (len(s) - 2) + s[-1]


def _infer_physical_dtype(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        # A NULL-bearing integer column upcasts to float64 in pandas; that
        # is a storage artifact, not evidence the column is fractional.
        if (non_null == non_null.round()).all():
            return "integer"
        return "float"
    return "string"


def _profile_column(series: pd.Series, column_name: str, config: ProfilerConfig) -> dict[str, Any]:
    row_count = len(series)
    null_count = int(series.isna().sum())
    null_percentage = round((null_count / row_count) * 100.0, 6) if row_count else 0.0
    non_null = series.dropna()
    distinct_count = int(non_null.nunique())
    distinct_ratio = round(distinct_count / row_count, 6) if row_count else 0.0
    dtype = _infer_physical_dtype(series)

    profile: dict[str, Any] = {
        "name": column_name,
        "row_count": row_count,
        "null_count": null_count,
        "null_percentage": null_percentage,
        "distinct_count": distinct_count,
        "distinct_ratio": distinct_ratio,
        "inferred_dtype": dtype,
        "minimum": None,
        "maximum": None,
        "mean": None,
        "median": None,
        "quantiles": None,
        "string_lengths": None,
        "category_frequencies": None,
        "representative_examples": [],
        # Export-safe companions - see ProfilerConfig's own comment. Never
        # None-but-omitted from a numeric/categorical column that has any
        # data at all, so a consumer can rely on their presence rather than
        # falling back to the exact (unsafe-to-export) fields above.
        "generation_lower_bound": None,
        "generation_upper_bound": None,
        "safe_category_frequencies": None,
        "warnings": [],
    }

    if dtype in ("integer", "float") and not non_null.empty:
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if not numeric.empty:
            profile["minimum"] = _safe_float(numeric.min())
            profile["maximum"] = _safe_float(numeric.max())
            profile["mean"] = _safe_float(numeric.mean())
            profile["median"] = _safe_float(numeric.median())
            quantiles = numeric.quantile(
                [config.extreme_value_lower_percentile, 0.25, 0.5, 0.75, config.extreme_value_upper_percentile]
            )
            profile["quantiles"] = {
                "p25": _safe_float(quantiles.loc[0.25]),
                "p50": _safe_float(quantiles.loc[0.5]),
                "p75": _safe_float(quantiles.loc[0.75]),
            }
            profile["generation_lower_bound"] = _safe_float(quantiles.loc[config.extreme_value_lower_percentile])
            profile["generation_upper_bound"] = _safe_float(quantiles.loc[config.extreme_value_upper_percentile])
        if distinct_count <= config.max_category_values:
            counts = non_null.value_counts()
            profile["category_frequencies"] = {str(k): int(v) for k, v in counts.items()}
            profile["safe_category_frequencies"] = suppress_rare_categories(
                profile["category_frequencies"], config.rare_category_minimum_support
            )

    if dtype == "datetime" and not non_null.empty:
        profile["minimum"] = non_null.min().isoformat()
        profile["maximum"] = non_null.max().isoformat()
        sorted_values = non_null.sort_values()
        lower_index = int(config.extreme_value_lower_percentile * (len(sorted_values) - 1))
        upper_index = int(config.extreme_value_upper_percentile * (len(sorted_values) - 1))
        profile["generation_lower_bound"] = sorted_values.iloc[lower_index].isoformat()
        profile["generation_upper_bound"] = sorted_values.iloc[upper_index].isoformat()

    if dtype == "string" and not non_null.empty:
        lengths = non_null.astype(str).str.len()
        profile["string_lengths"] = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": _safe_float(lengths.mean()),
        }
        # Purely factual parse-rate evidence - not a semantic-type claim.
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        parse_success_rate = float(parsed.notna().sum()) / len(non_null)
        if parse_success_rate >= config.date_parse_threshold:
            profile["date_parse_success_rate"] = round(parse_success_rate, 6)
            if parse_success_rate < 1.0:
                profile["warnings"].append("invalid_dates")
        if distinct_count <= config.max_category_values:
            counts = non_null.value_counts()
            profile["category_frequencies"] = {str(k): int(v) for k, v in counts.items()}
            profile["safe_category_frequencies"] = suppress_rare_categories(
                profile["category_frequencies"], config.rare_category_minimum_support
            )

    if dtype == "boolean" and not non_null.empty:
        counts = non_null.value_counts()
        profile["category_frequencies"] = {str(k): int(v) for k, v in counts.items()}
        # A boolean column only ever has 2 categories - rare-category
        # suppression would either do nothing or destroy the column
        # entirely, so it is deliberately not applied here.
        profile["safe_category_frequencies"] = profile["category_frequencies"]

    examples = list(non_null.unique())[: config.example_count]
    profile["representative_examples"] = [_mask_value(v) for v in examples]

    if row_count > 0 and null_count < row_count and distinct_count <= 1:
        profile["warnings"].append("constant_column")
    if row_count > 0 and distinct_count > 1 and distinct_ratio >= config.near_unique_threshold:
        profile["warnings"].append("near_unique_column")
    if row_count > 0 and null_percentage >= config.missingness_threshold * 100:
        profile["warnings"].append("extreme_missingness")
    if profile["category_frequencies"] and distinct_count > 1:
        top_count = max(profile["category_frequencies"].values())
        if (top_count / row_count) >= config.imbalance_threshold:
            profile["warnings"].append("severe_category_imbalance")

    return profile


def _compute_correlations(
    df: pd.DataFrame, columns: dict[str, Any]
) -> list[dict[str, Any]]:
    numeric_cols = [name for name, profile in columns.items() if profile["inferred_dtype"] in ("integer", "float")]
    if len(numeric_cols) < 2:
        return []

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric_df.corr(numeric_only=True)

    correlations = []
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            value = corr_matrix.loc[col_a, col_b]
            if pd.isna(value):
                continue  # not mathematically valid (e.g. zero variance)
            correlations.append(
                {"column_a": col_a, "column_b": col_b, "correlation": round(float(value), 6)}
            )
    return correlations


def _apply_primary_key_cross_checks(
    columns: dict[str, Any], schema: dict[str, Any] | None, row_count: int
) -> None:
    if not schema:
        return
    for pk_col in schema.get("primary_key", []):
        col_profile = columns.get(pk_col)
        if col_profile is None:
            continue
        if col_profile["null_count"] > 0:
            col_profile["warnings"].append("primary_key_has_nulls")
        if row_count > 0 and col_profile["distinct_count"] < row_count:
            col_profile["warnings"].append("primary_key_duplicates_in_sample")


class StructuredProfiler:
    """Turns a DataFrame + discovered schema into a serialisable profile dict."""

    def __init__(self, config: ProfilerConfig | None = None) -> None:
        self._config = config or ProfilerConfig()

    def profile_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        row_count = len(df)

        columns: dict[str, Any] = {
            str(col): _profile_column(df[col], str(col), self._config) for col in df.columns
        }
        _apply_primary_key_cross_checks(columns, schema, row_count)
        correlations = _compute_correlations(df, columns)

        dataset_warnings: list[str] = []
        if row_count == 0:
            dataset_warnings.append("empty_table")
        for col_name, col_profile in columns.items():
            dataset_warnings.extend(f"{col_name}: {w}" for w in col_profile["warnings"])

        duration_ms = round((time.perf_counter() - start) * 1000, 3)

        return {
            "table_name": table_name,
            "profiled_at": datetime.now(UTC).isoformat(),
            "row_count": row_count,
            "columns": columns,
            "correlations": correlations,
            "warnings": dataset_warnings,
            "execution_metadata": {
                "duration_ms": duration_ms,
                "profiler_version": PROFILER_VERSION,
                "sample_row_count": row_count,
            },
        }

    def profile_table_chunked(
        self,
        chunks: Iterable[pd.DataFrame],
        table_name: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Memory-bounded counterpart to profile_table(): profiles each chunk
        independently, then merges the results, so peak memory is bounded by
        chunk size rather than by the full sampled table.

        Trade-off vs. profile_table(): cross-column correlations require the
        full numeric matrix, so they are intentionally omitted (empty list)
        rather than computed on a partial, misleading basis - a table-level
        warning records this. Column-level statistics are exact except
        distinct_count for high-cardinality columns, which is a capped
        approximation (see merge_chunk_profiles).
        """
        start = time.perf_counter()
        chunk_profiles: list[dict[str, Any]] = []
        chunk_count = 0
        for chunk in chunks:
            chunk_count += 1
            chunk_profiles.append(self.profile_table(chunk, table_name, schema))

        merged = merge_chunk_profiles(chunk_profiles)
        row_count = merged.get("row_count", 0)
        columns = merged.get("columns", {})

        for col_profile in columns.values():
            if col_profile.get("category_frequencies") is not None:
                col_profile["safe_category_frequencies"] = suppress_rare_categories(
                    col_profile["category_frequencies"], self._config.rare_category_minimum_support
                )

        dataset_warnings: list[str] = ["correlations_skipped_in_chunked_mode"]
        if row_count == 0:
            dataset_warnings.append("empty_table")
        for col_name, col_profile in columns.items():
            dataset_warnings.extend(f"{col_name}: {w}" for w in col_profile.get("warnings", []))

        duration_ms = round((time.perf_counter() - start) * 1000, 3)

        return {
            "table_name": table_name,
            "profiled_at": datetime.now(UTC).isoformat(),
            "row_count": row_count,
            "columns": columns,
            "correlations": [],
            "warnings": dataset_warnings,
            "execution_metadata": {
                "duration_ms": duration_ms,
                "profiler_version": PROFILER_VERSION,
                "sample_row_count": row_count,
                "chunk_count": chunk_count,
            },
        }
