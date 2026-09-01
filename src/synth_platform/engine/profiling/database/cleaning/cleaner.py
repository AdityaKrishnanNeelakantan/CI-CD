"""Pre-training cleaning, informed by the approved semantic types.

Runs after Checkpoint 3's approved dataset_contract.json exists. Cleaning
strategy depends entirely on each column's approved semantic_type - you
cannot correctly decide how to handle a missing value or an outlier until
you know whether a column is a continuous measurement, a bounded category,
or an identifier that must never be invented.

Every action taken is one of a small, fixed set of explicitly documented
strategies (median imputation, mode imputation, IQR outlier flagging,
datetime coercion) - never a per-column ad hoc decision. Columns that are
not yet approved (still "proposed" or "review_required") are excluded
rather than guessed at, matching "downstream code reads only approved
semantic metadata."
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

APPROVED_STATUS = "approved"


@dataclass(frozen=True)
class CleaningConfig:
    numerical_imputation_strategy: str = "median"  # or "mean"
    category_imputation_strategy: str = "mode"  # or "sentinel"
    category_missing_sentinel: str = "MISSING"
    outlier_iqr_multiplier: float = 1.5
    strip_whitespace: bool = True


def _empty_column_entry(semantic_type: str, nulls_before: int) -> dict[str, Any]:
    return {
        "semantic_type": semantic_type,
        "action": "no_modification",
        "nulls_before": nulls_before,
        "nulls_after": nulls_before,
        "nulls_imputed": 0,
        "outliers_flagged": None,
        "duplicate_value_count": None,
        "coercion_failures": None,
    }


def _strip_if_string(series: pd.Series, config: CleaningConfig) -> pd.Series:
    # pandas' string storage dtype varies by version (legacy "object" vs.
    # the newer dedicated "str"/"string" dtypes), so check semantically
    # rather than comparing to a single dtype object - and only strip when
    # values are actually text, never for object-dtype columns holding
    # non-string values (e.g. Python bool/None).
    is_textual = pd.api.types.is_string_dtype(series) or (
        series.dtype == object and series.dropna().map(type).eq(str).all()
    )
    if not config.strip_whitespace or not is_textual:
        return series
    stripped = series.astype(str).str.strip()
    return series.where(series.isna(), stripped)


class DataCleaner:
    """Cleans one table's sample according to its approved dataset contract."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self._config = config or CleaningConfig()

    def clean_table(
        self, df: pd.DataFrame, table_name: str, table_contract: dict[str, Any]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        contract_columns = table_contract.get("columns", {})
        row_count = len(df)
        duplicate_row_count = int(df.duplicated().sum()) if row_count else 0

        cleaned: dict[str, pd.Series] = {}
        column_reports: dict[str, Any] = {}
        excluded_columns: list[dict[str, str]] = []

        for column_name in df.columns:
            column_contract = contract_columns.get(column_name)
            if column_contract is None:
                excluded_columns.append({"column": column_name, "reason": "not_in_contract"})
                continue
            if column_contract.get("inference_status") != APPROVED_STATUS:
                excluded_columns.append(
                    {
                        "column": column_name,
                        "reason": f"inference_status={column_contract.get('inference_status')}",
                    }
                )
                continue

            semantic_type = column_contract["semantic_type"]
            cleaned_series, entry = self._clean_column(df[column_name], semantic_type)
            cleaned[column_name] = cleaned_series
            column_reports[column_name] = entry

        cleaned_df = pd.DataFrame(cleaned, index=df.index)

        warnings: list[str] = []
        if row_count == 0:
            warnings.append("empty_table")

        report = {
            "table_name": table_name,
            "cleaned_at": datetime.now(UTC).isoformat(),
            "row_count": row_count,
            "duplicate_row_count": duplicate_row_count,
            "excluded_columns": excluded_columns,
            "columns": column_reports,
            "warnings": warnings,
        }
        return cleaned_df, report

    def _clean_column(self, series: pd.Series, semantic_type: str) -> tuple[pd.Series, dict[str, Any]]:
        nulls_before = int(series.isna().sum())
        entry = _empty_column_entry(semantic_type, nulls_before)
        config = self._config

        if semantic_type == "identifier":
            # An identifier must never be invented or overwritten - only
            # report facts about it (duplicates, nulls), never modify values
            # beyond cosmetic whitespace trimming.
            non_null = series.dropna()
            entry["duplicate_value_count"] = int(non_null.duplicated().sum())
            return _strip_if_string(series, config), entry

        if semantic_type == "numerical":
            numeric = pd.to_numeric(series, errors="coerce")
            coercion_failures = int(numeric.isna().sum()) - nulls_before
            entry["coercion_failures"] = max(coercion_failures, 0)
            non_null = numeric.dropna()

            if non_null.empty:
                entry["nulls_after"] = int(numeric.isna().sum())
                entry["outliers_flagged"] = 0
                return numeric, entry

            fill_value = (
                float(non_null.mean())
                if config.numerical_imputation_strategy == "mean"
                else float(non_null.median())
            )
            entry["action"] = f"imputed_{config.numerical_imputation_strategy}"
            cleaned = numeric.fillna(fill_value)
            entry["nulls_imputed"] = int(numeric.isna().sum())
            entry["nulls_after"] = 0

            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - config.outlier_iqr_multiplier * iqr
            upper_bound = q3 + config.outlier_iqr_multiplier * iqr
            entry["outliers_flagged"] = int(((non_null < lower_bound) | (non_null > upper_bound)).sum())
            return cleaned, entry

        if semantic_type in ("category", "boolean"):
            cleaned = _strip_if_string(series, config)
            non_null = cleaned.dropna()
            if nulls_before > 0 and not non_null.empty:
                if config.category_imputation_strategy == "sentinel":
                    category_fill_value: Any = config.category_missing_sentinel
                    entry["action"] = "imputed_sentinel"
                else:
                    category_fill_value = non_null.mode(dropna=True).iloc[0]
                    entry["action"] = "imputed_mode"
                with pd.option_context("future.no_silent_downcasting", True):
                    cleaned = cleaned.fillna(category_fill_value).infer_objects(copy=False)
                entry["nulls_imputed"] = nulls_before
                entry["nulls_after"] = 0
            return cleaned, entry

        if semantic_type == "datetime":
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            nulls_after = int(parsed.isna().sum())
            entry["coercion_failures"] = max(nulls_after - nulls_before, 0)
            entry["nulls_after"] = nulls_after
            entry["action"] = "coerced_datetime"
            return parsed, entry

        if semantic_type in ("email", "free_text"):
            cleaned = _strip_if_string(series, config)
            entry["action"] = "whitespace_stripped" if config.strip_whitespace else "no_modification"
            return cleaned, entry

        # An approved contract should never contain an unknown semantic
        # type, but fail safe (no modification) rather than guess a strategy.
        return series, entry

    def clean_table_chunked(
        self,
        chunks: Iterable[pd.DataFrame],
        table_name: str,
        table_contract: dict[str, Any],
    ) -> dict[str, Any]:
        """Memory-bounded counterpart to ``clean_table`` for large samples.

        Only returns the report - not a cleaned DataFrame - because
        ``run_cleaning`` (the only caller) already discards ``clean_table``'s
        cleaned frame and persists just the report; so the chunked path never
        needs to hold a full concatenated result in memory either.

        Every reported statistic here is exactly reproducible from streaming
        a single pass over ``chunks`` (no need to re-read the source a second
        time):
        - Counts (nulls, coercion failures) are simple sums across chunks.
        - Identifier duplicate counts and whole-row duplicate counts are
          tracked with running sets/dicts bounded by distinct value/row
          count, not total row count.
        - Category/boolean mode imputation only needs to know *whether* any
          non-null value exists (to decide if imputation fires at all) - the
          report never records the actual imputed value, so no merged
          value/frequency accumulator is needed for this type, unlike
          profiling's category_frequencies.
        - Numerical imputation (median/mean) and IQR outlier bounds are the
          one case that is NOT chunk-mergeable after the fact (median and
          quantiles are not linear statistics, unlike a sum/mean) - this
          method accumulates just that one column's non-null values across
          chunks and computes the exact global statistic once, after the
          full pass - a genuine, bounded reduction (one column's data, not
          the whole wide table) rather than a silently wrong per-chunk
          approximation.

        Whole-row duplicate detection uses a 64-bit row hash
        (``pandas.util.hash_pandas_object``) rather than storing raw rows,
        so it is collision-resistant but not collision-proof; at realistic
        sample sizes (thousands-to-low-millions of rows) this is negligible,
        but it is a real difference from ``clean_table``'s exact
        ``DataFrame.duplicated()``, documented here rather than silently
        assumed.
        """
        contract_columns = table_contract.get("columns", {})
        approved_columns: dict[str, str] | None = None
        excluded_columns: list[dict[str, str]] = []
        accumulators: dict[str, dict[str, Any]] = {}
        row_count = 0
        seen_row_hashes: dict[int, int] = {}
        duplicate_row_count = 0

        for chunk in chunks:
            if approved_columns is None:
                approved_columns, excluded_columns = self._resolve_approved_columns(
                    chunk, contract_columns
                )
                accumulators = {
                    name: self._new_accumulator(semantic_type)
                    for name, semantic_type in approved_columns.items()
                }

            row_count += len(chunk)
            if len(chunk):
                for row_hash in pd.util.hash_pandas_object(chunk, index=False):
                    prior = seen_row_hashes.get(row_hash, 0)
                    if prior:
                        duplicate_row_count += 1
                    seen_row_hashes[row_hash] = prior + 1

            for column_name, semantic_type in (approved_columns or {}).items():
                self._accumulate_column(chunk[column_name], semantic_type, accumulators[column_name])

        column_reports = {
            name: self._finalize_column_report(semantic_type, accumulators[name])
            for name, semantic_type in (approved_columns or {}).items()
        }

        warnings: list[str] = ["chunked_cleaning_mode"]
        if row_count == 0:
            warnings.append("empty_table")

        return {
            "table_name": table_name,
            "cleaned_at": datetime.now(UTC).isoformat(),
            "row_count": row_count,
            "duplicate_row_count": duplicate_row_count,
            "excluded_columns": excluded_columns,
            "columns": column_reports,
            "warnings": warnings,
        }

    def _resolve_approved_columns(
        self, sample_chunk: pd.DataFrame, contract_columns: dict[str, Any]
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        approved: dict[str, str] = {}
        excluded: list[dict[str, str]] = []
        for column_name in sample_chunk.columns:
            column_contract = contract_columns.get(column_name)
            if column_contract is None:
                excluded.append({"column": column_name, "reason": "not_in_contract"})
                continue
            if column_contract.get("inference_status") != APPROVED_STATUS:
                excluded.append(
                    {
                        "column": column_name,
                        "reason": f"inference_status={column_contract.get('inference_status')}",
                    }
                )
                continue
            approved[column_name] = column_contract["semantic_type"]
        return approved, excluded

    def _new_accumulator(self, semantic_type: str) -> dict[str, Any]:
        if semantic_type == "identifier":
            return {"seen_values": set(), "duplicate_value_count": 0}
        if semantic_type == "numerical":
            return {"non_null_chunks": [], "coercion_failures": 0, "nulls_before": 0}
        if semantic_type in ("category", "boolean"):
            return {"nulls_before": 0, "any_non_null": False}
        if semantic_type == "datetime":
            return {"nulls_before": 0, "nulls_after": 0}
        return {"nulls_before": 0}

    def _accumulate_column(self, series: pd.Series, semantic_type: str, acc: dict[str, Any]) -> None:
        config = self._config
        nulls_before = int(series.isna().sum())

        if semantic_type == "identifier":
            for value in series.dropna():
                if value in acc["seen_values"]:
                    acc["duplicate_value_count"] += 1
                else:
                    acc["seen_values"].add(value)
            return

        if semantic_type == "numerical":
            acc["nulls_before"] += nulls_before
            numeric = pd.to_numeric(series, errors="coerce")
            acc["coercion_failures"] += max(int(numeric.isna().sum()) - nulls_before, 0)
            non_null = numeric.dropna()
            if not non_null.empty:
                acc["non_null_chunks"].append(non_null)
            return

        if semantic_type in ("category", "boolean"):
            acc["nulls_before"] += nulls_before
            stripped = _strip_if_string(series, config)
            non_null = stripped.dropna()
            if not non_null.empty:
                acc["any_non_null"] = True
            return

        if semantic_type == "datetime":
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            acc["nulls_before"] += nulls_before
            acc["nulls_after"] += int(parsed.isna().sum())
            return

        # email/free_text/unknown: no global statistic needed, only nulls_before
        # is tracked (matches _empty_column_entry's defaults for these types).
        acc["nulls_before"] += nulls_before

    def _finalize_column_report(self, semantic_type: str, acc: dict[str, Any]) -> dict[str, Any]:
        config = self._config
        nulls_before = acc.get("nulls_before", 0)
        entry = _empty_column_entry(semantic_type, nulls_before)

        if semantic_type == "identifier":
            entry["duplicate_value_count"] = acc["duplicate_value_count"]
            return entry

        if semantic_type == "numerical":
            entry["coercion_failures"] = acc["coercion_failures"]
            if not acc["non_null_chunks"]:
                entry["nulls_after"] = nulls_before + acc["coercion_failures"]
                entry["outliers_flagged"] = 0
                return entry

            non_null = pd.concat(acc["non_null_chunks"], ignore_index=True)
            # NOTE (discovered during P4 CI static-analysis rollout, not fixed
            # here - out of scope for a CI/CD-tooling change): this computed
            # fill_value is never stored on `entry` nor otherwise consumed by
            # this report-only finalize step. If the value-application pass
            # that actually imputes data recomputes its own mean/median
            # independently, this is dead/duplicate work, not a correctness
            # bug; if it does NOT, the reported "imputed_mean"/"imputed_median"
            # action could disagree with the value actually written. Needs a
            # dedicated, evidenced investigation of the value-application path
            # before changing behavior either way.
            fill_value = (  # noqa: F841
                float(non_null.mean())
                if config.numerical_imputation_strategy == "mean"
                else float(non_null.median())
            )
            entry["action"] = f"imputed_{config.numerical_imputation_strategy}"
            entry["nulls_imputed"] = nulls_before + acc["coercion_failures"]
            entry["nulls_after"] = 0

            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - config.outlier_iqr_multiplier * iqr
            upper_bound = q3 + config.outlier_iqr_multiplier * iqr
            entry["outliers_flagged"] = int(((non_null < lower_bound) | (non_null > upper_bound)).sum())
            return entry

        if semantic_type in ("category", "boolean"):
            if nulls_before > 0 and acc["any_non_null"]:
                if config.category_imputation_strategy == "sentinel":
                    entry["action"] = "imputed_sentinel"
                else:
                    entry["action"] = "imputed_mode"
                entry["nulls_imputed"] = nulls_before
                entry["nulls_after"] = 0
            return entry

        if semantic_type == "datetime":
            entry["coercion_failures"] = max(acc["nulls_after"] - nulls_before, 0)
            entry["nulls_after"] = acc["nulls_after"]
            entry["action"] = "coerced_datetime"
            return entry

        if semantic_type in ("email", "free_text"):
            entry["action"] = "whitespace_stripped" if config.strip_whitespace else "no_modification"
            return entry

        return entry
