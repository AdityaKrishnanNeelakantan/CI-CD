"""Checkpoint 3: rule-based semantic role inference.

Converts physical database fields into candidate semantic roles. This is
strictly a proposal engine - it never marks anything "approved". Every
candidate carries its confidence, the weighted evidence that produced it,
and alternative candidates, so a human (or a future automated policy) can
review the reasoning instead of trusting an opaque score.

Database types do not describe business meaning (a numeric customer_id
must not be modelled as a continuous quantity), so evidence is combined
from several independent signal families: column name, physical dtype,
uniqueness/cardinality, and value patterns in the sample. Confidence comes
from explicit weighted rules only - never invented text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

SEMANTIC_TYPES = frozenset(
    {
        "identifier",
        "numerical",
        "category",
        "boolean",
        "datetime",
        "email",
        "person_name",
        "phone_number",
        "free_text",
    }
)
STATUS_PROPOSED = "proposed"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class InferenceConfig:
    min_confidence: float = 0.5
    ambiguity_margin: float = 0.15
    near_unique_ratio: float = 0.98
    low_cardinality_max_distinct: int = 20
    low_cardinality_max_ratio: float = 0.5
    pattern_match_threshold: float = 0.8
    identifier_max_avg_length: float = 30.0

    # Name-pattern weights
    w_name_identifier: float = 3.0
    w_name_email: float = 3.0
    w_name_datetime: float = 2.5
    w_name_boolean: float = 2.5
    w_name_category_code: float = 1.5
    w_name_free_text: float = 2.0
    # Deliberately as strong as w_pattern_email (a verified value-pattern
    # match): a column literally named first_name/last_name/surname/... is
    # comparably decisive evidence on its own. Without this, a person-name
    # column with a handful of common values (a small sample repeating
    # "Smith", "Garcia", ...) loses to the low-cardinality "category"
    # signal and gets frequency-modelled from real values - a confirmed
    # privacy leak (real surnames sampled back out verbatim) found by
    # running this engine against a real medical fixture.
    w_name_person: float = 4.0
    w_name_phone: float = 3.0
    # Address-family / org columns must beat low-cardinality "category"
    # so real street/city/company strings are never frequency-modelled.
    w_name_address: float = 4.0
    w_name_city: float = 4.0
    w_name_state: float = 3.5
    w_name_postal: float = 3.5
    w_name_country: float = 3.5
    w_name_controlled_category: float = 3.0
    w_name_company: float = 4.0

    # Deliberately as strong as w_pattern_email: a column of real phone
    # numbers found to be low-cardinality (source data with a handful of
    # repeated numbers) previously outscored this on the "category"
    # signal alone and got frequency-modelled from real values - a
    # confirmed privacy leak (real phone numbers sampled back out
    # verbatim) found the same way as the person_name gap above, on the
    # same real medical fixture.
    w_pattern_phone: float = 4.0
    w_pattern_postal: float = 3.5

    # Physical dtype weights
    w_dtype_boolean: float = 4.0
    w_dtype_datetime: float = 4.0
    w_dtype_numeric: float = 2.0
    w_dtype_string_baseline: float = 0.5  # split across free_text and category

    # Uniqueness / cardinality weights. Deliberately modest: per the spec's
    # own worked example, identifier confidence should come from uniqueness
    # *combined with* a name/PK hint, not from uniqueness alone - a bare
    # near-unique column with no other signal (e.g. person names, a
    # generic numeric column) must not default to "identifier" just
    # because a small sample happened to have no repeats.
    w_near_unique_identifier: float = 0.8
    w_low_cardinality_category: float = 2.0

    # Value-pattern weights
    w_pattern_email: float = 4.0
    w_pattern_postcode: float = 2.0
    w_pattern_date: float = 3.0
    w_pattern_boolean_values: float = 3.0
    w_pattern_person_name: float = 3.5
    w_profile_quality_category: float = 0.5


_NAME_IDENTIFIER_RE = re.compile(r"(^|_)(id|uuid|guid|key)$", re.IGNORECASE)
_NAME_EMAIL_RE = re.compile(r"email", re.IGNORECASE)
# Person-name field conventions used across HR/CRM/EMR/finance. Includes
# exact `name` / `full_name` / `customer_name` style columns, but deliberately
# avoids a bare `_name$` suffix so provider_name/facility_name/company_name/
# product_name stay eligible for ordinary category/free_text treatment.
_NAME_PERSON_RE = re.compile(
    r"(^name$|(^|_)((first|last|given|family|middle|maiden|sur)_?names?|"
    r"full_?name|customer_?name|client_?name|patient_?name|contact_?name)$)",
    re.IGNORECASE,
)
_NAME_DATETIME_RE = re.compile(r"(^|_)(date|dt|time|timestamp|at)$", re.IGNORECASE)
_NAME_BOOLEAN_RE = re.compile(r"^(is|has|can|should)_|_flag$", re.IGNORECASE)
_NAME_CATEGORY_CODE_RE = re.compile(r"(^|_)(code|status_?code|type_?code)s?$", re.IGNORECASE)
_NAME_POSTAL_RE = re.compile(r"(^|_)(postal|zip|zipcode|postcode)(_?code)?s?$", re.IGNORECASE)
_NAME_FREE_TEXT_RE = re.compile(r"(description|comment|note|bio|summary|text)s?$", re.IGNORECASE)
_NAME_PHONE_RE = re.compile(r"(^|_)(phone|mobile|telephone|fax|cell)(_?number|_?no)?s?$", re.IGNORECASE)
_NAME_ADDRESS_RE = re.compile(r"(^|_)(address|street|street_?address|addr)(_|$)", re.IGNORECASE)
_NAME_CITY_RE = re.compile(r"(^|_)(city|town)(_|$)", re.IGNORECASE)
_NAME_STATE_RE = re.compile(r"(^|_)(state|province|region)(_|$)", re.IGNORECASE)
_NAME_COUNTRY_RE = re.compile(r"(^|_)(country|nation|country_?code)(_|$)", re.IGNORECASE)
_NAME_COMPANY_RE = re.compile(
    r"(^|_)(company|organization|organisation|employer|business|facility|provider|hospital|clinic|"
    r"vendor|supplier|merchant|payee|retailer|store)(_?name)?$",
    re.IGNORECASE,
)

_EMAIL_VALUE_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Multi-token title-ish values ("Ada Lovelace"). Single tokens and long
# free-text sentences are intentionally excluded â€” those stay under
# category/free_text/identifier scoring.
_PERSON_NAME_VALUE_RE = re.compile(
    r"^[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]+)+$"
)
_ORG_OR_THING_NAME_RE = re.compile(
    r"(company|facility|provider|product|organisation|organization|business|hospital|clinic|vendor|supplier|"
    r"merchant|payee|retailer|store)",
    re.IGNORECASE,
)
# A pure char-class/length regex (like the postcode ones below) is not
# enough on its own: "2024-01-01" (8 digits, dashes) satisfies the same
# shape as a formatted phone number and was found matching in testing.
# Phone numbers are instead validated on their DIGIT COUNT specifically
# (9-15, per typical domestic/E.164 lengths - an 8-digit date can never
# reach 9), plus a separator or leading "+" to stay distinguishable from
# a bare numeric identifier of similar length.
_PHONE_SHAPE_RE = re.compile(r"^\+?[\d\-.\s()]{7,20}$")
_PHONE_HAS_SEPARATOR_RE = re.compile(r"[\-.\s()]|^\+")
_POSTCODE_VALUE_RES = (
    re.compile(r"^\d{4,6}$"),
    re.compile(r"^[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$"),
)
_BOOLEAN_VALUE_SET = {"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"}


def _match_rate(values: pd.Series, pattern: re.Pattern[str]) -> float:
    non_null = values.dropna().astype(str)
    if non_null.empty:
        return 0.0
    matches = non_null.str.match(pattern)
    return float(matches.sum()) / len(non_null)


def _is_phone_shaped(value: str, *, require_separator: bool) -> bool:
    if not _PHONE_SHAPE_RE.match(value):
        return False
    digit_count = sum(1 for c in value if c.isdigit())
    if not (9 <= digit_count <= 15):
        return False
    return not require_separator or bool(_PHONE_HAS_SEPARATOR_RE.search(value))


def _phone_match_rate(values: pd.Series, *, require_separator: bool) -> float:
    non_null = values.dropna().astype(str)
    if non_null.empty:
        return 0.0
    matched = non_null.apply(lambda v: _is_phone_shaped(v, require_separator=require_separator))
    return float(matched.sum()) / len(non_null)


def _postcode_match_rate(values: pd.Series) -> float:
    non_null = values.dropna().astype(str)
    if non_null.empty:
        return 0.0
    matched = non_null.apply(lambda v: any(p.match(v) for p in _POSTCODE_VALUE_RES))
    return float(matched.sum()) / len(non_null)


def _boolean_value_set_rate(values: pd.Series) -> float:
    non_null = values.dropna().astype(str).str.lower()
    if non_null.empty:
        return 0.0
    matched = non_null.isin(_BOOLEAN_VALUE_SET)
    return float(matched.sum()) / len(non_null)


class _ScoreBoard:
    def __init__(self) -> None:
        self.scores: dict[str, float] = {t: 0.0 for t in SEMANTIC_TYPES}
        self.evidence: list[str] = []

    def add(self, semantic_type: str, weight: float, reason: str) -> None:
        self.scores[semantic_type] += weight
        self.evidence.append(reason)


def _score_name(column_name: str, board: _ScoreBoard, config: InferenceConfig) -> None:
    if _NAME_IDENTIFIER_RE.search(column_name):
        board.add("identifier", config.w_name_identifier, "name_pattern=identifier")
    if _NAME_EMAIL_RE.search(column_name):
        board.add("email", config.w_name_email, "name_pattern=email")
    if _NAME_PERSON_RE.search(column_name):
        board.add("person_name", config.w_name_person, "name_pattern=person_name")
    if _NAME_PHONE_RE.search(column_name):
        board.add("phone_number", config.w_name_phone, "name_pattern=phone_number")
    if _NAME_DATETIME_RE.search(column_name):
        board.add("datetime", config.w_name_datetime, "name_pattern=datetime")
    if _NAME_BOOLEAN_RE.search(column_name):
        board.add("boolean", config.w_name_boolean, "name_pattern=boolean")
    if _NAME_CATEGORY_CODE_RE.search(column_name) and not _NAME_POSTAL_RE.search(column_name):
        board.add("category", config.w_name_category_code, "name_pattern=category_code")
    if _NAME_POSTAL_RE.search(column_name):
        board.add("category", config.w_name_postal, "name_pattern=postal_code")
    if _NAME_CITY_RE.search(column_name):
        board.add("category", config.w_name_city, "name_pattern=city")
    if _NAME_STATE_RE.search(column_name):
        board.add("category", config.w_name_state, "name_pattern=state_region")
    if _NAME_COUNTRY_RE.search(column_name):
        board.add("category", config.w_name_country, "name_pattern=country")
    if _NAME_COMPANY_RE.search(column_name):
        board.add("free_text", config.w_name_company, "name_pattern=company")
    if re.search(r"(^|_)(currency|language|locale|timezone|segment|tier|status|type)(_|$)", column_name, re.IGNORECASE):
        board.add("category", config.w_name_controlled_category, "name_pattern=controlled_category")
    if _NAME_FREE_TEXT_RE.search(column_name):
        board.add("free_text", config.w_name_free_text, "name_pattern=free_text")


def _score_dtype(dtype: str, board: _ScoreBoard, config: InferenceConfig) -> None:
    if dtype == "boolean":
        board.add("boolean", config.w_dtype_boolean, "dtype=boolean")
    elif dtype == "datetime":
        board.add("datetime", config.w_dtype_datetime, "dtype=datetime")
    elif dtype in ("integer", "float"):
        board.add("numerical", config.w_dtype_numeric, "dtype=numeric")
    elif dtype == "string":
        # A bare string, absent any other signal, is genuinely ambiguous
        # between a free-text field and a category - splitting the weight
        # reflects that honestly instead of picking one arbitrarily.
        board.add("free_text", config.w_dtype_string_baseline, "dtype=string")
        board.add("category", config.w_dtype_string_baseline, "dtype=string")


def _score_cardinality(
    dtype: str,
    distinct_ratio: float,
    distinct_count: int,
    row_count: int,
    avg_string_length: float | None,
    board: _ScoreBoard,
    config: InferenceConfig,
) -> None:
    if row_count == 0:
        return
    if distinct_count > 1 and distinct_ratio >= config.near_unique_ratio:
        # Continuous measurements are naturally near-unique by construction
        # (float precision), so that alone is not identifier evidence.
        # Long near-unique strings (free text) look nothing like the short
        # structured tokens real identifiers use, so length gates them too.
        looks_like_free_text = (
            dtype == "string"
            and avg_string_length is not None
            and avg_string_length > config.identifier_max_avg_length
        )
        if dtype != "float" and not looks_like_free_text:
            board.add("identifier", config.w_near_unique_identifier, f"distinct_ratio={distinct_ratio}")
    # Both conditions together, not either alone: a small sample of
    # genuinely unique values (e.g. 4 free-text comments) trivially has a
    # "low" absolute distinct count without showing any real repetition,
    # which is what actually signals a bounded category set.
    if dtype in ("string", "boolean") and (
        distinct_count <= config.low_cardinality_max_distinct
        and distinct_ratio <= config.low_cardinality_max_ratio
    ):
        board.add("category", config.w_low_cardinality_category, f"cardinality={distinct_count}")


def _person_name_match_rate(values: pd.Series) -> float:
    non_null = values.dropna().astype(str)
    if non_null.empty:
        return 0.0
    matched = 0
    for value in non_null:
        text = value.strip()
        if not text or any(ch.isdigit() for ch in text):
            continue
        if len(text) > 60:
            continue
        tokens = text.split()
        if len(tokens) < 2:
            continue
        if _PERSON_NAME_VALUE_RE.match(text):
            matched += 1
    return float(matched) / len(non_null)


def _score_value_patterns(
    series: pd.Series, dtype: str, column_name: str, board: _ScoreBoard, config: InferenceConfig
) -> None:
    if dtype != "string":
        return
    email_rate = _match_rate(series, _EMAIL_VALUE_RE)
    if email_rate >= config.pattern_match_threshold:
        board.add("email", config.w_pattern_email, f"value_pattern_match_rate(email)={round(email_rate, 4)}")

    # Person-name value evidence is suppressed for organisation/thing-name
    # columns and for explicit identifier-named fields.
    if not _NAME_IDENTIFIER_RE.search(column_name) and not _ORG_OR_THING_NAME_RE.search(column_name):
        person_rate = _person_name_match_rate(series)
        if person_rate >= config.pattern_match_threshold:
            board.add(
                "person_name",
                config.w_pattern_person_name,
                f"value_pattern_match_rate(person_name)={round(person_rate, 4)}",
            )

    postcode_rate = _postcode_match_rate(series)
    if postcode_rate >= config.pattern_match_threshold:
        board.add(
            "category", config.w_pattern_postcode, f"value_pattern_match_rate(postcode)={round(postcode_rate, 4)}"
        )

    # An unformatted (no separator, no leading "+") digit-only value is
    # shape-identical to a bare numeric identifier, so the relaxed match is
    # only trusted when the column name doesn't already look like an
    # identifier (id/uuid/guid/key) - otherwise a sequential numeric ID
    # column picks up a false phone signal and loses its previously
    # confident classification. Columns with an ambiguous/non-identifier
    # name still need this relaxed check: a real medical fixture had a
    # repeated, unformatted phone column with no "phone"-like name that
    # the separator-only check let fall through to "category" and leak
    # verbatim.
    require_separator = bool(_NAME_IDENTIFIER_RE.search(column_name))
    phone_rate = _phone_match_rate(series, require_separator=require_separator)
    if phone_rate >= config.pattern_match_threshold:
        board.add(
            "phone_number", config.w_pattern_phone, f"value_pattern_match_rate(phone)={round(phone_rate, 4)}"
        )

    non_null = series.dropna()
    if not non_null.empty:
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        date_rate = float(parsed.notna().sum()) / len(non_null)
        if date_rate >= config.pattern_match_threshold:
            board.add("datetime", config.w_pattern_date, f"date_parse_success_rate={round(date_rate, 4)}")

    bool_rate = _boolean_value_set_rate(series)
    if bool_rate >= config.pattern_match_threshold:
        board.add(
            "boolean", config.w_pattern_boolean_values, f"value_pattern_match_rate(boolean)={round(bool_rate, 4)}"
        )


def _score_profile_warnings(
    column_warnings: list[str], board: _ScoreBoard, config: InferenceConfig
) -> None:
    for warning in ("severe_category_imbalance", "constant_column"):
        if warning in column_warnings:
            board.add("category", config.w_profile_quality_category, f"quality_warning={warning}")


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
        if (non_null == non_null.round()).all():
            return "integer"
        return "float"
    return "string"


class SemanticInferenceEngine:
    """Produces semantic-type candidates for every column in a table.

    Consumes a raw sample (for value-pattern matching only - raw values are
    never included in the returned candidates, only derived match-rate
    evidence) plus discovery and profile evidence.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self._config = config or InferenceConfig()

    def infer_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        discovery_table: dict[str, Any] | None = None,
        profile_table: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        config = self._config
        primary_key = set((discovery_table or {}).get("primary_key", []))
        profile_columns = (profile_table or {}).get("columns", {})

        candidates: dict[str, dict[str, Any]] = {}
        for column_name in df.columns:
            series = df[column_name]
            dtype = _infer_physical_dtype(series)
            row_count = len(series)
            non_null = series.dropna()
            distinct_count = int(non_null.nunique())
            distinct_ratio = round(distinct_count / row_count, 6) if row_count else 0.0

            avg_string_length = None
            if dtype == "string" and not non_null.empty:
                avg_string_length = float(non_null.astype(str).str.len().mean())

            board = _ScoreBoard()
            _score_name(str(column_name), board, config)
            _score_dtype(dtype, board, config)
            _score_cardinality(
                dtype, distinct_ratio, distinct_count, row_count, avg_string_length, board, config
            )
            _score_value_patterns(series, dtype, str(column_name), board, config)
            _score_profile_warnings(
                profile_columns.get(str(column_name), {}).get("warnings", []), board, config
            )

            if column_name in primary_key:
                board.add("identifier", config.w_near_unique_identifier, "primary_key=true")

            candidates[str(column_name)] = self._resolve(str(column_name), dtype, board, config)

        return candidates

    def infer_table_chunked(
        self,
        chunks: Iterable[pd.DataFrame],
        table_name: str,
        discovery_table: dict[str, Any] | None = None,
        profile_table: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Memory-bounded counterpart to infer_table().

        Unlike a report-only statistic, inference makes a *classification
        decision* per column that depends on distinct_count/distinct_ratio
        computed across the WHOLE column - these are not safely
        approximable the way profiling's reported distinct_count is (a
        per-chunk-sum approximation would systematically OVERCOUNT a truly
        low-cardinality column repeated across chunks, wrongly
        disqualifying it from "category" classification - a real
        classification regression, not just an imprecise number). So this
        method tracks the EXACT set of distinct non-null values per column
        across the whole stream before scoring:
        - For string-dtype columns, the full (non-deduplicated) non-null
          values are held to preserve value FREQUENCY (needed for
          value-pattern match rates, not just membership) - an honest,
          unavoidable exception (same category as cleaning's
          numerical-quantile case), since a distinct-only set would silently
          corrupt pattern match rates for duplicated values.
        - For every other dtype, only a deduplicated value set is held
          (cardinality only - no value-pattern matching applies to
          non-string dtypes), which stays cheap by construction for the
          common "low-cardinality category" case.

        Reuses _score_name/_score_dtype/_score_cardinality/
        _score_value_patterns/_score_profile_warnings/_resolve unchanged -
        the scoring and ambiguity-resolution logic (including the
        privacy-sensitive person_name/phone_number weighting) is identical
        to infer_table(), only fed pre-aggregated global statistics instead
        of a single in-memory DataFrame.
        """
        config = self._config
        primary_key = set((discovery_table or {}).get("primary_key", []))
        profile_columns = (profile_table or {}).get("columns", {})

        accumulators: dict[str, dict[str, Any]] = {}
        column_order: list[str] = []

        for chunk in chunks:
            if not column_order:
                column_order = [str(c) for c in chunk.columns]
                accumulators = {name: self._new_inference_accumulator() for name in column_order}
            for column_name in column_order:
                self._accumulate_inference_column(chunk[column_name], accumulators[column_name])

        return {
            column_name: self._finalize_inference_column(
                column_name, accumulators[column_name], primary_key, profile_columns, config
            )
            for column_name in column_order
        }

    def _new_inference_accumulator(self) -> dict[str, Any]:
        return {
            "pandas_kind": None,
            "row_count": 0,
            "non_null_count": 0,
            "all_round": True,
            "distinct_values": set(),
            "string_chunks": [],
        }

    def _accumulate_inference_column(self, series: pd.Series, acc: dict[str, Any]) -> None:
        acc["row_count"] += len(series)
        non_null = series.dropna()
        acc["non_null_count"] += len(non_null)
        if non_null.empty:
            return

        if acc["pandas_kind"] is None:
            if pd.api.types.is_bool_dtype(series):
                acc["pandas_kind"] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series):
                acc["pandas_kind"] = "datetime"
            elif pd.api.types.is_integer_dtype(series):
                acc["pandas_kind"] = "integer"
            elif pd.api.types.is_float_dtype(series):
                acc["pandas_kind"] = "float"
            else:
                acc["pandas_kind"] = "string"

        if acc["pandas_kind"] == "float" and not (non_null == non_null.round()).all():
            acc["all_round"] = False

        if acc["pandas_kind"] == "string":
            acc["string_chunks"].append(non_null.astype(str))
        else:
            acc["distinct_values"].update(non_null.tolist())

    def _finalize_inference_column(
        self,
        column_name: str,
        acc: dict[str, Any],
        primary_key: set[str],
        profile_columns: dict[str, Any],
        config: InferenceConfig,
    ) -> dict[str, Any]:
        kind = acc["pandas_kind"]
        if kind is None:
            dtype = "unknown"
        elif kind == "float":
            dtype = "integer" if acc["all_round"] else "float"
        else:
            dtype = kind

        row_count = acc["row_count"]
        full_string_series: pd.Series | None = None
        if dtype == "string":
            full_string_series = (
                pd.concat(acc["string_chunks"], ignore_index=True)
                if acc["string_chunks"]
                else pd.Series(dtype=object)
            )
            distinct_count = int(full_string_series.nunique())
            avg_string_length = (
                float(full_string_series.str.len().mean()) if not full_string_series.empty else None
            )
        else:
            distinct_count = len(acc["distinct_values"])
            avg_string_length = None

        distinct_ratio = round(distinct_count / row_count, 6) if row_count else 0.0

        board = _ScoreBoard()
        _score_name(column_name, board, config)
        _score_dtype(dtype, board, config)
        _score_cardinality(dtype, distinct_ratio, distinct_count, row_count, avg_string_length, board, config)
        if dtype == "string" and full_string_series is not None:
            _score_value_patterns(full_string_series, dtype, column_name, board, config)
        _score_profile_warnings(profile_columns.get(column_name, {}).get("warnings", []), board, config)

        if column_name in primary_key:
            board.add("identifier", config.w_near_unique_identifier, "primary_key=true")

        return self._resolve(column_name, dtype, board, config)

    def _resolve(
        self, column_name: str, dtype: str, board: _ScoreBoard, config: InferenceConfig
    ) -> dict[str, Any]:
        total = sum(board.scores.values())
        if total <= 0:
            return {
                "column": column_name,
                "semantic_type": "free_text",
                "status": STATUS_REVIEW_REQUIRED,
                "confidence": 0.0,
                "evidence": [f"dtype={dtype}", "no_signal"],
                "alternatives": [],
            }

        ranked = sorted(board.scores.items(), key=lambda item: item[1], reverse=True)
        ranked = [(t, s) for t, s in ranked if s > 0]
        top_type, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        confidence = round(top_score / total, 6)
        second_confidence = round(second_score / total, 6)
        is_ambiguous = confidence < config.min_confidence or (
            confidence - second_confidence
        ) < config.ambiguity_margin

        alternatives = [
            {"semantic_type": t, "confidence": round(s / total, 6)} for t, s in ranked[1:4]
        ]

        return {
            "column": column_name,
            "semantic_type": top_type,
            "status": STATUS_REVIEW_REQUIRED if is_ambiguous else STATUS_PROPOSED,
            "confidence": confidence,
            "evidence": board.evidence,
            "alternatives": alternatives,
        }

