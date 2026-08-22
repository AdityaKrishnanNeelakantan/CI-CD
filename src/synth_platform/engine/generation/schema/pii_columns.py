"""PII-aware column detection and fresh-value generation for mimic flows."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_SSN_NAME_HINTS = {"ssn", "social_security", "social_security_number", "social_sec_num"}
_AADHAAR_NAME_HINTS = {"aadhaar", "aadhar", "aadhaar_number", "aadhar_number"}
_NATIONAL_ID_HINTS = {
    "national_id",
    "nationalid",
    "nid",
    "tax_id",
    "taxid",
    "cpf",
    "nin",
    "ni_number",
    "pesel",
    "my_number",
    "ssn",
    "aadhaar",
    "aadhar",
}
_NAME_HINTS = {"name", "full_name", "first_name", "last_name", "fname", "lname", "customer_name"}
_EMAIL_HINTS = {"email", "email_address", "e_mail"}
_PHONE_NAME_HINTS = {"phone", "mobile", "cell", "telephone", "contact", "phone_number"}

_SSN_VALUE_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_AADHAAR_VALUE_RE = re.compile(r"^\d{4}\s\d{4}\s\d{4}$")
_CPF_VALUE_RE = re.compile(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
_NIN_VALUE_RE = re.compile(r"^[A-Z]{2}\d{6}[A-Z]$")
_PHONE_VALUE_RE = re.compile(r"^[\+\d\(\)\-\s\.]{7,20}$")

SOURCE_ENGINE_MIN_ROWS = 200
SOURCE_ENGINE_MIN_ROWS_WITH_PII = 500


def resolve_column_semantic(column_name: str) -> Optional[str]:
    """Return a fresh-generation semantic type for known PII / identity columns."""
    name = str(column_name or "").lower().strip()
    if name in _SSN_NAME_HINTS or name.endswith("_ssn"):
        return "ssn"
    if name in _AADHAAR_NAME_HINTS:
        return "aadhaar"
    if name in {"cpf", "cpf_number"}:
        return "cpf"
    if name in {"nin", "ni_number", "national_insurance"}:
        return "nin"
    if name in _NATIONAL_ID_HINTS or "national_id" in name:
        return "national_id"
    if name in _EMAIL_HINTS or "email" in name:
        return "email"
    if name in _PHONE_NAME_HINTS or "phone" in name or "mobile" in name:
        return "phone"
    if name in _NAME_HINTS or name.endswith("_name") or name.endswith("name"):
        return "name"
    return None


def _values_match_pattern(series: pd.Series, pattern: re.Pattern[str], *, threshold: float = 0.8) -> bool:
    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    return float(sample.str.match(pattern).mean()) >= threshold


def infer_value_semantic(column_name: str, series: pd.Series) -> Optional[str]:
    """Infer semantic type from column name and observed values."""
    named = resolve_column_semantic(column_name)
    if named:
        return named
    if _values_match_pattern(series, _SSN_VALUE_RE):
        return "ssn"
    if _values_match_pattern(series, _AADHAAR_VALUE_RE):
        return "aadhaar"
    if _values_match_pattern(series, _CPF_VALUE_RE):
        return "cpf"
    if _values_match_pattern(series, _NIN_VALUE_RE):
        return "nin"
    return None


def column_requires_fresh_generation(column_name: str, series: pd.Series) -> bool:
    """True when a column must never replay source values in synthetic output."""
    return infer_value_semantic(column_name, series) is not None


_TEXT_HEAVY_NAME_PARTS = {
    "note",
    "notes",
    "comment",
    "comments",
    "description",
    "descriptions",
    "review",
    "reviews",
    "summary",
    "summaries",
    "feedback",
    "message",
    "messages",
    "narrative",
    "narratives",
    "memo",
    "memos",
    "ticket_body",
    "support_message",
    "free_text",
}


def is_text_heavy_column(column_name: str, series: pd.Series) -> bool:
    """Detect long free-text columns that must not replay source text by default."""
    name = str(column_name or "").lower().strip()
    if any(part in name for part in _TEXT_HEAVY_NAME_PARTS):
        return True
    if column_requires_fresh_generation(name, series):
        return False
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    avg_len = float(non_null.str.len().mean())
    ratio = float(non_null.nunique()) / float(len(non_null))
    return avg_len >= 40.0 or (avg_len >= 20.0 and ratio >= 0.8)


def is_true_identifier_column(column_name: str, series: pd.Series) -> bool:
    """Numeric / *_id keys that are not PII strings or business measures."""
    if column_requires_fresh_generation(column_name, series):
        return False
    lowered = str(column_name or "").lower().strip()
    if lowered in {"id", "uuid", "guid"} or lowered.endswith("_id") or lowered.endswith("_uuid"):
        return True
    if pd.api.types.is_numeric_dtype(series):
        # Continuous floats (balance, score, amount) are not row identifiers.
        if pd.api.types.is_float_dtype(series):
            return False
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        # High-cardinality integers alone are not IDs unless the name suggests one.
        id_name_hint = (
            lowered.endswith("_key")
            or lowered.startswith("id_")
            or lowered.endswith("_id")
            or lowered.endswith("_uuid")
            or lowered in {"id", "uuid", "guid"}
        )
        if not id_name_hint:
            return False
        return float(non_null.nunique()) / float(len(non_null)) >= 0.95
    return False


def should_prefer_schema_engine(df: pd.DataFrame, *, table_name: str = "source_table") -> Tuple[bool, str]:
    """Product guardrail: when auto engine should fall back to schema-driven mimic."""
    row_count = len(df)
    if row_count < SOURCE_ENGINE_MIN_ROWS:
        return True, f"Only {row_count} source rows — schema-driven is safer below {SOURCE_ENGINE_MIN_ROWS}."

    pii_columns = [
        name for name in df.columns if column_requires_fresh_generation(name, df[name])
    ]
    sdv_columns = [
        name
        for name in df.columns
        if not column_requires_fresh_generation(name, df[name])
        and not is_true_identifier_column(name, df[name])
    ]

    if pii_columns and row_count < SOURCE_ENGINE_MIN_ROWS_WITH_PII:
        return True, (
            f"PII columns ({', '.join(pii_columns)}) with only {row_count} rows — "
            f"use schema-driven below {SOURCE_ENGINE_MIN_ROWS_WITH_PII}."
        )
    if not sdv_columns:
        return True, "No SDV-safe columns detected — schema-driven generates all fields fresh."
    return False, ""


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(int(seed or 0))


def _normalize_protected_key(value: Any) -> str:
    return str(value).strip().lower()


def _protected_forbidden_set(protected_values: Optional[Iterable[Any]]) -> set[str]:
    if not protected_values:
        return set()
    return {_normalize_protected_key(value) for value in protected_values if value is not None and str(value).strip()}


def _value_is_forbidden(value: Any, *, seen: set[str], forbidden: set[str]) -> bool:
    key = _normalize_protected_key(value)
    return key in seen or key in forbidden


def _unique_values(
    values: List[Any],
    count: int,
    generator: Any,
    *,
    forbidden: Optional[set[str]] = None,
) -> List[Any]:
    forbidden = forbidden or set()
    seen: set[str] = set()
    output: List[Any] = []
    attempts = 0
    max_attempts = max(count * 50, 100)
    while len(output) < count and attempts < max_attempts:
        attempts += 1
        value = generator()
        if _value_is_forbidden(value, seen=seen, forbidden=forbidden):
            continue
        seen.add(_normalize_protected_key(value))
        output.append(value)
    suffix = 0
    while len(output) < count:
        value = f"{generator()}-{suffix}"
        suffix += 1
        if _value_is_forbidden(value, seen=seen, forbidden=forbidden):
            continue
        seen.add(_normalize_protected_key(value))
        output.append(value)
    return output[:count]


def _unique_batch_generate(
    generate_one: Any,
    count: int,
    *,
    batch_size: int = 512,
    forbidden: Optional[set[str]] = None,
) -> List[Any]:
    """Generate unique values in batches to reduce Python loop overhead."""
    forbidden = forbidden or set()
    seen: set[str] = set()
    output: List[Any] = []
    attempts = 0
    max_attempts = max(count * 20, 200)
    while len(output) < count and attempts < max_attempts:
        attempts += 1
        need = count - len(output)
        batch_n = min(batch_size, max(need, 32))
        for value in (generate_one() for _ in range(batch_n)):
            if _value_is_forbidden(value, seen=seen, forbidden=forbidden):
                continue
            seen.add(_normalize_protected_key(value))
            output.append(value)
            if len(output) >= count:
                break
    suffix = 0
    while len(output) < count:
        value = f"{generate_one()}-{suffix}"
        suffix += 1
        if _value_is_forbidden(value, seen=seen, forbidden=forbidden):
            continue
        seen.add(_normalize_protected_key(value))
        output.append(value)
    return output[:count]


def _scrub_series_replays(
    series: pd.Series,
    forbidden: set[str],
    *,
    column_name: str,
    row_offset: int = 0,
) -> pd.Series:
    """Replace any values that collide with protected source values."""
    if series.empty or not forbidden:
        return series
    values = series.tolist()
    for idx, value in enumerate(values):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if _normalize_protected_key(value) in forbidden:
            values[idx] = f"synthetic_{column_name}_{row_offset + idx}"
    return pd.Series(values, dtype=object)


def _generate_text_heavy_batch(
    column_name: str,
    source_series: pd.Series,
    num_rows: int,
    *,
    seed: Optional[int] = None,
) -> pd.Series:
    rng = _rng_from_seed(seed)
    non_null = source_series.dropna().astype(str)
    null_rate = float(source_series.isna().mean()) if len(source_series) else 0.0
    if len(non_null):
        lengths = non_null.str.len()
        min_len = max(16, int(lengths.quantile(0.25)))
        max_len = max(min_len + 8, int(lengths.quantile(0.75)))
    else:
        min_len, max_len = 24, 96
    values: List[Any] = []
    for idx in range(num_rows):
        if null_rate > 0 and rng.random() < null_rate:
            values.append(None)
            continue
        length = int(rng.integers(min_len, max_len + 1))
        token = f"{column_name.replace('_', ' ')} detail {idx + 1}"
        text = (token + " " + "Synthetic narrative content for validation.")[:length]
        values.append(text)
    return pd.Series(values, dtype=object)


def generate_fresh_column_batch(
    column_name: str,
    source_series: pd.Series,
    num_rows: int,
    *,
    seed: Optional[int] = None,
    semantic: Optional[str] = None,
    unique: bool = True,
    row_offset: int = 0,
    protected_values: Optional[Iterable[Any]] = None,
) -> pd.Series:
    """Batch-optimized fresh column generation for large synthetic runs."""
    if num_rows <= 0:
        return pd.Series([], dtype=object)

    forbidden = _protected_forbidden_set(protected_values)
    if not forbidden and source_series is not None and len(source_series):
        forbidden = _protected_forbidden_set(source_series.dropna().tolist())

    rng = _rng_from_seed(seed)
    semantic = semantic or infer_value_semantic(column_name, source_series)
    if semantic is None and is_text_heavy_column(column_name, source_series):
        semantic = "text_heavy"
    if semantic is None and is_true_identifier_column(column_name, source_series):
        semantic = "id_numeric" if pd.api.types.is_numeric_dtype(source_series) else "id_like"
    semantic = semantic or "name"

    if semantic == "text_heavy":
        series = _generate_text_heavy_batch(column_name, source_series, num_rows, seed=seed)
        return _scrub_series_replays(series, forbidden, column_name=column_name, row_offset=row_offset)

    if semantic == "id_numeric" or (
        semantic == "id_like"
        and is_true_identifier_column(column_name, source_series)
        and pd.api.types.is_numeric_dtype(source_series)
    ):
        non_null = pd.to_numeric(source_series, errors="coerce").dropna()
        if len(non_null):
            low = int(non_null.min())
            high = int(non_null.max())
            global_offset = int(row_offset)
            if unique:
                start = high + 1 + global_offset
                chosen = np.arange(start, start + num_rows, dtype=np.int64)
            else:
                chosen = rng.integers(low, high + 1, size=num_rows)
            dtype = non_null.dtype if pd.api.types.is_integer_dtype(non_null) else int
            return pd.Series(chosen[:num_rows], dtype=dtype)

    if semantic == "ssn":
        from synth_platform.engine.generation.schema.smart_values import generate_ssn

        if unique:
            values = _unique_batch_generate(lambda: generate_ssn(rng), num_rows, forbidden=forbidden)
        else:
            values = [generate_ssn(rng) for _ in range(num_rows)]
        return _scrub_series_replays(
            pd.Series(values[:num_rows], dtype=object), forbidden, column_name=column_name, row_offset=row_offset
        )

    if semantic == "aadhaar":
        from synth_platform.engine.generation.schema.smart_values import generate_aadhaar

        if unique:
            values = _unique_batch_generate(lambda: generate_aadhaar(rng), num_rows, forbidden=forbidden)
        else:
            values = [generate_aadhaar(rng) for _ in range(num_rows)]
        return _scrub_series_replays(
            pd.Series(values[:num_rows], dtype=object), forbidden, column_name=column_name, row_offset=row_offset
        )

    if semantic in {"national_id", "cpf", "nin"}:
        try:
            from synth_platform.engine.generation.schema.generators.base import TextGenerator

            generator = TextGenerator(seed=int(seed or 0), locale="en_US")
            text_type = "cpf" if semantic == "cpf" else "national_id"
            if unique:
                values = _unique_batch_generate(
                    lambda: generator.generate(1, {"text_type": text_type})[0],
                    num_rows,
                    forbidden=forbidden,
                )
            else:
                values = generator.generate(num_rows, {"text_type": text_type}).tolist()
            return _scrub_series_replays(
                pd.Series(values[:num_rows], dtype=object), forbidden, column_name=column_name, row_offset=row_offset
            )
        except Exception:
            prefix = semantic.upper()
            return pd.Series([f"SYN-{prefix}-{row_offset + i:06d}" for i in range(num_rows)], dtype=object)

    if semantic == "email":
        try:
            from faker import Faker

            fake = Faker()
            Faker.seed(int(seed or 0))
            if unique:
                values = _unique_batch_generate(
                    lambda: fake.email(), num_rows, batch_size=1024, forbidden=forbidden
                )
            else:
                values = [fake.email() for _ in range(num_rows)]
            return _scrub_series_replays(
                pd.Series(values[:num_rows], dtype=object), forbidden, column_name=column_name, row_offset=row_offset
            )
        except Exception:
            return pd.Series(
                [f"user{row_offset + i}@example.com" for i in range(num_rows)],
                dtype=object,
            )

    if semantic == "phone":
        try:
            from faker import Faker

            fake = Faker()
            Faker.seed(int(seed or 0))
            if unique:
                values = _unique_batch_generate(
                    lambda: fake.phone_number(), num_rows, batch_size=1024, forbidden=forbidden
                )
            else:
                values = [fake.phone_number() for _ in range(num_rows)]
            return _scrub_series_replays(
                pd.Series(values[:num_rows], dtype=object), forbidden, column_name=column_name, row_offset=row_offset
            )
        except Exception:
            return pd.Series([f"+1-555-{rng.integers(1000, 9999):04d}" for _ in range(num_rows)], dtype=object)

    if semantic == "name":
        try:
            from faker import Faker

            fake = Faker()
            Faker.seed(int(seed or 0))
            if unique:
                values = _unique_batch_generate(
                    lambda: fake.name(), num_rows, batch_size=1024, forbidden=forbidden
                )
            else:
                values = [fake.name() for _ in range(num_rows)]
            return _scrub_series_replays(
                pd.Series(values[:num_rows], dtype=object), forbidden, column_name=column_name, row_offset=row_offset
            )
        except Exception:
            return pd.Series([f"Synthetic User {row_offset + i}" for i in range(num_rows)], dtype=object)

    offset = int(row_offset)
    return pd.Series([f"synthetic_{column_name}_{offset + i}" for i in range(num_rows)], dtype=object)


def generate_fresh_column(
    column_name: str,
    source_series: pd.Series,
    num_rows: int,
    *,
    seed: Optional[int] = None,
    semantic: Optional[str] = None,
    unique: bool = True,
    row_offset: int = 0,
) -> pd.Series:
    """Generate synthetic values for a PII or identifier column without replaying source."""
    return generate_fresh_column_batch(
        column_name,
        source_series,
        num_rows,
        seed=seed,
        semantic=semantic,
        unique=unique,
        row_offset=row_offset,
    )


def validate_no_pii_replay(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    fresh_columns: Sequence[str],
    *,
    protected_cache: Optional[Any] = None,
) -> Dict[str, Any]:
    """Verify sensitive columns do not reuse production values."""
    if protected_cache is not None and hasattr(protected_cache, "replay_report"):
        return protected_cache.replay_report(synthetic)

    column_replays: Dict[str, int] = {}
    for name in fresh_columns:
        if name not in source.columns or name not in synthetic.columns:
            continue
        source_set = set(source[name].astype(str).dropna().str.strip().str.lower())
        syn = synthetic[name].dropna().astype(str).str.strip().str.lower()
        column_replays[name] = int(syn.isin(source_set).sum())
    total = int(sum(column_replays.values()))
    return {
        "passed": total == 0,
        "column_replays": column_replays,
        "total_replay_values": total,
    }


def validate_row_uniqueness(synthetic: pd.DataFrame, unique_columns: Sequence[str]) -> Dict[str, Any]:
    issues: Dict[str, int] = {}
    for name in unique_columns:
        if name not in synthetic.columns:
            continue
        dupes = int(synthetic[name].duplicated().sum())
        if dupes:
            issues[name] = dupes
    return {"passed": not issues, "duplicate_counts": issues}
