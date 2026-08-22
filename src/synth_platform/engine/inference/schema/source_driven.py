"""
Source-driven synthetic generation — learn from production data with SDV.

Profiles numeric, categorical, date, boolean, and ID-like columns, fits an SDV
synthesizer (Gaussian Copula, CTGAN, or TVAE), persists profile/model metadata
to disk, and generates synthetic rows that preserve marginals and joint patterns.

Schema-driven generation (``DataSimulator``, ``mvp mimic``) is unchanged.
"""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from synth_platform.engine.profiling.schema.profiler import (
    DataProfiler,
    _cardinality_ratio,
    _fit_categorical,
    _fit_date,
    _fit_numeric,
    _is_date_col,
)
from synth_platform.engine.generation.schema.pii_columns import (
    column_requires_fresh_generation,
    generate_fresh_column,
    infer_value_semantic,
    is_text_heavy_column,
    is_true_identifier_column,
    validate_no_pii_replay,
    validate_row_uniqueness,
)

try:
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
    from sdv.utils import load_synthesizer

    SDV_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via importorskip in tests
    SDV_AVAILABLE = False
    Metadata = None  # type: ignore[misc, assignment]
    CTGANSynthesizer = None  # type: ignore[misc, assignment]
    GaussianCopulaSynthesizer = None  # type: ignore[misc, assignment]
    TVAESynthesizer = None  # type: ignore[misc, assignment]
    load_synthesizer = None  # type: ignore[misc, assignment]


class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"
    BOOLEAN = "boolean"
    ID_LIKE = "id_like"
    TEXT = "text"


class SourceModelType(str, Enum):
    GAUSSIAN_COPULA = "gaussian_copula"
    CTGAN = "ctgan"
    TVAE = "tvae"


PROFILE_FILENAME = "profile.json"
METADATA_FILENAME = "sdv_metadata.json"
MANIFEST_FILENAME = "manifest.json"
MODEL_FILENAME = "model.pkl"
PROTECTED_CACHE_FILENAME = "protected_cache.json"
DEFAULT_SYNTHETIC_FILENAME = "synthetic.csv"
DEFAULT_FIT_MAX_ROWS = 5_000
DEFAULT_PROFILE_MAX_ROWS = 5_000
DEFAULT_SDV_EPOCHS = 50
PROFILE_CATEGORICAL_SAMPLE_ROWS = 10_000

_SYNTHESIZER_CLASSES = {
    SourceModelType.GAUSSIAN_COPULA: GaussianCopulaSynthesizer,
    SourceModelType.CTGAN: CTGANSynthesizer,
    SourceModelType.TVAE: TVAESynthesizer,
}


def require_sdv() -> None:
    if not SDV_AVAILABLE:
        raise ImportError(
            "Source-driven generation requires SDV. Install with: pip install 'mvp[advanced]'"
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _series_base_stats(series: pd.Series) -> Dict[str, Any]:
    """Single-pass null rate, uniqueness, and non-null values for a column."""
    total = len(series)
    null_rate = float(series.isna().mean()) if total else 0.0
    non_null = series.dropna()
    n_non_null = len(non_null)
    if n_non_null == 0:
        return {
            "null_rate": null_rate,
            "unique_count": 0,
            "cardinality_ratio": 1.0,
            "non_null": non_null,
        }
    unique_count = int(non_null.nunique())
    return {
        "null_rate": null_rate,
        "unique_count": unique_count,
        "cardinality_ratio": unique_count / n_non_null,
        "non_null": non_null,
    }


def _series_is_integer_like(non_null: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(non_null):
        return True
    if len(non_null) == 0:
        return False
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.isna().any():
        return False
    return bool(np.all(np.isclose(numeric % 1, 0, rtol=0, atol=1e-9)))


def _fit_categorical_for_profile(series: pd.Series) -> Dict[str, Any]:
    non_null = series.dropna().astype(str)
    if len(non_null) > PROFILE_CATEGORICAL_SAMPLE_ROWS:
        non_null = non_null.sample(PROFILE_CATEGORICAL_SAMPLE_ROWS, random_state=0)
    return _fit_categorical(non_null)


def _resolve_row_cap(cap: Optional[int], row_count: int, default: int) -> Optional[int]:
    if cap is not None:
        return int(cap) if row_count > int(cap) else None
    if row_count > default:
        return default
    return None


def _sample_df_for_cap(df: pd.DataFrame, cap: Optional[int], *, seed: Optional[int]) -> pd.DataFrame:
    if cap is None or len(df) <= cap:
        return df
    return df.sample(n=int(cap), random_state=int(seed or 0))


def _is_id_like_column(name: str, series: pd.Series) -> bool:
    if _is_date_col(series):
        return False
    if column_requires_fresh_generation(name, series):
        return False
    return is_true_identifier_column(name, series)


def _detect_column_kind(
    name: str,
    series: pd.Series,
    stats: Optional[Dict[str, Any]] = None,
) -> ColumnKind:
    stats = stats or _series_base_stats(series)
    non_null = stats["non_null"]
    n_unique = int(stats["unique_count"])
    ratio = float(stats["cardinality_ratio"])

    if pd.api.types.is_bool_dtype(series):
        return ColumnKind.BOOLEAN
    if _is_date_col(series):
        return ColumnKind.DATE
    if _is_id_like_column(name, series):
        return ColumnKind.ID_LIKE
    if pd.api.types.is_numeric_dtype(series):
        is_int = pd.api.types.is_integer_dtype(series) or _series_is_integer_like(non_null)
        if n_unique <= 20 and ratio < DataProfiler.CATEGORICAL_RATIO:
            return ColumnKind.CATEGORICAL
        return ColumnKind.NUMERIC

    if n_unique <= DataProfiler.CATEGORICAL_MAX and ratio < DataProfiler.CATEGORICAL_RATIO:
        return ColumnKind.CATEGORICAL
    return ColumnKind.TEXT


def _sdtype_for_kind(kind: ColumnKind) -> str:
    mapping = {
        ColumnKind.NUMERIC: "numerical",
        ColumnKind.CATEGORICAL: "categorical",
        ColumnKind.DATE: "datetime",
        ColumnKind.BOOLEAN: "boolean",
        ColumnKind.ID_LIKE: "id",
        ColumnKind.TEXT: "categorical",
    }
    return mapping[kind]


def _numeric_correlations(df: pd.DataFrame, *, max_columns: int = 12) -> List[Dict[str, Any]]:
    """Top absolute Pearson correlations between numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    cols = list(numeric.columns[:max_columns])
    corr = numeric[cols].corr(numeric_only=True)
    pairs: List[Dict[str, Any]] = []
    for idx, col_a in enumerate(cols):
        for col_b in cols[idx + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            pairs.append(
                {
                    "column_a": str(col_a),
                    "column_b": str(col_b),
                    "correlation": round(float(value), 4),
                }
            )
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return pairs[:20]


def _missingness_summary(columns: Mapping[str, ColumnProfile]) -> Dict[str, Any]:
    if not columns:
        return {"columns_with_nulls": 0, "avg_null_rate": 0.0, "max_null_rate": 0.0}
    rates = [float(col.null_rate) for col in columns.values()]
    return {
        "columns_with_nulls": sum(1 for rate in rates if rate > 0),
        "avg_null_rate": round(float(np.mean(rates)), 6),
        "max_null_rate": round(float(max(rates)), 6),
    }


def _marginal_for_column(
    kind: ColumnKind,
    series: pd.Series,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stats = stats or _series_base_stats(series)
    null_rate = float(stats["null_rate"])
    unique_count = int(stats["unique_count"])
    cardinality_ratio = float(stats["cardinality_ratio"])
    base = {
        "null_rate": round(null_rate, 6),
        "unique_count": unique_count,
        "cardinality_ratio": round(cardinality_ratio, 6),
    }

    if kind == ColumnKind.BOOLEAN:
        non_null = stats["non_null"]
        true_rate = float(non_null.astype(bool).mean()) if len(non_null) else 0.0
        return {**base, "true_rate": round(true_rate, 6)}

    if kind == ColumnKind.DATE:
        from synth_platform.engine.generation.schema.datetime_utils import coerce_datetime_series

        parsed = coerce_datetime_series(series).dropna()
        fitted = _fit_date(series)
        if len(parsed):
            return {
                **base,
                **fitted,
                "median": parsed.median().strftime("%Y-%m-%d"),
            }
        return {**base, **fitted}

    if kind == ColumnKind.CATEGORICAL:
        fitted = _fit_categorical_for_profile(series.astype(str))
        top_values = [
            {"value": choice, "proportion": prob}
            for choice, prob in zip(fitted["choices"], fitted["probabilities"])
        ]
        return {**base, "top_values": top_values[:20]}

    if kind == ColumnKind.NUMERIC:
        non_null = stats["non_null"]
        is_int = pd.api.types.is_integer_dtype(series) or _series_is_integer_like(non_null)
        fitted = _fit_numeric(series, is_int=is_int)
        if len(non_null):
            return {
                **base,
                **fitted,
                "median": round(float(non_null.median()), 6),
                "skew": round(float(non_null.astype(float).skew()), 6),
            }
        return {**base, **fitted}

    if kind == ColumnKind.ID_LIKE:
        sample_values = [str(v) for v in series.dropna().astype(str).head(5).tolist()]
        return {**base, "sample_values": sample_values}

    sample_values = [str(v) for v in series.dropna().astype(str).head(5).tolist()]
    return {**base, "sample_values": sample_values}


@dataclass
class ColumnProfile:
    name: str
    kind: str
    sdtype: str
    null_rate: float
    unique_count: int
    cardinality_ratio: float
    marginal: Dict[str, Any] = field(default_factory=dict)
    pii_sensitive: bool = False
    fresh_generate: bool = False
    use_sdv: bool = True
    semantic_type: Optional[str] = None

    @classmethod
    def from_series(cls, name: str, series: pd.Series) -> ColumnProfile:
        stats = _series_base_stats(series)
        semantic = infer_value_semantic(name, series)
        fresh = column_requires_fresh_generation(name, series)
        text_heavy = is_text_heavy_column(name, series)
        kind = _detect_column_kind(name, series, stats)
        if fresh or text_heavy:
            kind = ColumnKind.TEXT if text_heavy else kind
        marginal = _marginal_for_column(kind, series, stats)
        use_sdv = not fresh and not text_heavy and not (
            kind == ColumnKind.ID_LIKE and is_true_identifier_column(name, series)
        )
        sdtype = _sdtype_for_kind(kind) if use_sdv else "fresh"
        return cls(
            name=name,
            kind=kind.value,
            sdtype=sdtype,
            null_rate=float(marginal.get("null_rate", stats["null_rate"])),
            unique_count=int(marginal.get("unique_count", stats["unique_count"])),
            cardinality_ratio=float(marginal.get("cardinality_ratio", stats["cardinality_ratio"])),
            marginal=marginal,
            pii_sensitive=bool(fresh),
            fresh_generate=bool(fresh or text_heavy or (kind == ColumnKind.ID_LIKE and not use_sdv)),
            use_sdv=bool(use_sdv),
            semantic_type="text_heavy" if text_heavy else semantic,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ColumnProfile:
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            sdtype=str(payload["sdtype"]),
            null_rate=float(payload["null_rate"]),
            unique_count=int(payload["unique_count"]),
            cardinality_ratio=float(payload["cardinality_ratio"]),
            marginal=dict(payload.get("marginal") or {}),
            pii_sensitive=bool(payload.get("pii_sensitive", False)),
            fresh_generate=bool(payload.get("fresh_generate", False)),
            use_sdv=bool(payload.get("use_sdv", True)),
            semantic_type=payload.get("semantic_type"),
        )


@dataclass
class SourceTableProfile:
    table_name: str
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    recommended_model: str
    created_at: str = field(default_factory=_utc_now_iso)
    numeric_correlations: List[Dict[str, Any]] = field(default_factory=list)
    missingness: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "recommended_model": self.recommended_model,
            "created_at": self.created_at,
            "numeric_correlations": self.numeric_correlations,
            "missingness": self.missingness,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceTableProfile:
        columns = {
            str(name): ColumnProfile.from_dict(col)
            for name, col in (payload.get("columns") or {}).items()
        }
        return cls(
            table_name=str(payload["table_name"]),
            row_count=int(payload["row_count"]),
            column_count=int(payload.get("column_count") or len(columns)),
            columns=columns,
            recommended_model=str(payload.get("recommended_model") or SourceModelType.GAUSSIAN_COPULA.value),
            created_at=str(payload.get("created_at") or _utc_now_iso()),
            numeric_correlations=list(payload.get("numeric_correlations") or []),
            missingness=dict(payload.get("missingness") or {}),
        )


class SourceDataProfiler:
    """Extract column-level profiles from production/source data."""

    def profile(
        self,
        df: pd.DataFrame,
        *,
        table_name: str = "source_table",
        source_row_count: Optional[int] = None,
    ) -> SourceTableProfile:
        row_count = int(source_row_count if source_row_count is not None else len(df))
        columns = {
            str(col_name): ColumnProfile.from_series(str(col_name), df[col_name])
            for col_name in df.columns
        }
        profile = SourceTableProfile(
            table_name=table_name,
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            recommended_model=recommend_source_model(row_count, columns),
            numeric_correlations=_numeric_correlations(df),
            missingness=_missingness_summary(columns),
        )
        return profile


def recommend_source_model(row_count: int, columns: Mapping[str, ColumnProfile]) -> str:
    """Pick a default SDV model — Gaussian Copula is fast; CTGAN/TVAE for very large tables."""
    numeric_or_date = sum(
        1 for col in columns.values() if col.kind in {ColumnKind.NUMERIC.value, ColumnKind.DATE.value}
    )
    if row_count >= 50_000 and numeric_or_date >= 5:
        return SourceModelType.CTGAN.value
    if row_count >= 20_000 and numeric_or_date >= 3:
        return SourceModelType.TVAE.value
    return SourceModelType.GAUSSIAN_COPULA.value


def _finalize_sdv_metadata(metadata: Metadata) -> Metadata:
    """Return a clean Metadata copy so SDV synthesizers skip init warnings."""
    return Metadata.load_from_dict(metadata.to_dict())


def _load_sdv_metadata_json(path: Union[str, Path]) -> Metadata:
    """Load SDV metadata, including legacy SingleTableMetadata JSON artifacts."""
    return Metadata.load_from_json(str(path))


def build_sdv_metadata(df: pd.DataFrame, profile: SourceTableProfile) -> Metadata:
    require_sdv()
    from synth_platform.engine.generation.schema.datetime_utils import infer_datetime_format

    sdv_columns = [name for name, col in profile.columns.items() if col.use_sdv and name in df.columns]
    if not sdv_columns:
        raise ValueError("No SDV-safe columns available for source-driven fit.")
    sdv_df = df[sdv_columns]
    table_name = profile.table_name or "source_table"

    metadata = Metadata()
    metadata.add_table(table_name)
    for name in sdv_columns:
        column = profile.columns[name]
        col_kwargs: Dict[str, Any] = {"sdtype": column.sdtype}
        if column.sdtype == "datetime":
            col_kwargs["datetime_format"] = infer_datetime_format(sdv_df[name])
        metadata.add_column(name, table_name=table_name, **col_kwargs)

    return _finalize_sdv_metadata(metadata)


def sdv_column_names(profile: SourceTableProfile) -> List[str]:
    return [name for name, col in profile.columns.items() if col.use_sdv]


def fresh_column_names(profile: SourceTableProfile) -> List[str]:
    return [name for name, col in profile.columns.items() if col.fresh_generate]


def pii_replay_column_names(profile: SourceTableProfile) -> List[str]:
    """Columns that must not copy production values (PII overlay + true row IDs)."""
    names: List[str] = []
    for name, col in profile.columns.items():
        if col.pii_sensitive or col.semantic_type == "text_heavy":
            names.append(name)
            continue
        if not col.fresh_generate or col.kind != ColumnKind.ID_LIKE.value:
            continue
        lowered = name.lower().strip()
        if (
            lowered.endswith("_id")
            or lowered.endswith("_uuid")
            or lowered in {"id", "uuid", "guid"}
        ):
            names.append(name)
    return names


def identifier_column_names(profile: SourceTableProfile) -> List[str]:
    return [
        name
        for name, col in profile.columns.items()
        if not col.fresh_generate and col.kind == ColumnKind.ID_LIKE.value
    ]


def _apply_fresh_columns(
    synthetic: pd.DataFrame,
    source_df: pd.DataFrame,
    profile: SourceTableProfile,
    *,
    seed: Optional[int],
    output_columns: Optional[Sequence[str]] = None,
    fresh_columns: Optional[Sequence[str]] = None,
    overlay_plan: Optional[Any] = None,
    row_offset: int = 0,
) -> pd.DataFrame:
    from synth_platform.engine.inference.schema.source_overlay import apply_overlay_plan, build_overlay_plan

    columns = list(output_columns or source_df.columns)
    fresh_names = list(fresh_columns or fresh_column_names(profile))
    plan = overlay_plan or build_overlay_plan(
        profile,
        source_df,
        output_columns=columns,
        sdv_columns=sdv_column_names(profile),
        fresh_columns=fresh_names,
    )
    return apply_overlay_plan(
        synthetic,
        plan,
        source_df,
        seed=seed,
        profile=profile,
        row_offset=row_offset,
    )


def _resolve_model_type(model_type: Optional[str], profile: SourceTableProfile) -> SourceModelType:
    if model_type is None:
        return SourceModelType(profile.recommended_model)
    return SourceModelType(model_type)


def _create_synthesizer(
    model_type: SourceModelType,
    metadata: Metadata,
    *,
    epochs: Optional[int] = None,
) -> Any:
    require_sdv()
    cls = _SYNTHESIZER_CLASSES[model_type]
    if model_type in {SourceModelType.CTGAN, SourceModelType.TVAE}:
        return cls(metadata, epochs=int(epochs or DEFAULT_SDV_EPOCHS), verbose=False)
    return cls(metadata)


@dataclass
class SourceDrivenArtifacts:
    output_dir: Path
    profile_path: Path
    metadata_path: Path
    manifest_path: Path
    model_path: Path
    synthetic_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, str]:
        payload = {
            "output_dir": str(self.output_dir),
            "profile_path": str(self.profile_path),
            "metadata_path": str(self.metadata_path),
            "manifest_path": str(self.manifest_path),
            "model_path": str(self.model_path),
        }
        if self.synthetic_path is not None:
            payload["synthetic_path"] = str(self.synthetic_path)
        return payload


@dataclass
class SourceDrivenManifest:
    table_name: str
    model_type: str
    source_rows: int
    synthetic_rows: int
    seed: Optional[int]
    created_at: str
    package_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceDrivenManifest:
        return cls(
            table_name=str(payload["table_name"]),
            model_type=str(payload["model_type"]),
            source_rows=int(payload["source_rows"]),
            synthetic_rows=int(payload["synthetic_rows"]),
            seed=payload.get("seed"),
            created_at=str(payload.get("created_at") or _utc_now_iso()),
            package_version=str(payload.get("package_version") or ""),
        )


class SourceDrivenGenerator:
    """Fit, persist, and sample from an SDV source-driven synthesizer."""

    def __init__(self) -> None:
        self.profile: Optional[SourceTableProfile] = None
        self.metadata: Optional[Metadata] = None
        self.synthesizer: Any = None
        self.model_type: Optional[SourceModelType] = None
        self.manifest: Optional[SourceDrivenManifest] = None
        self._source_df: Optional[pd.DataFrame] = None
        self._sdv_columns: List[str] = []
        self._fresh_only: bool = False
        self._output_columns: List[str] = []
        self._fresh_columns: List[str] = []
        self._overlay_plan: Optional[Any] = None

    @property
    def is_fitted(self) -> bool:
        return self.profile is not None and self._source_df is not None and (
            self.synthesizer is not None or self._fresh_only
        )

    def fit(
        self,
        df: pd.DataFrame,
        *,
        table_name: str = "source_table",
        model_type: Optional[str] = None,
        seed: Optional[int] = None,
        epochs: Optional[int] = None,
        fit_max_rows: Optional[int] = None,
        profile_max_rows: Optional[int] = None,
        stages: Optional[Any] = None,
    ) -> SourceTableProfile:
        require_sdv()
        if seed is not None:
            _set_random_seeds(seed)

        source_rows = len(df)
        effective_fit_cap = _resolve_row_cap(fit_max_rows, source_rows, DEFAULT_FIT_MAX_ROWS)
        effective_profile_cap = _resolve_row_cap(
            profile_max_rows if profile_max_rows is not None else fit_max_rows,
            source_rows,
            DEFAULT_PROFILE_MAX_ROWS,
        )
        profile_df = _sample_df_for_cap(df, effective_profile_cap, seed=seed)
        training_df = _sample_df_for_cap(df, effective_fit_cap, seed=seed)

        profiler = SourceDataProfiler()
        profile_timer = nullcontext()
        if stages is not None:
            from synth_platform.engine.validation.schema.throughput import time_stage

            profile_timer = time_stage(stages, "profile_seconds")
        with profile_timer:
            self.profile = profiler.profile(
                profile_df,
                table_name=table_name,
                source_row_count=source_rows,
            )
        self._sdv_columns = sdv_column_names(self.profile)
        self._fresh_only = len(self._sdv_columns) == 0
        self._output_columns = list(df.columns)
        self._fresh_columns = fresh_column_names(self.profile)
        resolved = _resolve_model_type(model_type, self.profile)
        self.model_type = resolved

        fit_timer = nullcontext()
        if stages is not None:
            from synth_platform.engine.validation.schema.throughput import time_stage

            fit_timer = time_stage(stages, "fit_seconds")
        with fit_timer:
            if not self._fresh_only:
                sdv_training = training_df[self._sdv_columns]
                self.metadata = build_sdv_metadata(sdv_training, self.profile)
                self.synthesizer = _create_synthesizer(resolved, self.metadata, epochs=epochs)
                self.synthesizer.fit(sdv_training)
            else:
                self.metadata = None
                self.synthesizer = None

        needed_columns = set(self._fresh_columns)
        needed_columns.update(
            name
            for name in identifier_column_names(self.profile)
            if name not in self._fresh_columns
        )
        if needed_columns:
            self._source_df = df[sorted(needed_columns, key=self._output_columns.index)].copy()
        else:
            self._source_df = _stub_source_df_from_profile(self.profile)
        from synth_platform.engine.inference.schema.source_overlay import build_overlay_plan

        self._overlay_plan = build_overlay_plan(
            self.profile,
            self._source_df,
            output_columns=self._output_columns,
            sdv_columns=self._sdv_columns,
            fresh_columns=self._fresh_columns,
        )
        return self.profile

    def sample(
        self,
        num_rows: int,
        *,
        seed: Optional[int] = None,
        stages: Optional[Any] = None,
        row_offset: int = 0,
    ) -> pd.DataFrame:
        require_sdv()
        if not self.is_fitted or self.profile is None or self._source_df is None:
            raise ValueError("SourceDrivenGenerator must be fit before sample()")
        import time

        if seed is not None:
            _set_random_seeds(seed)

        sdv_start = time.perf_counter()
        if self._fresh_only:
            synthetic = pd.DataFrame(index=range(int(num_rows)))
        else:
            synthetic = self.synthesizer.sample(num_rows=int(num_rows))
        sdv_elapsed = time.perf_counter() - sdv_start

        overlay_start = time.perf_counter()
        result = _apply_fresh_columns(
            synthetic,
            self._source_df,
            self.profile,
            seed=seed,
            output_columns=self._output_columns,
            fresh_columns=self._fresh_columns,
            overlay_plan=self._overlay_plan,
            row_offset=row_offset,
        )
        overlay_elapsed = time.perf_counter() - overlay_start

        if stages is not None:
            stages.extra["sdv_sample_seconds"] = stages.extra.get("sdv_sample_seconds", 0.0) + sdv_elapsed
            stages.pii_overlay_seconds += overlay_elapsed
        return result

    def sample_chunks(
        self,
        total_rows: int,
        chunk_size: int,
        *,
        seed: Optional[int] = None,
        stages: Optional[Any] = None,
    ) -> Iterator[pd.DataFrame]:
        """Yield synthetic rows in bounded-memory chunks."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        remaining = int(total_rows)
        chunk_index = 0
        rows_emitted = 0
        while remaining > 0:
            batch = min(chunk_size, remaining)
            chunk_seed = None if seed is None else int(seed) + chunk_index
            chunk_start = __import__("time").perf_counter()
            chunk = self.sample(batch, seed=chunk_seed, stages=stages, row_offset=rows_emitted)
            if stages is not None:
                stages.extra["chunk_sample_seconds"] = stages.extra.get("chunk_sample_seconds", 0.0) + (
                    __import__("time").perf_counter() - chunk_start
                )
                stages.extra["chunk_count"] = stages.extra.get("chunk_count", 0) + 1
            yield chunk
            remaining -= batch
            rows_emitted += batch
            chunk_index += 1

    def fitted_model_type(self) -> Optional[str]:
        return self.model_type.value if self.model_type is not None else None

    def save(self, output_dir: Union[str, Path], *, synthetic_rows: int, seed: Optional[int] = None) -> SourceDrivenArtifacts:
        require_sdv()
        if not self.is_fitted or self.profile is None or self.model_type is None:
            raise ValueError("SourceDrivenGenerator must be fit before save()")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        profile_path = out / PROFILE_FILENAME
        metadata_path = out / METADATA_FILENAME
        manifest_path = out / MANIFEST_FILENAME
        model_path = out / MODEL_FILENAME

        profile_path.write_text(json.dumps(self.profile.to_dict(), indent=2), encoding="utf-8")
        if self.metadata is not None:
            self.metadata.save_to_json(str(metadata_path))
        else:
            metadata_path.write_text(json.dumps({"fresh_only": True}, indent=2), encoding="utf-8")
        if self.synthesizer is not None:
            self.synthesizer.save(model_path)
        else:
            model_path.write_text(json.dumps({"fresh_only": True}, indent=2), encoding="utf-8")

        from synth_platform.engine.generation.schema import __version__

        self.manifest = SourceDrivenManifest(
            table_name=self.profile.table_name,
            model_type=self.model_type.value,
            source_rows=self.profile.row_count,
            synthetic_rows=int(synthetic_rows),
            seed=seed,
            created_at=_utc_now_iso(),
            package_version=__version__,
        )
        manifest_path.write_text(json.dumps(self.manifest.to_dict(), indent=2), encoding="utf-8")

        if self._overlay_plan is not None:
            cache_path = out / PROTECTED_CACHE_FILENAME
            cache_path.write_text(
                json.dumps(self._overlay_plan.protected_cache.to_dict(), indent=2),
                encoding="utf-8",
            )

        return SourceDrivenArtifacts(
            output_dir=out,
            profile_path=profile_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            model_path=model_path,
        )

    @classmethod
    def load(cls, output_dir: Union[str, Path]) -> SourceDrivenGenerator:
        require_sdv()
        out = Path(output_dir)
        profile_path = out / PROFILE_FILENAME
        metadata_path = out / METADATA_FILENAME
        manifest_path = out / MANIFEST_FILENAME
        model_path = out / MODEL_FILENAME

        for path in (profile_path, metadata_path, manifest_path, model_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing source-driven artifact: {path}")

        generator = cls()
        generator.profile = SourceTableProfile.from_dict(json.loads(profile_path.read_text(encoding="utf-8")))
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_payload.get("fresh_only"):
            generator.metadata = None
            generator._fresh_only = True
            generator._sdv_columns = []
        else:
            generator.metadata = _load_sdv_metadata_json(metadata_path)
            generator._sdv_columns = sdv_column_names(generator.profile)
            generator._fresh_only = len(generator._sdv_columns) == 0
        generator.manifest = SourceDrivenManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        generator.model_type = SourceModelType(generator.manifest.model_type)
        if generator._fresh_only or metadata_payload.get("fresh_only"):
            generator.synthesizer = None
        else:
            generator.synthesizer = load_synthesizer(model_path)
        generator._source_df = _stub_source_df_from_profile(generator.profile)
        generator._output_columns = list(generator.profile.columns.keys())
        generator._fresh_columns = fresh_column_names(generator.profile)
        from synth_platform.engine.inference.schema.source_overlay import build_overlay_plan

        generator._overlay_plan = build_overlay_plan(
            generator.profile,
            generator._source_df,
            output_columns=generator._output_columns,
            sdv_columns=generator._sdv_columns,
            fresh_columns=generator._fresh_columns,
        )
        cache_path = out / PROTECTED_CACHE_FILENAME
        if cache_path.exists():
            from synth_platform.engine.inference.schema.source_overlay import ProtectedValueCache

            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            generator._overlay_plan.protected_cache = ProtectedValueCache.from_dict(payload)
        return generator


def _stub_source_df_from_profile(profile: SourceTableProfile) -> pd.DataFrame:
    """Rebuild minimal source context for fresh-column generation after load."""
    names = list(profile.columns.keys())
    data: Dict[str, List[Any]] = {}
    for name in names:
        column = profile.columns[name]
        if column.kind == ColumnKind.NUMERIC.value or column.kind == ColumnKind.ID_LIKE.value:
            mn = column.marginal.get("min", 1)
            mx = column.marginal.get("max", max(int(mn) + 1, profile.row_count * 10))
            data[name] = [mn, mx]
        else:
            data[name] = ["source", "value"]
    return pd.DataFrame(data)[names]


def _load_source_dataframe(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source)


def build_fidelity_report(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    profile: SourceTableProfile,
    *,
    fitted_model_type: Optional[str] = None,
    protected_cache: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compare marginals between source and synthetic data for UI/oracle reporting."""
    columns: List[Dict[str, Any]] = []
    for name, col_profile in profile.columns.items():
        if name not in source.columns or name not in synthetic.columns:
            continue
        src = source[name]
        syn = synthetic[name]
        entry: Dict[str, Any] = {
            "column": name,
            "kind": col_profile.kind,
            "source_null_rate": round(float(src.isna().mean()), 4),
            "synthetic_null_rate": round(float(syn.isna().mean()), 4),
        }
        if col_profile.kind == ColumnKind.NUMERIC.value:
            src_num = pd.to_numeric(src, errors="coerce").dropna()
            syn_num = pd.to_numeric(syn, errors="coerce").dropna()
            if len(src_num) and len(syn_num):
                entry["source_mean"] = round(float(src_num.mean()), 4)
                entry["synthetic_mean"] = round(float(syn_num.mean()), 4)
                entry["mean_delta_pct"] = round(
                    abs(float(syn_num.mean()) - float(src_num.mean()))
                    / max(abs(float(src_num.mean())), 1e-9)
                    * 100.0,
                    2,
                )
        elif col_profile.kind == ColumnKind.CATEGORICAL.value:
            src_counts = src.astype(str).value_counts(normalize=True).head(5)
            syn_counts = syn.astype(str).value_counts(normalize=True).head(5)
            overlap = float(src_counts.index.intersection(syn_counts.index).size) / max(len(src_counts), 1)
            entry["top_category_overlap"] = round(overlap, 4)
        elif col_profile.kind == ColumnKind.BOOLEAN.value:
            entry["source_true_rate"] = round(float(src.dropna().astype(bool).mean()), 4)
            entry["synthetic_true_rate"] = round(float(syn.dropna().astype(bool).mean()), 4)
        columns.append(entry)

    numeric_deltas = [c.get("mean_delta_pct", 0.0) for c in columns if "mean_delta_pct" in c]
    avg_numeric_delta = float(np.mean(numeric_deltas)) if numeric_deltas else 0.0
    privacy = validate_no_pii_replay(
        source,
        synthetic,
        pii_replay_column_names(profile),
        protected_cache=protected_cache,
    )
    unique_targets = fresh_column_names(profile) + identifier_column_names(profile)
    uniqueness = validate_row_uniqueness(synthetic, unique_targets)

    correlation_checks: List[Dict[str, Any]] = []
    numeric = source.select_dtypes(include=[np.number])
    syn_numeric = synthetic.select_dtypes(include=[np.number])
    for pair in profile.numeric_correlations[:10]:
        col_a = pair["column_a"]
        col_b = pair["column_b"]
        if col_a not in numeric.columns or col_b not in numeric.columns:
            continue
        if col_a not in syn_numeric.columns or col_b not in syn_numeric.columns:
            continue
        src_corr = float(numeric[[col_a, col_b]].corr(numeric_only=True).iloc[0, 1])
        syn_corr = float(syn_numeric[[col_a, col_b]].corr(numeric_only=True).iloc[0, 1])
        if pd.isna(src_corr) or pd.isna(syn_corr):
            continue
        correlation_checks.append(
            {
                "column_a": col_a,
                "column_b": col_b,
                "source_correlation": round(src_corr, 4),
                "synthetic_correlation": round(syn_corr, 4),
                "delta": round(abs(syn_corr - src_corr), 4),
            }
        )
    avg_corr_delta = (
        float(np.mean([item["delta"] for item in correlation_checks]))
        if correlation_checks
        else 0.0
    )
    passed = (
        avg_numeric_delta <= 35.0
        and avg_corr_delta <= 0.35
        and privacy["passed"]
        and uniqueness["passed"]
    )

    from synth_platform.engine.validation.schema.metrics.readiness import build_metrics_report
    from synth_platform.engine.inference.schema.schema import SchemaConfig, Table
    from synth_platform.engine.inference.schema.schema_columns import collect_categorical_review_issues, column_from_source_profile

    schema_columns = [column_from_source_profile(name, col_profile) for name, col_profile in profile.columns.items()]
    categorical_review = collect_categorical_review_issues(schema_columns)

    placeholder_schema = SchemaConfig(
        name=profile.table_name,
        tables=[Table(name=profile.table_name, row_count=len(synthetic))],
        columns={profile.table_name: schema_columns},
    )
    metrics = build_metrics_report(
        {profile.table_name: synthetic},
        placeholder_schema,
        source_tables={profile.table_name: source},
        source_profile=profile,
        generation_mode="source_driven",
        hard_validation_passed=passed,
    )

    return {
        "passed": passed,
        "generation_mode": "source_driven",
        "model_type": fitted_model_type or profile.recommended_model,
        "table_name": profile.table_name,
        "source_rows": profile.row_count,
        "synthetic_rows": len(synthetic),
        "avg_numeric_mean_delta_pct": round(avg_numeric_delta, 2),
        "avg_correlation_delta": round(avg_corr_delta, 4),
        "missingness": profile.missingness,
        "numeric_correlations": profile.numeric_correlations,
        "correlation_checks": correlation_checks,
        "privacy": privacy,
        "uniqueness": uniqueness,
        "sdv_columns": sdv_column_names(profile),
        "fresh_columns": fresh_column_names(profile),
        "columns": columns,
        "metrics": metrics,
        "categorical_review": categorical_review,
        "final_readiness": metrics.get("final_readiness", {}),
    }


def mimic_from_source(
    source: Union[str, Path, pd.DataFrame],
    *,
    rows: Optional[int] = None,
    model_type: Optional[str] = None,
    seed: Optional[int] = None,
    table_name: str = "source_table",
    epochs: Optional[int] = None,
    fit_max_rows: Optional[int] = None,
    profile_max_rows: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_artifacts: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], SourceTableProfile, Optional[SourceDrivenArtifacts], Dict[str, Any]]:
    """
    Source-driven mimic: profile, fit SDV, sample, and return tables + fidelity report.

    Returns ``({table_name: synthetic_df}, profile, artifacts_or_none, fidelity_report)``.
    """
    df = _load_source_dataframe(source)
    num_rows = int(rows if rows is not None else len(df))

    generator = SourceDrivenGenerator()
    profile = generator.fit(
        df,
        table_name=table_name,
        model_type=model_type,
        seed=seed,
        epochs=epochs,
        fit_max_rows=fit_max_rows,
        profile_max_rows=profile_max_rows,
    )
    synthetic = generator.sample(num_rows, seed=seed)
    fidelity = build_fidelity_report(
        df,
        synthetic,
        profile,
        fitted_model_type=generator.fitted_model_type(),
    )

    artifacts: Optional[SourceDrivenArtifacts] = None
    if save_artifacts:
        if output_dir is None:
            import tempfile

            output_dir = Path(tempfile.mkdtemp(prefix="mvp_source_driven_"))
        artifacts = generator.save(output_dir, synthetic_rows=num_rows, seed=seed)

    return {table_name: synthetic}, profile, artifacts, fidelity


def generate_from_source(
    source: Union[str, Path, pd.DataFrame],
    *,
    rows: Optional[int] = None,
    model_type: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    table_name: str = "source_table",
    epochs: Optional[int] = None,
    fit_max_rows: Optional[int] = None,
    profile_max_rows: Optional[int] = None,
    save_synthetic_csv: bool = True,
    chunk_size: Optional[int] = None,
    export_format: str = "csv",
) -> Tuple[pd.DataFrame, SourceDrivenArtifacts]:
    """
    Profile source data, fit an SDV synthesizer, optionally persist artifacts, and sample.

    Returns synthetic data and artifact locations. When *output_dir* is omitted, artifacts
    are written to a temporary directory that the caller should manage.
    """
    df = _load_source_dataframe(source)
    num_rows = int(rows if rows is not None else len(df))

    if output_dir is None:
        import tempfile

        output_dir = Path(tempfile.mkdtemp(prefix="mvp_source_driven_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    generator = SourceDrivenGenerator()
    generator.fit(
        df,
        table_name=table_name,
        model_type=model_type,
        seed=seed,
        epochs=epochs,
        fit_max_rows=fit_max_rows,
        profile_max_rows=profile_max_rows,
    )
    effective_chunk = int(chunk_size) if chunk_size else num_rows
    fmt = (export_format or "csv").lower()
    synthetic: pd.DataFrame

    if effective_chunk < num_rows and save_synthetic_csv:
        from synth_platform.engine.generation.schema.chunked_export import export_table_streaming

        preview_cap = min(5000, num_rows, effective_chunk)
        captured_preview: List[pd.DataFrame] = []

        def _chunks_with_preview() -> Iterator[pd.DataFrame]:
            for chunk in generator.sample_chunks(num_rows, effective_chunk, seed=seed):
                if not captured_preview:
                    captured_preview.append(chunk.head(preview_cap).copy())
                yield chunk

        export_table_streaming(table_name, _chunks_with_preview(), output_dir, fmt=fmt)
        synthetic = captured_preview[0] if captured_preview else generator.sample(preview_cap, seed=seed)
    else:
        synthetic = generator.sample(num_rows, seed=seed)
        if save_synthetic_csv:
            if fmt == "parquet":
                from synth_platform.engine.generation.schema.chunked_export import export_parquet_chunks

                export_parquet_chunks(iter([synthetic]), Path(output_dir) / f"{table_name}.parquet")
            else:
                synthetic.to_csv(Path(output_dir) / DEFAULT_SYNTHETIC_FILENAME, index=False)

    artifacts = generator.save(output_dir, synthetic_rows=num_rows, seed=seed)
    fidelity = build_fidelity_report(
        df,
        synthetic,
        generator.profile,  # type: ignore[arg-type]
        fitted_model_type=generator.fitted_model_type(),
    )
    fidelity_path = Path(output_dir) / "fidelity_report.json"
    fidelity_path.write_text(json.dumps(fidelity, indent=2), encoding="utf-8")

    if save_synthetic_csv:
        if fmt == "parquet":
            artifacts.synthetic_path = Path(output_dir) / f"{table_name}.parquet"
        elif effective_chunk >= num_rows:
            artifacts.synthetic_path = Path(output_dir) / DEFAULT_SYNTHETIC_FILENAME
        else:
            artifacts.synthetic_path = Path(output_dir) / (f"{table_name}.parquet" if fmt == "parquet" else f"{table_name}.csv")
    return synthetic, artifacts


def load_and_sample(
    artifacts_dir: Union[str, Path],
    *,
    rows: int,
    seed: Optional[int] = None,
    chunk_size: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    export_format: str = "csv",
    table_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load a persisted source-driven model and generate synthetic rows."""
    generator = SourceDrivenGenerator.load(artifacts_dir)
    name = table_name or (generator.profile.table_name if generator.profile else "source_table")
    num_rows = int(rows)
    effective_chunk = int(chunk_size) if chunk_size else num_rows

    if output_dir is not None and effective_chunk < num_rows:
        from synth_platform.engine.generation.schema.chunked_export import export_table_streaming

        preview_cap = min(5000, num_rows, effective_chunk)
        captured: List[pd.DataFrame] = []

        def _chunks() -> Iterator[pd.DataFrame]:
            for chunk in generator.sample_chunks(num_rows, effective_chunk, seed=seed):
                if not captured:
                    captured.append(chunk.head(preview_cap).copy())
                yield chunk

        export_table_streaming(name, _chunks(), output_dir, fmt=export_format)
        return captured[0] if captured else generator.sample(min(num_rows, preview_cap), seed=seed)

    return generator.sample(num_rows, seed=seed)


def run_source_driven_pipeline(
    source: Union[str, Path, pd.DataFrame],
    *,
    rows: Optional[int] = None,
    model_type: Optional[str] = None,
    output_dir: Union[str, Path],
    seed: Optional[int] = None,
    table_name: str = "source_table",
    epochs: Optional[int] = None,
    fit_max_rows: Optional[int] = None,
    profile_max_rows: Optional[int] = None,
    chunk_size: int = 50_000,
    export_format: str = "parquet",
    write_fidelity: bool = True,
) -> Dict[str, Any]:
    """End-to-end source-driven run with stage timings and performance metrics."""
    from synth_platform.engine.validation.schema.throughput import StageTimings, build_performance_report, time_stage, track_peak_memory

    df = _load_source_dataframe(source)
    num_rows = int(rows if rows is not None else len(df))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stages = StageTimings()
    peak_mb = 0.0
    export_info: Dict[str, Any] = {"rows_written": num_rows}

    with track_peak_memory() as peaks:
        generator = SourceDrivenGenerator()
        generator.fit(
            df,
            table_name=table_name,
            model_type=model_type,
            seed=seed,
            epochs=epochs,
            fit_max_rows=fit_max_rows,
            profile_max_rows=profile_max_rows,
            stages=stages,
        )

        preview_cap = min(5000, num_rows, chunk_size)
        captured: List[pd.DataFrame] = []

        with time_stage(stages, "sample_seconds"):
            if chunk_size < num_rows:
                from synth_platform.engine.generation.schema.chunked_export import export_table_streaming

                def _chunks() -> Iterator[pd.DataFrame]:
                    for chunk in generator.sample_chunks(num_rows, chunk_size, seed=seed):
                        if not captured:
                            captured.append(chunk.head(preview_cap).copy())
                        yield chunk

                with time_stage(stages, "export_seconds"):
                    export_info = export_table_streaming(
                        table_name,
                        _chunks(),
                        out,
                        fmt=export_format,
                    )
                preview = captured[0] if captured else generator.sample(preview_cap, seed=seed)
            else:
                preview = generator.sample(num_rows, seed=seed)
                from synth_platform.engine.generation.schema.chunked_export import export_table_streaming

                with time_stage(stages, "export_seconds"):
                    export_info = export_table_streaming(
                        table_name,
                        iter([preview]),
                        out,
                        fmt=export_format,
                    )

        artifacts = generator.save(out / "artifacts", synthetic_rows=num_rows, seed=seed)
        fidelity: Dict[str, Any] = {}
        if write_fidelity:
            fidelity = build_fidelity_report(
                df,
                preview,
                generator.profile,  # type: ignore[arg-type]
                fitted_model_type=generator.fitted_model_type(),
            )
            (out / "fidelity_report.json").write_text(json.dumps(fidelity, indent=2), encoding="utf-8")
        peak_mb = peaks[0]

    stages.total_seconds = (
        stages.profile_seconds + stages.fit_seconds + stages.sample_seconds + stages.export_seconds
    )
    performance = build_performance_report(
        generation_mode="source_driven",
        target_rows=num_rows,
        rows_generated=num_rows,
        chunk_size=chunk_size,
        stages=stages,
        peak_memory_mb=peak_mb,
        export_format=export_format,
        export_path=out,
        notes=[f"export_rows={export_info.get('rows_written', num_rows)}"],
    )
    perf_path = out / "performance_report.json"
    perf_path.write_text(json.dumps(performance.to_dict(), indent=2), encoding="utf-8")

    return {
        "preview": preview,
        "profile": generator.profile,
        "artifacts": artifacts,
        "fidelity": fidelity,
        "performance": performance.to_dict(),
        "export": export_info,
    }
