"""Generic optimized PII overlay for source-driven generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from synth_platform.engine.generation.schema.pii_columns import infer_value_semantic, is_text_heavy_column, is_true_identifier_column


@dataclass(frozen=True)
class ColumnOverlaySpec:
    column: str
    pii_kind: str
    requires_unique: bool
    semantic: Optional[str] = None


@dataclass
class ProtectedValueCache:
    """Normalized protected source values built once per fit."""

    column_sets: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    @classmethod
    def from_source(cls, source_df: pd.DataFrame, columns: Sequence[str]) -> ProtectedValueCache:
        sets: Dict[str, FrozenSet[str]] = {}
        for name in columns:
            if name not in source_df.columns:
                continue
            values = source_df[name].dropna().astype(str).str.strip().str.lower()
            sets[name] = frozenset(values.unique())
        return cls(column_sets=sets)

    def to_dict(self) -> Dict[str, List[str]]:
        return {name: sorted(values) for name, values in self.column_sets.items()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Sequence[str]]) -> "ProtectedValueCache":
        return cls(column_sets={name: frozenset(values) for name, values in payload.items()})

    def replay_count(self, synthetic_df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, int]:
        replays: Dict[str, int] = {}
        for name in columns:
            if name not in synthetic_df.columns or name not in self.column_sets:
                continue
            protected = self.column_sets[name]
            if not protected:
                replays[name] = 0
                continue
            syn = synthetic_df[name].dropna().astype(str).str.strip().str.lower()
            replays[name] = int(syn.isin(protected).sum())
        return replays

    def total_replay(self, synthetic_df: pd.DataFrame, columns: Sequence[str]) -> int:
        return int(sum(self.replay_count(synthetic_df, columns).values()))


@dataclass
class SourceOverlayPlan:
    """Precomputed overlay strategy for all output columns."""

    output_columns: List[str]
    sdv_columns: List[str]
    fresh_columns: List[str]
    privacy_columns: List[str]
    column_specs: Dict[str, ColumnOverlaySpec]
    protected_cache: ProtectedValueCache
    has_fresh_overlay: bool

    def replay_report(self, synthetic: pd.DataFrame) -> Dict[str, Any]:
        check_cols = self.privacy_columns or self.fresh_columns
        column_replays = self.protected_cache.replay_count(synthetic, check_cols)
        total = int(sum(column_replays.values()))
        return {
            "passed": total == 0,
            "column_replays": column_replays,
            "total_replay_values": total,
        }


def _pii_kind(column_name: str, series: pd.Series, semantic: Optional[str]) -> str:
    sem = semantic or infer_value_semantic(column_name, series)
    if sem:
        return sem
    if is_true_identifier_column(column_name, series):
        return "id_like"
    return "generic"


def build_overlay_plan(
    profile: Any,
    source_df: pd.DataFrame,
    *,
    output_columns: Sequence[str],
    sdv_columns: Sequence[str],
    fresh_columns: Sequence[str],
) -> SourceOverlayPlan:
    from synth_platform.engine.inference.schema.source_driven import fresh_column_names, pii_replay_column_names

    fresh = list(fresh_columns or fresh_column_names(profile))
    privacy = list(pii_replay_column_names(profile))
    specs: Dict[str, ColumnOverlaySpec] = {}
    for name in output_columns:
        if name not in profile.columns:
            continue
        column = profile.columns[name]
        series = source_df[name] if name in source_df.columns else pd.Series(dtype=object)
        kind = _pii_kind(name, series, getattr(column, "semantic_type", None))
        requires_unique = bool(name in fresh or column.kind == "id_like")
        specs[name] = ColumnOverlaySpec(
            column=name,
            pii_kind=kind,
            requires_unique=requires_unique,
            semantic=getattr(column, "semantic_type", None),
        )
    return SourceOverlayPlan(
        output_columns=list(output_columns),
        sdv_columns=list(sdv_columns),
        fresh_columns=fresh,
        privacy_columns=privacy,
        column_specs=specs,
        protected_cache=ProtectedValueCache.from_source(source_df, fresh),
        has_fresh_overlay=bool(fresh),
    )


def _batch_faker_values(
    *,
    kind: str,
    count: int,
    seed: Optional[int],
    unique: bool,
) -> List[Any]:
    from synth_platform.engine.generation.schema.pii_columns import generate_fresh_column

    if count <= 0:
        return []
    # Delegate to optimized batch path inside generate_fresh_column_batch
    return generate_fresh_column_batch(
        column_name=kind,
        source_series=pd.Series(dtype=object),
        num_rows=count,
        seed=seed,
        semantic=kind,
        unique=unique,
    ).tolist()


def apply_overlay_plan(
    synthetic: pd.DataFrame,
    plan: SourceOverlayPlan,
    source_df: pd.DataFrame,
    *,
    seed: Optional[int] = None,
    profile: Optional[Any] = None,
    row_offset: int = 0,
) -> pd.DataFrame:
    """Apply fresh PII overlay to an SDV chunk using a precomputed plan."""
    num_rows = len(synthetic)
    if num_rows <= 0:
        raise ValueError("Synthetic sample must contain at least one row.")

    if not plan.has_fresh_overlay and set(plan.output_columns) <= set(synthetic.columns):
        return synthetic[plan.output_columns]

    fresh_names = set(plan.fresh_columns)
    sdv_names = set(plan.sdv_columns)
    missing = [name for name in plan.output_columns if name not in synthetic.columns]
    if not plan.has_fresh_overlay and missing:
        result = synthetic.copy()
        for name in missing:
            spec = plan.column_specs.get(name)
            col_profile = profile.columns[name] if profile is not None and name in profile.columns else None
            from synth_platform.engine.inference.schema.source_driven import ColumnKind
            from synth_platform.engine.generation.schema.pii_columns import generate_fresh_column_batch

            if col_profile is not None and getattr(col_profile, "kind", None) == ColumnKind.ID_LIKE.value:
                col_seed = None if seed is None else int(seed) + abs(hash(name)) % 10000
                source_series = source_df[name] if name in source_df.columns else pd.Series(dtype=object)
                result[name] = generate_fresh_column_batch(
                    name,
                    source_series,
                    num_rows,
                    seed=col_seed,
                    semantic=spec.semantic if spec else None,
                    unique=bool(spec.requires_unique if spec else True),
                    row_offset=row_offset,
                    protected_values=plan.protected_cache.column_sets.get(name),
                ).to_numpy()
        return _scrub_overlay_replays(result[plan.output_columns], plan, row_offset=row_offset)
    columns: Dict[str, Any] = {}
    for name in plan.output_columns:
        spec = plan.column_specs.get(name)
        col_profile = profile.columns[name] if profile is not None and name in profile.columns else None
        from synth_platform.engine.inference.schema.source_driven import ColumnKind

        needs_fresh = name in fresh_names or (
            col_profile is not None
            and not getattr(col_profile, "use_sdv", True)
            and getattr(col_profile, "kind", None) == ColumnKind.ID_LIKE.value
        )
        if not needs_fresh and name in synthetic.columns:
            columns[name] = synthetic[name].to_numpy()
            continue
        if needs_fresh:
            col_seed = None if seed is None else int(seed) + abs(hash(name)) % 10000
            source_series = source_df[name] if name in source_df.columns else pd.Series(dtype=object)
            from synth_platform.engine.generation.schema.pii_columns import generate_fresh_column_batch

            columns[name] = generate_fresh_column_batch(
                name,
                source_series,
                num_rows,
                seed=col_seed,
                semantic=spec.semantic if spec else None,
                unique=bool(spec.requires_unique if spec else True),
                row_offset=row_offset,
                protected_values=plan.protected_cache.column_sets.get(name),
            ).to_numpy()
        elif name in synthetic.columns:
            columns[name] = synthetic[name].to_numpy()
        elif name in sdv_names:
            columns[name] = synthetic[name].to_numpy() if name in synthetic.columns else np.full(num_rows, None, dtype=object)
        else:
            columns[name] = np.full(num_rows, None, dtype=object)

    return _scrub_overlay_replays(pd.DataFrame(columns)[plan.output_columns], plan, row_offset=row_offset)


def _scrub_overlay_replays(
    frame: pd.DataFrame,
    plan: SourceOverlayPlan,
    *,
    row_offset: int = 0,
) -> pd.DataFrame:
    """Final guarantee: no fresh-overlay column reuses a protected source value."""
    if frame.empty or not plan.fresh_columns:
        return frame
    result = frame.copy()
    for name in plan.fresh_columns:
        if name not in result.columns:
            continue
        protected = plan.protected_cache.column_sets.get(name)
        if not protected:
            continue
        series = result[name]
        for idx, value in enumerate(series.tolist()):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if str(value).strip().lower() in protected:
                result.iat[idx, result.columns.get_loc(name)] = (
                    f"synthetic_{name.replace(' ', '_')}_{row_offset + idx}"
                )
    return result
