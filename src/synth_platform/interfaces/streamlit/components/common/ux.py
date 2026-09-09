"""Shared client-facing UX helpers for the Synthetic Data Twin Streamlit app.

Presentation only - no generation, training, or validation algorithms.
"""

from __future__ import annotations

import time
import tracemalloc
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import pandas as pd
import streamlit as st

from synth_platform.interfaces.streamlit.ui_config import ui_value

_CONTROLLED_TEXT_COLUMN_RE = re.compile(
    r"(^|_)(country|nation|country_?code|state|province|region|currency|language|locale|timezone|"
    r"city|town|company|organization|organisation|employer|business|merchant|vendor|supplier|payee|"
    r"retailer|store|facility|provider|hospital|clinic|status|status_?code|type|type_?code|segment|"
    r"tier|postal|zip|zipcode|postcode)(_|$)",
    re.IGNORECASE,
)


def platform_intro() -> None:
    st.caption(
        ui_value("app", "intro", default="One synthetic-data platform - multiple SSOT input workflows")
    )


def step_guide(*, what: str, next_step: str | None = None) -> None:
    """Short client-facing guidance under a step header (used across all modes)."""
    st.caption(what)
    if next_step:
        st.caption(f"Next: {next_step}")


def mode_outcomes(bullets: Sequence[str]) -> None:
    """What the user receives - keep short."""
    st.markdown("**What you receive**")
    for item in bullets:
        st.markdown(f"- {item}")


def render_progress(steps: Sequence[tuple[str, bool]], *, current_hint: str | None = None) -> None:
    """Compact workflow progress: each step is (label, done)."""
    parts: list[str] = []
    for label, done in steps:
        mark = "[x]" if done else "[ ]"
        parts.append(f"{mark} {label}")
    st.markdown(" -> ".join(parts))
    if current_hint:
        st.caption(current_hint)


def step_header(number: int, title: str, done: bool, *, help_text: str | None = None) -> None:
    icon = ":material/check_circle:" if done else ":material/radio_button_unchecked:"
    st.subheader(f"{icon} {number}. {title}", anchor=False)
    if help_text:
        st.caption(help_text)


def show_user_error(
    message: str,
    *,
    technical: object | None = None,
    next_action: str | None = None,
) -> None:
    """Client-facing error; optional expandable technical detail."""
    st.error(message)
    if next_action:
        st.caption(next_action)
    if technical is not None:
        with st.expander("Technical details", expanded=False):
            st.code(str(technical))


def show_success(message: str) -> None:
    st.success(message)


def validation_badge(passed: bool, *, passed_label: str = "Validation passed", failed_label: str = "Validation needs attention") -> None:
    if passed:
        st.success(passed_label)
    else:
        st.warning(failed_label)


def artifact_only_banner(*, disconnected: bool, has_artifact: bool) -> None:
    """Reflect real runtime state - never fake artifact-only generation."""
    if disconnected and has_artifact:
        st.info(
            "**Generating from the trained twin only.** The source database is disconnected. "
            "Output comes from the portable artifact - not live source access."
        )
    elif has_artifact and not disconnected:
        st.caption("Trained twin is ready. Disconnect the source before generating from the artifact alone.")
    elif disconnected and not has_artifact:
        st.warning("Source disconnected, but no trained twin is loaded yet.")


def friendly_table_description(table_name: str, *, existing: str | None = None) -> str:
    """Human-readable table blurb when the schema has no description."""
    if existing and str(existing).strip():
        return str(existing).strip()
    label = table_name.replace("_", " ").strip().title() or "Table"
    return f"{label} records"


@contextmanager
def measure_generation() -> Iterator[dict[str, float]]:
    """Capture wall time and peak traced memory for a generation block."""
    stats: dict[str, float] = {"seconds": 0.0, "peak_memory_mb": 0.0}
    tracemalloc.start()
    started = time.perf_counter()
    try:
        yield stats
    finally:
        stats["seconds"] = round(time.perf_counter() - started, 3)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats["peak_memory_mb"] = round(peak / (1024 * 1024), 2)


def render_run_metrics(
    *,
    seconds: float | None = None,
    peak_memory_mb: float | None = None,
    rows_per_second: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Simple trust strip: time taken and memory for the last generation."""
    cols = st.columns(4)
    cols[0].metric("Time taken", f"{seconds:.2f}s" if seconds is not None else "-")
    cols[1].metric(
        "Peak memory",
        f"{peak_memory_mb:.1f} MB" if peak_memory_mb is not None else "-",
    )
    cols[2].metric(
        "Throughput",
        f"{rows_per_second:,.0f} rows/s" if rows_per_second else "-",
    )
    note = (extra or {}).get("note") or "Measured for this run on your machine."
    cols[3].metric("Run scope", str((extra or {}).get("scope") or "this generation"))
    st.caption(str(note))


def render_integrity_metrics(
    *,
    hard_checks_passed: bool | None,
    fk_label: str,
    pk_label: str,
) -> None:
    c1, c2, c3 = st.columns(3)
    if hard_checks_passed is None:
        c1.metric("Integrity checks", "-")
    else:
        c1.metric("Integrity checks", "passed" if hard_checks_passed else "failed")
    c2.metric("FK integrity", fk_label)
    c3.metric("PK uniqueness", pk_label)


def render_data_drift_panel(
    *,
    mode: str,
    rows: Sequence[Mapping[str, Any]] | None = None,
    footnote: str | None = None,
) -> None:
    """Explain data drift in plain language; show a small table when available."""
    st.markdown("**Data drift / quality signal**")
    if mode == "schema":
        st.caption(str(ui_value("quality_signals", "schema_caption", default="")))
    elif mode == "database":
        st.caption(str(ui_value("quality_signals", "database_caption", default="")))
    elif mode == "pdf":
        st.caption(str(ui_value("quality_signals", "pdf_caption", default="")))
    if rows:
        st.dataframe(pd.DataFrame(list(rows)), hide_index=True, width="stretch")
    if footnote:
        st.caption(footnote)


def render_pk_fk_legend() -> None:
    st.caption(
        "**PK** = primary key (unique identity column). "
        "**FK** = foreign key (link to another table)."
    )


def list_contract_free_text_columns(contract: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Text-heavy / free_text columns from an approved dataset contract."""
    rows: list[dict[str, str]] = []
    if not contract:
        return rows
    for table_name, table in (contract.get("tables") or {}).items():
        for col_name, col in (table.get("columns") or {}).items():
            if not isinstance(col, Mapping):
                continue
            if str(col.get("semantic_type") or "").lower() != "free_text":
                continue
            if _CONTROLLED_TEXT_COLUMN_RE.search(str(col_name)):
                continue
            rows.append({"table": str(table_name), "column": str(col_name), "role": "free_text"})
    return rows


def relationship_volume_warnings(
    contract: Mapping[str, Any] | None,
    row_counts: Mapping[str, int],
    source_preview_tables: Mapping[str, Mapping[str, Any]] | None,
    *,
    ratio_tolerance: float = 0.35,
) -> list[str]:
    """Warn only when requested parent-child volume drifts from SSOT shape."""
    warnings: list[str] = []
    if not contract or not source_preview_tables:
        return warnings
    source_counts = {
        str(table_name): int(table.get("row_count") or 0)
        for table_name, table in source_preview_tables.items()
        if isinstance(table, Mapping)
    }
    if not source_counts:
        return warnings

    for child_table, table_contract in (contract.get("tables") or {}).items():
        if not isinstance(table_contract, Mapping):
            continue
        for fk in table_contract.get("foreign_keys") or []:
            if not isinstance(fk, Mapping):
                continue
            parent_table = fk.get("references_table") or fk.get("parent_table")
            if not parent_table or parent_table not in row_counts or child_table not in row_counts:
                continue
            source_parent = int(source_counts.get(str(parent_table)) or 0)
            source_child = int(source_counts.get(str(child_table)) or 0)
            requested_parent = int(row_counts.get(str(parent_table)) or 0)
            requested_child = int(row_counts.get(str(child_table)) or 0)
            if source_parent <= 0 or requested_parent <= 0:
                continue

            source_ratio = source_child / source_parent
            requested_ratio = requested_child / requested_parent
            if source_ratio <= 0:
                continue

            relative_delta = abs(requested_ratio - source_ratio) / source_ratio
            if relative_delta <= ratio_tolerance:
                continue

            direction = "lower" if requested_ratio < source_ratio else "higher"
            warnings.append(
                f"{child_table} requested parent-child volume is {direction} than the SSOT shape "
                f"({requested_ratio:.2f} vs {source_ratio:.2f} {child_table} per {parent_table})."
            )
    return warnings


def list_pdf_narrative_bindings(binding_map: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Narrative / notes-style PDF fields eligible for optional LLM text."""
    narrative_roles = {
        "narrative_text",
        "transaction_description",
        "medical_reason",
        "notes",
        "comments",
        "summary",
    }
    narrative_strategies = {
        "fake_narrative",
        "fake_transaction_description",
        "fake_medical_reason",
        "fake_generic_text",
    }
    rows: list[dict[str, str]] = []
    if not binding_map:
        return rows
    for binding in binding_map.get("bindings") or []:
        if binding.get("binding_type") != "field":
            continue
        role = str(binding.get("semantic_role") or "")
        strategy = str(binding.get("generator_strategy") or "")
        if role not in narrative_roles and strategy not in narrative_strategies:
            continue
        rows.append(
            {
                "label": str(binding.get("label") or binding.get("region_id") or ""),
                "role": role or strategy,
                "strategy": strategy,
            }
        )
    return rows


def fidelity_drift_rows(qa_report: Mapping[str, Any] | None, *, limit: int = 12) -> list[dict[str, Any]]:
    """Flatten QA fidelity checks into a small UI table (logic-driven drift)."""
    rows: list[dict[str, Any]] = []
    if not qa_report:
        return rows
    fidelity = ((qa_report.get("report") or {}).get("fidelity")) or {}
    for table_name, columns in fidelity.items():
        if not isinstance(columns, Mapping):
            continue
        for col_name, check in columns.items():
            if not isinstance(check, Mapping):
                continue
            kind = str(check.get("check") or "fidelity")
            if kind == "bounded_range":
                score = check.get("within_bounds_rate")
                detail = (
                    f"within [{check.get('generation_lower_bound')}, "
                    f"{check.get('generation_upper_bound')}]"
                )
            elif kind == "known_vocabulary":
                score = check.get("matched_rate")
                detail = f"{check.get('matched_count')}/{check.get('total_count')} known categories"
            else:
                score = None
                detail = str(check)
            rows.append(
                {
                    "table": table_name,
                    "column": col_name,
                    "check": kind,
                    "score": f"{float(score):.0%}" if isinstance(score, (int, float)) else "-",
                    "detail": detail,
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def profile_vs_synthetic_drift_rows(
    profile: Mapping[str, Any] | None,
    table_name: str,
    synthetic_df: pd.DataFrame,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Compare profiled source shape to a synthetic preview sample."""
    rows: list[dict[str, Any]] = []
    if profile is None or not table_name:
        return rows
    columns = ((profile.get("tables") or {}).get(table_name) or {}).get("columns") or {}
    for col_name, col_profile in columns.items():
        if col_name not in synthetic_df.columns:
            continue
        if not isinstance(col_profile, Mapping):
            continue
        src_null = float(col_profile.get("null_percentage") or 0)
        syn_null = float(synthetic_df[col_name].isna().mean() * 100)
        src_distinct = col_profile.get("distinct_count")
        syn_distinct = int(synthetic_df[col_name].nunique(dropna=True))
        rows.append(
            {
                "column": col_name,
                "source null %": round(src_null, 1),
                "synthetic null %": round(syn_null, 1),
                "source distinct": src_distinct if src_distinct is not None else "-",
                "synthetic distinct (sample)": syn_distinct,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def schema_integrity_signal_rows(
    *,
    hard_checks_passed: bool | None,
    fk_passed: bool | None,
    pk_passed: bool | None,
) -> list[dict[str, Any]]:
    """Return integrity evidence rows for source-free schema generation."""
    return [
        {
            "signal": "Integrity checks",
            "value": "-" if hard_checks_passed is None else ("passed" if hard_checks_passed else "failed"),
            "meaning": "Structural validation of the synthetic export",
        },
        {
            "signal": "FK integrity",
            "value": "-" if fk_passed is None else ("valid" if fk_passed else "failed"),
            "meaning": "Child keys resolve to parent keys",
        },
        {
            "signal": "PK uniqueness",
            "value": "-" if pk_passed is None else ("unique" if pk_passed else "duplicates"),
            "meaning": "Primary key columns have no duplicates",
        },
    ]


def apply_llm_free_text_to_tables(
    report: Mapping[str, Any],
    free_text_columns: Sequence[Mapping[str, str]],
    *,
    seed: int = 7,
    max_llm_rows: int = 50,
) -> list[dict[str, Any]]:
    """Post-pass: regenerate free_text CSV columns via TextGenerationEngine (no source text)."""
    from synth_platform.engine.inference.schema.schema import Column
    from synth_platform.engine.generation.text import TextGenerationConfig, generate_text_column

    evidence_rows: list[dict[str, Any]] = []
    by_table: dict[str, list[str]] = {}
    for item in free_text_columns:
        by_table.setdefault(str(item["table"]), []).append(str(item["column"]))

    for table_name, columns in by_table.items():
        entry = (report.get("tables") or {}).get(table_name)
        if not entry:
            continue
        path = Path(entry["path"])
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        changed = False
        for col_name in columns:
            if col_name not in df.columns:
                continue
            column = Column(
                name=col_name,
                type="text",
                distribution_params={
                    "text_type": "narrative",
                    "llm_enabled": True,
                    "llm_text": True,
                },
            )
            result = generate_text_column(
                table_name=table_name,
                column=column,
                size=len(df),
                table_data=df,
                config=TextGenerationConfig(
                    llm_enabled=True,
                    is_preview=False,
                    max_llm_rows=max_llm_rows,
                    seed=seed,
                ),
                source_series=None,
            )
            df[col_name] = result.values
            changed = True
            evidence_rows.append(
                {
                    "table": table_name,
                    "column": col_name,
                    "llm_used": bool(getattr(result.evidence, "llm_used", False)),
                    "fallback_count": int(getattr(result.evidence, "fallback_count", 0) or 0),
                    "rows": len(df),
                }
            )
        if changed:
            df.to_csv(path, index=False)
    return evidence_rows
