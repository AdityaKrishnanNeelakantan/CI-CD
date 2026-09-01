"""Database twin: Database A -> discover -> profile -> infer semantics ->
train -> export -> (disconnect) -> load -> generate (with distribution
tweaking) -> write to Database B -> validate.

UI-focused: every step below calls straight into the already-tested
src/ pipeline modules. No business logic lives in this file.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from synth_platform.settings import Settings
from synth_platform.interfaces.streamlit.components.common.intent_presets import database_intent_preset
from synth_platform.interfaces.streamlit.components.common.ux import (
    apply_llm_free_text_to_tables,
    artifact_only_banner,
    fidelity_drift_rows,
    friendly_table_description,
    list_contract_free_text_columns,
    measure_generation,
    mode_outcomes,
    platform_intro,
    profile_vs_synthetic_drift_rows,
    render_data_drift_panel,
    render_integrity_metrics,
    render_pk_fk_legend,
    render_progress,
    render_run_metrics,
    relationship_volume_warnings,
    show_success,
    show_user_error,
    step_guide,
    step_header,
    validation_badge,
)
from synth_platform.interfaces.streamlit.ui_config import ui_step, ui_value
from synth_platform.application.workflows.database_twin import (
    PROFILE_FILENAME,
    RunManifest,
    SQLiteSourceAdapter,
    build_sample_database,
    get_synthesizer_adapter_class,
    load_artifact,
    load_dataset_contract,
    load_profile,
    load_qa_report,
    load_relational_generation_report,
    load_target_write_report,
    load_training_report,
    recommend_synthesizer,
    run_artifact_export,
    run_contract_approval,
    run_discovery,
    run_inference,
    run_profiling,
    run_qa_validation,
    run_relational_generation,
    run_target_write,
    run_training_and_sampling,
    summarize_source_preview,
)

SETTINGS = Settings.from_env()
DATABASE_UI = ui_value("database", default={})

MODEL_TYPE_LABELS = DATABASE_UI.get(
    "model_type_labels",
    {
        "safe_gaussian_copula": "Standard ML twin",
        "dp_gaussian_copula": "ML twin with differential privacy",
        "sdv_gaussian_copula": "SDV Gaussian copula",
    },
)

STATISTICAL_SEMANTIC_TYPES = {"numerical", "category", "boolean", "datetime"}
NON_STATISTICAL_SEMANTIC_TYPES = {"identifier", "email", "person_name", "phone_number", "free_text"}

st.title(DATABASE_UI.get("title", "Database Twin"))
platform_intro()
st.caption(
    DATABASE_UI.get(
        "caption",
        "Learn from an existing database, download a portable trained twin, disconnect the source, "
        "then generate and validate synthetic data from the artifact alone.",
    )
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "db_workdir": None,
    "db_source_path": None,
    "db_source_preview": None,
    "db_uploaded_file_id": None,
    "db_manifest": None,
    "db_discovery": None,
    "db_profile": None,
    "db_candidates": None,
    "db_contract": None,
    "db_model_type": "safe_gaussian_copula",
    "db_dp_epsilon": 3.0,
    "db_dp_bounds": {},
    "db_training_report": None,
    "db_artifact_path": None,
    "db_source_disconnected": False,
    "db_loaded_artifact": None,
    "db_row_counts": {},
    "db_row_counts_initialized": False,
    "db_target_total_rows": None,
    "db_category_overrides": {},
    "db_relational_report": None,
    "db_qa_report": None,
    "db_target_path": None,
    "db_write_report": None,
    "db_chunk_size": None,
    "db_run_metrics": None,
    "db_llm_text": False,
    "db_llm_evidence": None,
    "db_intent": DATABASE_UI.get("default_intent", "Development & testing"),
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

if st.session_state.db_workdir is None:
    st.session_state.db_workdir = Path(tempfile.mkdtemp(prefix="db_twin_demo_"))


def _progress_steps() -> list[tuple[str, bool]]:
    return [
        ("Intent", True),
        ("Connect", st.session_state.db_source_path is not None),
        ("Discover", st.session_state.db_discovery is not None),
        ("Understand", st.session_state.db_contract is not None),
        ("Train", st.session_state.db_training_report is not None),
        ("Download twin", st.session_state.db_artifact_path is not None),
        ("Disconnect", st.session_state.db_source_disconnected),
        ("Generate", st.session_state.db_relational_report is not None),
        ("Validate", st.session_state.db_qa_report is not None),
        ("Export", st.session_state.db_write_report is not None),
    ]


render_progress(
    _progress_steps(),
    current_hint="Source -> Understand -> Train twin -> Download artifact -> Disconnect -> Generate -> Validate -> Export",
)

def _reset_downstream(from_key: str) -> None:
    """Clear every session-state key that depends on `from_key`, so
    re-running an earlier step doesn't leave stale later-step results
    displayed alongside inconsistent new ones.
    """
    order = [
        "db_source_path", "db_source_preview", "db_discovery", "db_profile", "db_candidates", "db_contract",
        "db_training_report", "db_artifact_path", "db_source_disconnected", "db_loaded_artifact",
        "db_row_counts", "db_row_counts_initialized", "db_target_total_rows",
        "db_relational_report", "db_qa_report", "db_target_path", "db_write_report",
    ]
    if from_key not in order:
        return
    for key in order[order.index(from_key) + 1 :]:
        st.session_state[key] = defaults.get(key)
    # Stage outputs are immutable per run - a redo must start a new run dir.
    st.session_state.db_manifest = None


def _active_manifest() -> RunManifest:
    """Return the session run, creating one if needed."""
    manifest = st.session_state.db_manifest
    workdir = st.session_state.db_workdir
    if manifest is None or manifest.run_dir.parent.parent != workdir:
        manifest = RunManifest.create(runs_dir=workdir / "runs")
        st.session_state.db_manifest = manifest
    return manifest


def _render_source_preview(preview: dict) -> None:
    tables = preview.get("tables") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SSOT tables", int(preview.get("table_count") or len(tables)))
    c2.metric("SSOT rows", f"{int(preview.get('total_rows') or 0):,}")
    c3.metric("SSOT columns", f"{int(preview.get('total_columns') or 0):,}")
    c4.metric("Source type", str(preview.get("source_type") or "database"))

    overview_rows = [
        {
            "table": table_name,
            "source rows": f"{int(table.get('row_count') or 0):,}",
            "columns": int(table.get("column_count") or len(table.get("columns") or [])),
            "preview rows": len(table.get("sample_rows") or []),
        }
        for table_name, table in tables.items()
    ]
    if overview_rows:
        st.dataframe(overview_rows, hide_index=True, width="stretch")
        selected = st.selectbox("SSOT preview table", list(tables), key="db_ssot_preview_table")
        selected_table = tables[selected]
        sample_rows = selected_table.get("sample_rows") or []
        if sample_rows:
            st.dataframe(pd.DataFrame(sample_rows), hide_index=True, width="stretch")
            st.caption(
                f"`{selected}` live source preview: {int(selected_table.get('row_count') or 0):,} rows, "
                f"{int(selected_table.get('column_count') or 0):,} columns "
                f"(showing {len(sample_rows)})."
            )
        else:
            st.caption(f"`{selected}` is visible but has no preview rows.")


def _profile_columns(table_profile: dict) -> dict[str, dict]:
    columns = table_profile.get("columns") or {}
    if isinstance(columns, dict):
        return {str(name): col if isinstance(col, dict) else {"value": col} for name, col in columns.items()}
    normalized: dict[str, dict] = {}
    if isinstance(columns, list):
        for index, col in enumerate(columns):
            if isinstance(col, dict):
                name = str(col.get("name") or col.get("column") or f"column_{index + 1}")
                normalized[name] = col
            else:
                normalized[str(col)] = {"name": str(col)}
    return normalized


def _render_profile_summary(profile: dict) -> None:
    tables = profile.get("tables") or {}
    table_count = len(tables)
    row_count = sum(int((table or {}).get("row_count") or 0) for table in tables.values() if isinstance(table, dict))
    column_count = sum(len(_profile_columns(table)) for table in tables.values() if isinstance(table, dict))
    warnings_count = sum(
        len((table or {}).get("warnings") or [])
        for table in tables.values()
        if isinstance(table, dict)
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profiled tables", table_count)
    c2.metric("Profiled rows", f"{row_count:,}")
    c3.metric("Profiled columns", f"{column_count:,}")
    c4.metric("Warnings", warnings_count)

    if not tables:
        st.warning("No profiled tables were found. Re-run discovery and profiling.")
        return

    table_name = st.selectbox("Table", list(tables), key="profile_table_select")
    table_profile = tables.get(table_name) or {}
    if not isinstance(table_profile, dict):
        st.warning("This table profile is malformed. Re-run profiling.")
        return

    meta = table_profile.get("execution_metadata") or {}
    m1, m2, m3 = st.columns(3)
    m1.metric("Sample rows", f"{int(table_profile.get('row_count') or 0):,}")
    m2.metric("Profiler time", f"{float(meta.get('duration_ms') or 0):,.1f} ms")
    m3.metric("Correlations", len(table_profile.get("correlations") or []))

    rows = []
    for name, col in _profile_columns(table_profile).items():
        warnings = col.get("warnings") or []
        if isinstance(warnings, str):
            warning_text = warnings
        else:
            warning_text = ", ".join(str(w) for w in warnings) or "-"
        rows.append(
            {
                "column": name,
                "dtype": col.get("inferred_dtype") or col.get("physical_type") or col.get("dtype") or "-",
                "null %": col.get("null_percentage", col.get("missing_rate", "-")),
                "distinct": col.get("distinct_count", col.get("cardinality", "-")),
                "warnings": warning_text,
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.warning("No column profiles were recorded for this table. Re-run profiling.")


def _model_type_label(model_type: str) -> str:
    return MODEL_TYPE_LABELS.get(model_type, model_type)


def _training_plan_rows(contract: dict, model_type: str, sample_limit: int, sample_rows: int) -> list[dict]:
    rows = []
    for table_name, table_contract in (contract.get("tables") or {}).items():
        approved_columns = [
            col_name
            for col_name, col in (table_contract.get("columns") or {}).items()
            if col.get("inference_status") == "approved"
        ]
        statistical_columns = [
            col_name
            for col_name, col in (table_contract.get("columns") or {}).items()
            if col.get("inference_status") == "approved"
            and col.get("semantic_type") in STATISTICAL_SEMANTIC_TYPES
        ]
        generated_context_columns = [
            col_name
            for col_name, col in (table_contract.get("columns") or {}).items()
            if col.get("inference_status") == "approved"
            and (col.get("semantic_type") in NON_STATISTICAL_SEMANTIC_TYPES or col.get("sensitive"))
        ]
        rows.append(
            {
                "table": table_name,
                "ML model": _model_type_label(model_type),
                "training sample cap": int(sample_limit),
                "approved columns": len(approved_columns),
                "statistical fit columns": len(statistical_columns),
                "privacy/generated columns": len(generated_context_columns),
                "validation sample rows": int(sample_rows),
            }
        )
    return rows


def _render_training_report(report: dict, contract: dict) -> None:
    tables = report.get("tables") or {}
    total_trained_columns = 0
    total_excluded_columns = 0
    total_validation_rows = 0
    total_fit_rows = 0
    total_duration_ms = 0.0
    overview_rows = []

    for table_name, table_report in tables.items():
        evidence = table_report.get("fit_evidence") or {}
        trace = table_report.get("training_trace") or {}
        trained_columns = evidence.get("trained_columns") or []
        excluded_columns = evidence.get("excluded_columns") or []
        masked_columns = (table_report.get("pre_input_mask") or {}).get("context_aware_columns") or []
        total_trained_columns += len(trained_columns)
        total_excluded_columns += len(excluded_columns)
        total_validation_rows += int(table_report.get("generated_row_count") or 0)
        total_fit_rows += int(trace.get("fit_rows") or 0)
        total_duration_ms += float(trace.get("total_duration_ms") or 0.0)
        overview_rows.append(
            {
                "table": table_name,
                "ML model": _model_type_label(str(table_report.get("model_type") or "")),
                "fit rows": int(trace.get("fit_rows") or 0),
                "trained columns": len(trained_columns),
                "excluded columns": len(excluded_columns),
                "privacy-masked columns": len(masked_columns),
                "validation sample rows": int(table_report.get("generated_row_count") or 0),
                "training seconds": round(float(trace.get("total_duration_ms") or 0.0) / 1000, 2),
                "artifact format": table_report.get("serialization_format") or "-",
            }
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models trained", len(tables))
    c2.metric("Columns learned", total_trained_columns)
    c3.metric("Fit rows", f"{total_fit_rows:,}")
    c4.metric("Training time", f"{total_duration_ms / 1000:.2f}s")
    if total_excluded_columns:
        st.caption(f"{total_excluded_columns} column(s) were excluded from statistical fit by contract or safeguards.")
    st.dataframe(pd.DataFrame(overview_rows), hide_index=True, width="stretch")

    for table_name, table_report in tables.items():
        evidence = table_report.get("fit_evidence") or {}
        trace = table_report.get("training_trace") or {}
        table_contract = (contract.get("tables") or {}).get(table_name, {})
        with st.expander(f"{table_name} training details", expanded=False):
            trained_columns = evidence.get("trained_columns") or []
            trained_rows = []
            for column_name in trained_columns:
                column_contract = (table_contract.get("columns") or {}).get(column_name, {})
                trained_rows.append(
                    {
                        "column": column_name,
                        "semantic type": column_contract.get("semantic_type") or "-",
                        "role": "statistical fit"
                        if column_contract.get("semantic_type") in STATISTICAL_SEMANTIC_TYPES
                        else "generated from safe context",
                    }
                )
            if trained_rows:
                st.markdown("**Columns in the trained model output**")
                st.dataframe(pd.DataFrame(trained_rows), hide_index=True, width="stretch")

            excluded_columns = evidence.get("excluded_columns") or []
            if excluded_columns:
                st.markdown("**Excluded from statistical fit**")
                st.dataframe(pd.DataFrame(excluded_columns), hide_index=True, width="stretch")

            masked_columns = (table_report.get("pre_input_mask") or {}).get("context_aware_columns") or []
            if masked_columns:
                st.markdown("**Pre-fit privacy mask**")
                st.dataframe(
                    pd.DataFrame({"column": masked_columns, "safeguard": "Faker/context replacement before fit"}),
                    hide_index=True,
                    width="stretch",
                )

            timings = trace.get("phase_durations_ms") or {}
            if timings:
                timing_rows = [
                    {"phase": phase.replace("_", " "), "milliseconds": round(float(duration), 2)}
                    for phase, duration in timings.items()
                ]
                st.markdown("**Training phases**")
                st.dataframe(pd.DataFrame(timing_rows), hide_index=True, width="stretch")

            if "privacy_summary" in evidence:
                privacy = evidence["privacy_summary"]
                st.markdown("**Differential privacy accounting**")
                st.dataframe(
                    [
                        {
                            "epsilon spent": round(float(privacy.get("total_epsilon_spent") or 0.0), 4),
                            "epsilon budget": round(float(privacy.get("epsilon_budget") or 0.0), 4),
                            "delta budget": privacy.get("delta_budget"),
                        }
                    ],
                    hide_index=True,
                    width="stretch",
                )

            st.caption(
                f"Model artifact: `{Path(table_report.get('model_path', '')).name}`; "
                f"validation sample: `{Path(table_report.get('sample_path', '')).name}`."
            )


def _source_counts_by_table(source_preview_tables: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table_name, table in source_preview_tables.items():
        try:
            row_count = int(table.get("row_count") or 0)
        except (TypeError, ValueError):
            row_count = 0
        if row_count > 0:
            counts[str(table_name)] = row_count
    return counts


def _relationship_adjusted_defaults(table_names: list[str], contract: dict) -> dict[str, int]:
    defaults = {table_name: SETTINGS.db_default_parent_rows for table_name in table_names}
    child_tables = set()
    for child_table, table_contract in (contract.get("tables") or {}).items():
        for fk in table_contract.get("foreign_keys") or []:
            parent_table = fk.get("references_table") or fk.get("parent_table")
            if parent_table in defaults and child_table in defaults:
                child_tables.add(child_table)
    for table_name in child_tables:
        defaults[table_name] = SETTINGS.db_default_child_rows
    return defaults


def _scale_counts(counts: dict[str, int], target_total: int) -> dict[str, int]:
    if not counts:
        return {}
    current_total = sum(max(0, int(value)) for value in counts.values())
    if current_total <= 0:
        return {table_name: 1 for table_name in counts}
    scaled = {
        table_name: max(1, int(round((int(value) / current_total) * target_total)))
        for table_name, value in counts.items()
    }
    drift = int(target_total) - sum(scaled.values())
    if drift:
        largest_table = max(scaled, key=scaled.get)
        scaled[largest_table] = max(1, scaled[largest_table] + drift)
    return scaled


def _recommended_row_counts(
    table_names: list[str],
    source_preview_tables: dict,
    contract: dict,
    target_total: int | None = None,
) -> tuple[dict[str, int], str]:
    source_counts = _source_counts_by_table(source_preview_tables)
    if source_counts:
        basis = {table_name: source_counts.get(table_name, 1) for table_name in table_names}
        default_total = sum(basis.values())
        if target_total is None:
            target_total = min(
                max(default_total, SETTINGS.db_min_recommended_total_rows),
                SETTINGS.db_max_recommended_total_rows,
            )
        return _scale_counts(basis, int(target_total)), "SSOT row-shape"

    basis = _relationship_adjusted_defaults(table_names, contract)
    if target_total is None:
        target_total = sum(basis.values())
    return _scale_counts(basis, int(target_total)), "relationship-shape"


def _apply_row_count_defaults(defaults_by_table: dict[str, int]) -> None:
    st.session_state.db_row_counts = {table: int(count) for table, count in defaults_by_table.items()}
    for table_name, count in defaults_by_table.items():
        st.session_state[f"rowcount_{table_name}"] = int(count)
    st.session_state.db_row_counts_initialized = True


def _manifest_for_new_outputs(
    *filenames: str,
    extra_paths: list[Path] | None = None,
) -> RunManifest:
    """Return a writable run for the given stage outputs.

    Pipeline stages refuse to overwrite existing files. If this session still
    points at a run that already wrote those outputs (e.g. train succeeded on
    disk but session state was cleared, or the user is retrying), start a new
    run instead of raising RuntimeError.
    """
    manifest = _active_manifest()
    collisions = [manifest.output_path(name).exists() for name in filenames]
    if extra_paths:
        collisions.extend(path.exists() for path in extra_paths)
    if any(collisions):
        manifest = RunManifest.create(runs_dir=st.session_state.db_workdir / "runs")
        st.session_state.db_manifest = manifest
    return manifest


# ---------------------------------------------------------------------------
# Step 1 - Intent
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "intent")
    step_header(1, step.get("title", "Intent"), True)
    step_guide(
        what=step.get(
            "what",
            "Records the business purpose for this run. It does not change policy or generation unless a configured backend policy says so.",
        ),
        next_step=step.get("next", "Connect a SQLite source database."),
    )
    intent_options = DATABASE_UI.get("intent_options", ["Development & testing"])
    st.session_state.db_intent = st.selectbox(
        "What will you use the synthetic twin for?",
        options=intent_options,
        index=intent_options.index(st.session_state.db_intent)
        if st.session_state.db_intent in intent_options
        else 0,
    )

# ---------------------------------------------------------------------------
# Step 2 - Source database
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "connect")
    step_header(2, step.get("title", "Connect"), st.session_state.db_source_path is not None)
    step_guide(
        what=step.get("what", "Provide a SQLite database to learn from (upload or generate a sample)."),
        next_step=step.get("next", "Discover tables and relationships."),
    )

    source_options = DATABASE_UI.get("source_options", ["Use a generated sample database", "Upload a SQLite file"])
    source_choice = st.radio(
        "Source",
        source_options,
        horizontal=True,
        label_visibility="collapsed",
    )

    if source_choice == source_options[0]:
        sample_entity_count = st.slider(
            "Sample entities to generate",
            SETTINGS.db_sample_database_min_entities,
            SETTINGS.db_sample_database_max_entities,
            int(database_intent_preset(st.session_state.db_intent).sample_entity_default),
            step=SETTINGS.db_sample_database_step_entities,
        )
        if st.button("Generate sample database", icon=":material/auto_awesome:"):
            path = st.session_state.db_workdir / "database_a.db"
            build_sample_database(path, seed=SETTINGS.seed, customer_count=sample_entity_count)
            st.session_state.db_source_path = path
            st.session_state.db_manifest = None
            _reset_downstream("db_source_path")
    else:
        uploaded = st.file_uploader("SQLite database file", type=["db", "sqlite", "sqlite3"])
        # st.file_uploader keeps returning the same UploadedFile on every
        # rerun for as long as it stays uploaded - unlike st.button, which
        # only fires True on the one rerun right after the click. Without
        # this file_id guard, "new source connected" (reset manifest, wipe
        # every downstream step) re-ran on *every* rerun this page ever
        # does, including the rerun run_discovery() itself triggers right
        # after succeeding - discovery could never survive past its own
        # completion. Confirmed by testing this exact path against a real
        # uploaded database: the button reappeared with db_discovery back
        # at None immediately after a verified-successful discovery run.
        if uploaded is not None and uploaded.file_id != st.session_state.db_uploaded_file_id:
            path = st.session_state.db_workdir / "database_a.db"
            path.write_bytes(uploaded.getvalue())
            st.session_state.db_source_path = path
            st.session_state.db_uploaded_file_id = uploaded.file_id
            st.session_state.db_manifest = None
            _reset_downstream("db_source_path")

    if st.session_state.db_source_path is not None:
        preview_adapter = SQLiteSourceAdapter({"path": str(st.session_state.db_source_path)})
        if st.session_state.db_source_preview is None:
            try:
                st.session_state.db_source_preview = summarize_source_preview(
                    preview_adapter,
                    sample_rows=SETTINGS.ui_source_preview_rows,
                )
            except Exception as exc:
                show_user_error(
                    "Connected, but couldn't read the live source preview.",
                    technical=exc,
                    next_action="You can still try discovery, or upload a readable SQLite database.",
                )
        tables = list((st.session_state.db_source_preview or {}).get("tables") or {})
        if st.session_state.db_source_preview is not None:
            st.markdown("**Live SSOT preview**")
            _render_source_preview(st.session_state.db_source_preview)
        show_success(f"Connected to `{st.session_state.db_source_path.name}` - tables: {', '.join(tables)}")

if st.session_state.db_source_path is None:
    st.stop()

source_adapter = SQLiteSourceAdapter({"path": str(st.session_state.db_source_path)})
manifest = _active_manifest()
metadata_dir = st.session_state.db_workdir / "metadata"

# ---------------------------------------------------------------------------
# Step 2 - Discover
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "discover")
    step_header(3, step.get("title", "Discover"), st.session_state.db_discovery is not None)
    step_guide(
        what=step.get("what", "Inspect tables, columns, primary keys, and foreign keys."),
        next_step=step.get("next", "Understand values and meaning."),
    )
    if st.session_state.db_discovery is None:
        if st.button("Discover structure", icon=":material/travel_explore:"):
            manifest = _manifest_for_new_outputs("discovery.json")
            result = run_discovery(source_adapter, manifest, config_path="config/project.yaml")
            if not result.is_success():
                show_user_error(
                    "We couldn't discover your database structure.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_discovery = json.loads((manifest.run_dir / "discovery.json").read_text())
            st.rerun()
    else:
        discovery = st.session_state.db_discovery
        render_pk_fk_legend()
        overview_rows = []
        for table_name, table in discovery["tables"].items():
            pks = table.get("primary_key") or []
            if isinstance(pks, str):
                pks = [pks]
            fk_bits = []
            for fk in table.get("foreign_keys") or []:
                fk_bits.append(
                    f"{fk.get('column')} -> {fk.get('references_table')}.{fk.get('references_column')}"
                )
            overview_rows.append(
                {
                    "table": table_name,
                    "rows (source)": table.get("estimated_row_count") if table.get("estimated_row_count") is not None else "-",
                    "columns": len(table.get("columns") or {}),
                    "primary_key": ", ".join(pks) if pks else "-",
                    "foreign_keys": "; ".join(fk_bits) if fk_bits else "-",
                    "description": friendly_table_description(table_name),
                }
            )
        total_discovered_rows = sum(
            int(table.get("estimated_row_count") or 0)
            for table in discovery["tables"].values()
        )
        total_discovered_columns = sum(
            len(table.get("columns") or [])
            for table in discovery["tables"].values()
        )
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Discovered tables", len(discovery["tables"]))
        d2.metric("Discovered rows", f"{total_discovered_rows:,}")
        d3.metric("Discovered columns", f"{total_discovered_columns:,}")
        d4.metric("Snapshot", "recorded")
        st.dataframe(overview_rows, hide_index=True, width="stretch")
        with st.expander("Column-level schema details", expanded=False):
            detail_rows = []
            for table_name, table in discovery["tables"].items():
                pk_set = set(table.get("primary_key") or [])
                fk_by_col = {
                    fk.get("column"): f"{fk.get('references_table')}.{fk.get('references_column')}"
                    for fk in (table.get("foreign_keys") or [])
                }
                columns = table.get("columns") or []
                if isinstance(columns, dict):
                    column_items = list(columns.items())
                else:
                    column_items = []
                    for col in columns:
                        if isinstance(col, dict):
                            column_items.append((col.get("name") or col.get("column") or "?", col))
                        else:
                            column_items.append((str(col), {"name": str(col)}))
                for col_name, col in column_items:
                    if isinstance(col, dict):
                        physical = (
                            col.get("database_type")
                            or col.get("physical_type")
                            or col.get("type")
                            or "-"
                        )
                    else:
                        physical = str(col)
                    role = "PK" if col_name in pk_set else ("FK" if col_name in fk_by_col else "-")
                    detail_rows.append(
                        {
                            "table": table_name,
                            "column": col_name,
                            "type": physical,
                            "role": role,
                            "references": fk_by_col.get(col_name, "-"),
                        }
                    )
            st.dataframe(detail_rows, hide_index=True, width="stretch")
        free_text_preview = []
        # Prefer contract when available; discovery alone has no semantics yet.
        if st.session_state.db_contract:
            free_text_preview = list_contract_free_text_columns(st.session_state.db_contract)
        if free_text_preview:
            st.caption(
                f"Detected {len(free_text_preview)} text-heavy column(s). "
                "Optional LLM rewriting is offered at Generate."
            )
        else:
            st.caption(
                "Text-heavy fields use privacy-safe synthetic generators (not raw source text)."
            )

if st.session_state.db_discovery is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 3 - Profile
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "profile")
    step_header(4, step.get("title", "Understand values"), st.session_state.db_profile is not None)
    step_guide(
        what=step.get("what", "Learn safe shape statistics (missingness, cardinality, distributions)."),
        next_step=step.get("next", "Approve how each column should be treated."),
    )
    if st.session_state.db_profile is None:
        with st.expander("Advanced: large tables"):
            chunked = st.checkbox(
                "Process large tables in chunks",
                help="Streams each table instead of loading the full sample at once. "
                "Recommended only for tables too large to fit in memory as a single sample.",
            )
            st.session_state.db_chunk_size = (
                st.number_input(
                    "Chunk size (rows)",
                    min_value=SETTINGS.db_min_chunk_size,
                    value=SETTINGS.db_default_chunk_size,
                    step=SETTINGS.db_chunk_step,
                )
                if chunked
                else None
            )
        if st.button("Understand values", icon=":material/query_stats:"):
            manifest = _manifest_for_new_outputs("profile.json")
            result = run_profiling(
                source_adapter, st.session_state.db_discovery, manifest, "discovery.json",
                sample_limit=SETTINGS.db_profile_sample_limit,
                chunk_size=st.session_state.db_chunk_size,
            )
            if not result.is_success():
                show_user_error(
                    "We couldn't analyze values in your database.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_profile = load_profile(manifest.output_path(PROFILE_FILENAME))
            st.rerun()
    else:
        _render_profile_summary(st.session_state.db_profile)

if st.session_state.db_profile is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 4 - Review semantics
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "understand")
    step_header(5, step.get("title", "Understand data"), st.session_state.db_contract is not None)
    step_guide(
        what=step.get("what", "Review column meanings, then approve the understanding used for training."),
        next_step=step.get("next", "Train the twin."),
    )
    if st.session_state.db_candidates is None:
        if st.button("Understand data", icon=":material/psychology:"):
            result = run_inference(
                source_adapter, st.session_state.db_discovery, st.session_state.db_profile, manifest,
                "discovery.json", "profile.json", sample_limit=SETTINGS.db_inference_sample_limit,
                chunk_size=st.session_state.db_chunk_size,
            )
            if not result.is_success():
                show_user_error(
                    "We couldn't understand your data semantics.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_candidates = json.loads(Path(result.output_references[0]).read_text())
            st.rerun()

    if st.session_state.db_candidates is not None and st.session_state.db_contract is None:
        candidates = st.session_state.db_candidates
        decisions: dict[str, dict[str, str]] = {}
        for table_name, cols in candidates["tables"].items():
            st.markdown(f"**{table_name}**")
            rows = []
            for col_name, cand in cols.items():
                rows.append(
                    {
                        "column": col_name,
                        "proposed semantic type": cand["semantic_type"],
                        "confidence": cand["confidence"],
                        "status": cand["status"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            decisions[table_name] = {col_name: cand["semantic_type"] for col_name, cand in cols.items()}

        if st.button("Approve understanding", icon=":material/task_alt:"):
            run_contract_approval(
                dataset_id="database_a", source_fingerprint=st.session_state.db_discovery["source_fingerprint"],
                discovery_data=st.session_state.db_discovery, candidates_by_table=candidates["tables"],
                manifest=manifest, metadata_dir=metadata_dir, candidates_reference="semantic_candidates.json",
                decisions=decisions,
            )
            st.session_state.db_contract = load_dataset_contract(metadata_dir)
            st.rerun()
    elif st.session_state.db_contract is not None:
        show_success("Data understanding approved - ready to train the twin.")

if st.session_state.db_contract is None:
    st.stop()

contract = st.session_state.db_contract

# ---------------------------------------------------------------------------
# Step 5 - Learning plan / Step 6 - Train
# ---------------------------------------------------------------------------
_bounded_columns: dict[str, list[str]] = {}
for table_name, table_contract in contract["tables"].items():
    for col_name, col in table_contract["columns"].items():
        if col["inference_status"] == "approved" and col["semantic_type"] in ("numerical", "datetime"):
            _bounded_columns.setdefault(table_name, []).append(col_name)

with st.container(border=True):
    step = ui_step("database", "train")
    step_header(6, step.get("title", "Train twin"), st.session_state.db_training_report is not None)
    step_guide(
        what=step.get("what", "Train or retrain a portable ML twin from the approved database understanding."),
        next_step=step.get("next", "Download the trained twin artifact."),
    )

    if st.session_state.db_training_report is None:
        _labels_by_model_type = {
            "safe_gaussian_copula": MODEL_TYPE_LABELS["safe_gaussian_copula"],
            "dp_gaussian_copula": MODEL_TYPE_LABELS["dp_gaussian_copula"],
        }
        _labels = list(_labels_by_model_type.values())
        recommendation = recommend_synthesizer(contract)
        st.info(
            f"Recommended: **{_labels_by_model_type[recommendation['recommended_model_type']]}** "
            f"- {recommendation['reasons'][0]}",
            icon=":material/lightbulb:",
        )
        model_label = st.selectbox(
            "Twin type",
            _labels,
            index=_labels.index(_labels_by_model_type[recommendation["recommended_model_type"]]),
            help="Both produce a portable trained twin. Recommendation is advisory.",
        )
        st.session_state.db_model_type = (
            "dp_gaussian_copula" if "differential privacy" in model_label else "safe_gaussian_copula"
        )
        with st.expander("Technical details - twin type", expanded=False):
            st.caption(
                "Models are pickle-free JSON artifacts (Gaussian-copula family). "
                "Differential privacy adds calibrated noise when selected."
            )

        model_kwargs: dict = {"category_minimum_support": 5}
        if st.session_state.db_model_type == "dp_gaussian_copula":
            st.session_state.db_dp_epsilon = st.slider(
                "Privacy budget (epsilon) - smaller = stronger privacy, more noise", 0.1, 10.0,
                st.session_state.db_dp_epsilon,
            )
            st.caption("Numeric/datetime column bounds - public domain knowledge, never derived from the data itself.")
            bounds: dict[str, tuple] = {}
            for table_name, cols in _bounded_columns.items():
                for col_name in cols:
                    is_datetime = contract["tables"][table_name]["columns"][col_name]["semantic_type"] == "datetime"
                    c1, c2 = st.columns(2)
                    if is_datetime:
                        lower = c1.date_input(f"{table_name}.{col_name} - earliest", key=f"lb_{table_name}_{col_name}")
                        upper = c2.date_input(f"{table_name}.{col_name} - latest", key=f"ub_{table_name}_{col_name}")
                        bounds[col_name] = (str(lower), str(upper))
                    else:
                        lower = c1.number_input(f"{table_name}.{col_name} - lower bound", value=0.0, key=f"lb_{table_name}_{col_name}")
                        upper = c2.number_input(
                            f"{table_name}.{col_name} - upper bound",
                            value=SETTINGS.db_default_numeric_upper_bound,
                            key=f"ub_{table_name}_{col_name}",
                        )
                        bounds[col_name] = (lower, upper)
            model_kwargs["epsilon_budget"] = st.session_state.db_dp_epsilon
            model_kwargs["column_bounds"] = bounds

        st.markdown("**ML model training plan**")
        st.dataframe(
            pd.DataFrame(
                _training_plan_rows(
                    contract,
                    st.session_state.db_model_type,
                    SETTINGS.db_training_sample_limit,
                    SETTINGS.db_training_preview_rows,
                )
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Training reads bounded samples from the source, cleans them, applies pre-fit privacy masking, "
            "fits one portable model per table, then writes a validation sample from the trained model."
        )

        if st.button("Train", icon=":material/model_training:"):
            manifest = _manifest_for_new_outputs("training_report.json")
            with st.status("Training ML models from the approved database understanding", expanded=True) as status:
                def _show_training_progress(event: dict) -> None:
                    status.write(
                        f"[{event.get('table_index')}/{event.get('table_count')}] "
                        f"{event.get('table')}: {event.get('message')}"
                    )

                result = run_training_and_sampling(
                    source_adapter, contract, manifest, "dataset_contract.json",
                    sample_limit=SETTINGS.db_training_sample_limit,
                    num_rows_to_generate=SETTINGS.db_training_preview_rows,
                    seed=SETTINGS.seed,
                    model_type=st.session_state.db_model_type,
                    model_kwargs=model_kwargs,
                    progress_callback=_show_training_progress,
                )
                if result.is_success():
                    status.update(label="ML model training completed", state="complete")
                else:
                    status.update(label="ML model training failed", state="error")
            if not result.is_success():
                show_user_error(
                    "Training the twin failed.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_training_report = load_training_report(Path(result.output_references[0]))
            st.rerun()
    else:
        _render_training_report(st.session_state.db_training_report, contract)
        st.caption(
            "This is a trained model snapshot. Retraining reads the current SSOT again, fits new "
            "table models, and clears generated/downloaded outputs that depended on the previous model."
        )
        if st.button("Retrain ML twin", icon=":material/model_training:"):
            _reset_downstream("db_contract")
            st.rerun()

if st.session_state.db_training_report is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 6 - Export trained twin
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "download_twin")
    step_header(7, step.get("title", "Download trained twin"), st.session_state.db_artifact_path is not None)
    step_guide(
        what=step.get("what", "Package the trained twin as a portable artifact you can move without the source."),
        next_step=step.get("next", "Disconnect the source database."),
    )
    if st.session_state.db_artifact_path is None:
        if st.button("Download trained twin", icon=":material/archive:"):
            artifact_path = _active_manifest().run_dir / "artifacts" / "generator-database_a-1.0.0.zip"
            manifest = _manifest_for_new_outputs(extra_paths=[artifact_path])
            result = run_artifact_export(
                "database_a", "1.0.0", st.session_state.db_discovery, contract, st.session_state.db_profile,
                st.session_state.db_training_report, manifest.run_id, manifest.code_version, manifest,
                "training_report.json",
            )
            if not result.is_success():
                show_user_error(
                    "We couldn't export the trained twin.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_artifact_path = Path(result.output_references[0])
            st.rerun()
    else:
        show_success(
            f"Trained twin ready: `{st.session_state.db_artifact_path.name}` "
            f"({st.session_state.db_artifact_path.stat().st_size:,} bytes)"
        )

if st.session_state.db_artifact_path is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 7 - Disconnect source
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "disconnect")
    step_header(8, step.get("title", "Disconnect source"), st.session_state.db_source_disconnected)
    step_guide(
        what=step.get("what", "Stop using the source so generation runs from the trained twin only."),
        next_step=step.get("next", "Generate synthetic data from the artifact."),
    )
    if not st.session_state.db_source_disconnected:
        st.caption("Disconnect before generating so the twin runs without source access.")
        if st.button("Disconnect source", icon=":material/link_off:"):
            st.session_state.db_source_disconnected = True
            st.rerun()
    else:
        c1, c2 = st.columns(2)
        c1.metric("Source", "DISCONNECTED")
        c2.metric("Trained twin", "LOADED" if st.session_state.db_loaded_artifact else "ready")

if not st.session_state.db_source_disconnected:
    st.stop()

# ---------------------------------------------------------------------------
# Step 8 - Load generator (source-free from here on)
# ---------------------------------------------------------------------------
if st.session_state.db_loaded_artifact is None:
    st.session_state.db_loaded_artifact = load_artifact(st.session_state.db_artifact_path)
loaded_artifact = st.session_state.db_loaded_artifact

adapters_by_table = {}
for table_name in loaded_artifact.manifest["tables"]:
    adapter_cls = get_synthesizer_adapter_class(loaded_artifact.manifest["tables"][table_name]["model_type"])
    # Trusted by construction: this artifact was just built earlier in this
    # same local session, not downloaded from elsewhere.
    adapters_by_table[table_name] = loaded_artifact.get_model(table_name, allow_cloudpickle_models=True)

# ---------------------------------------------------------------------------
# Step 9 - Generate (row counts + distribution tweaking)
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "generate")
    step_header(9, step.get("title", "Generate"), st.session_state.db_relational_report is not None)
    step_guide(
        what=step.get("what", "Create synthetic tables from the portable artifact. Tune category mixes if needed."),
        next_step=step.get("next", "Validate integrity and quality."),
    )

    artifact_only_banner(
        disconnected=st.session_state.db_source_disconnected,
        has_artifact=bool(st.session_state.db_artifact_path or st.session_state.db_loaded_artifact),
    )

    source_preview_tables = (st.session_state.db_source_preview or {}).get("tables") or {}
    table_names = list(loaded_artifact.manifest["tables"])
    initial_recommendation, recommendation_basis = _recommended_row_counts(
        table_names,
        source_preview_tables,
        contract,
    )
    if not st.session_state.db_row_counts_initialized:
        _apply_row_count_defaults(initial_recommendation)

    current_total = sum(
        int(st.session_state.db_row_counts.get(table_name) or initial_recommendation.get(table_name) or 1)
        for table_name in table_names
    )
    if st.session_state.db_target_total_rows is None:
        st.session_state.db_target_total_rows = max(len(table_names), int(current_total))
    c_total, c_basis, c_reset = st.columns([2, 2, 1])
    target_total = c_total.number_input(
        "Target total synthetic rows",
        min_value=len(table_names),
        step=100,
        help="Scales the recommended table mix. You can still edit every table below.",
        key="db_target_total_rows",
    )
    c_basis.metric("Recommendation basis", recommendation_basis)
    if c_reset.button("Reset counts", icon=":material/restart_alt:", help="Rebalance table counts from the SSOT shape."):
        recommended, _basis = _recommended_row_counts(
            table_names,
            source_preview_tables,
            contract,
            target_total=int(target_total),
        )
        _apply_row_count_defaults(recommended)
        st.rerun()

    recommended_counts, _basis = _recommended_row_counts(
        table_names,
        source_preview_tables,
        contract,
        target_total=int(target_total),
    )
    st.caption(
        "Counts start from the SSOT table-volume shape, then scale to your requested total. "
        "Source size is reference only; generation uses the editable counts below."
    )
    overview = []
    for table_name in table_names:
        source_rows = source_preview_tables.get(table_name, {}).get("row_count")
        overview.append(
            {
                "table": table_name,
                "source rows": int(source_rows) if source_rows is not None else "-",
                "recommended": int(recommended_counts.get(table_name, 1)),
                "requested": int(st.session_state.db_row_counts.get(table_name) or recommended_counts.get(table_name, 1)),
            }
        )
    st.dataframe(pd.DataFrame(overview), hide_index=True, width="stretch")

    row_counts = {}
    for table_name in table_names:
        default = int(st.session_state.db_row_counts.get(table_name) or recommended_counts.get(table_name, 1))
        st.session_state.setdefault(f"rowcount_{table_name}", default)
        source_rows = source_preview_tables.get(table_name, {}).get("row_count")
        help_text = (
            f"Source has {int(source_rows):,} rows. This is reference only; requested output size is independent."
            if source_rows is not None
            else f"Recommended from {recommendation_basis}; requested output size is independent of source size."
        )
        row_counts[table_name] = st.number_input(
            f"Rows to generate - {table_name}",
            min_value=1,
            step=50,
            key=f"rowcount_{table_name}",
            help=help_text,
        )
    st.session_state.db_row_counts = row_counts
    if source_preview_tables:
        st.caption("Source row counts are shown as reference only. Generation uses the requested counts above.")
    ratio_warnings = relationship_volume_warnings(contract, row_counts, source_preview_tables)
    for warning in ratio_warnings:
        st.warning(warning)

    estimated_total_rows = sum(int(value) for value in row_counts.values())
    if estimated_total_rows >= SETTINGS.db_large_generation_warning_rows:
        st.warning(
            f"Large generation request: {estimated_total_rows:,} rows. "
            "This will use streamed generation, take longer, and write larger CSV artifacts."
        )

    st.markdown("**Tweak a category's distribution (no retraining)**")
    category_overrides: dict[str, dict[str, dict[str, float]]] = {}
    override_errors: list[str] = []
    for table_name, adapter in adapters_by_table.items():
        for col_name, col in contract["tables"][table_name]["columns"].items():
            if col["semantic_type"] != "category" or not hasattr(adapter, "get_category_distribution"):
                continue
            try:
                learned = adapter.get_category_distribution(col_name)
            except Exception:
                continue
            if not learned:
                continue

            with st.expander(f"{table_name}.{col_name} - learned: " + ", ".join(f"{k} {v:.0%}" for k, v in learned.items())):
                chart_df = pd.DataFrame({"category": list(learned), "proportion": list(learned.values())})
                st.bar_chart(chart_df, x="category", y="proportion", horizontal=True)

                enable = st.checkbox("Override this distribution", key=f"override_enable_{table_name}_{col_name}")
                if enable:
                    if st.button("Reset to learned distribution", key=f"override_reset_{table_name}_{col_name}"):
                        for category, learned_value in learned.items():
                            st.session_state[f"override_pct_{table_name}_{col_name}_{category}"] = round(
                                float(learned_value) * 100,
                                2,
                            )
                        st.rerun()
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "category": list(learned),
                                "learned %": [round(float(value) * 100, 2) for value in learned.values()],
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                    targets = {}
                    for category in learned:
                        key = f"override_pct_{table_name}_{col_name}_{category}"
                        st.session_state.setdefault(key, round(float(learned[category]) * 100, 2))
                        targets[category] = st.number_input(
                            f"{category} target %",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.01,
                            key=key,
                        )
                    total_pct = round(sum(float(value) for value in targets.values()), 2)
                    delta = round(total_pct - 100.0, 2)
                    if abs(delta) > 0.01:
                        direction = "over" if delta > 0 else "under"
                        message = (
                            f"{table_name}.{col_name} target percentages total {total_pct:.2f}%; "
                            f"{direction}-allocated by {abs(delta):.2f}%."
                        )
                        override_errors.append(message)
                        st.error(message)
                    else:
                        requested = {k: float(v) / 100.0 for k, v in targets.items()}
                        st.caption("Requested target: " + ", ".join(f"{k} {v:.2f}%" for k, v in targets.items()))
                        category_overrides.setdefault(table_name, {})[col_name] = requested

    st.session_state.db_category_overrides = category_overrides

    free_text_cols = list_contract_free_text_columns(contract)
    st.markdown("**Eligible narrative or free-text fields (optional LLM)**")
    if free_text_cols:
        st.dataframe(free_text_cols, hide_index=True, width="stretch")
        st.session_state.db_llm_text = st.toggle(
            "Use LLM for eligible narrative or free-text fields only",
            value=bool(st.session_state.db_llm_text),
            help="After relational generation, rewrites free_text columns with the shared "
            "text engine (LLM when configured, otherwise local templates). Never uses source text.",
        )
    else:
        st.session_state.db_llm_text = False
        st.caption("No eligible narrative or free-text fields in this contract; LLM option is hidden.")

    if st.button("Generate synthetic database", icon=":material/auto_fix_high:", type="primary"):
        if override_errors:
            show_user_error(
                "Category percentage targets must total exactly 100%.",
                technical="\n".join(override_errors),
                next_action="Edit the percentage fields. The system will not guess which category is wrong.",
            )
            st.stop()
        max_rows = max(int(v) for v in row_counts.values()) if row_counts else 0
        # Streaming activates for large runs; small/E2E paths keep in-memory default.
        gen_batch_size = (
            SETTINGS.db_default_chunk_size
            if max_rows >= SETTINGS.db_large_generation_warning_rows
            else None
        )
        manifest = _manifest_for_new_outputs("relational_generation_report.json")
        with measure_generation() as run_stats:
            result = run_relational_generation(
                contract, adapters_by_table, row_counts, manifest, contract_reference="dataset_contract.json",
                seed=SETTINGS.db_generation_seed, category_overrides_by_table=category_overrides or None,
                batch_size=gen_batch_size,
            )
            if result.is_success() and st.session_state.db_llm_text and free_text_cols:
                report_path = manifest.output_path("relational_generation_report.json")
                interim = load_relational_generation_report(report_path)
                st.session_state.db_llm_evidence = apply_llm_free_text_to_tables(
                    interim,
                    free_text_cols,
                    seed=SETTINGS.db_generation_seed,
                    max_llm_rows=SETTINGS.db_llm_max_rows,
                )
            else:
                st.session_state.db_llm_evidence = None
        if not result.is_success():
            show_user_error(
                "Synthetic data generation failed.",
                technical=result.errors,
            )
            st.stop()
        st.session_state.db_run_metrics = dict(run_stats)
        st.session_state.db_relational_report = load_relational_generation_report(
            manifest.output_path("relational_generation_report.json")
        )
        st.rerun()

    if st.session_state.db_relational_report is not None:
        report = st.session_state.db_relational_report
        total_rows = sum(int(entry.get("row_count") or 0) for entry in report["tables"].values())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tables", len(report["tables"]))
        m2.metric("Total rows", f"{total_rows:,}")
        m3.metric("FK validity", f"{report['fk_validity']['overall_fk_validity']:.0%}")
        if report.get("streaming"):
            m4.metric("Generation", f"Streaming ({report.get('batch_size'):,}/batch)")
        else:
            m4.metric("Generation", "In memory")
        st.dataframe(
            [
                {
                    "table": name,
                    "requested rows": int(row_counts.get(name, 0)),
                    "actual rows": int(entry.get("row_count") or 0),
                }
                for name, entry in report["tables"].items()
            ],
            hide_index=True,
            width="stretch",
        )

        metrics = st.session_state.db_run_metrics or {}
        rps = (total_rows / metrics["seconds"]) if metrics.get("seconds") else None
        render_run_metrics(
            seconds=metrics.get("seconds"),
            peak_memory_mb=metrics.get("peak_memory_mb"),
            rows_per_second=rps,
            extra={"scope": "database twin generation"},
        )
        if st.session_state.db_llm_evidence:
            with st.expander("LLM text generation evidence", expanded=False):
                st.dataframe(st.session_state.db_llm_evidence, hide_index=True, width="stretch")

        table_name = st.selectbox("Preview table", list(report["tables"]), key="generated_preview_select")
        # Preview only - never reload the full frame into session_state.
        generated_df = pd.read_csv(report["tables"][table_name]["path"], nrows=50)
        st.dataframe(generated_df, width="stretch")
        st.caption(
            f"`{table_name}`: {int(report['tables'][table_name].get('row_count') or 0):,} rows "
            "(showing first 50)"
        )

        drift_rows = profile_vs_synthetic_drift_rows(
            st.session_state.db_profile,
            table_name,
            pd.read_csv(report["tables"][table_name]["path"], nrows=SETTINGS.db_drift_preview_rows),
            limit=12,
        )
        render_data_drift_panel(mode="database", rows=drift_rows or None)

        if "gender" in generated_df.columns:
            # Gender split needs a fuller read of that one column only.
            gender_series = pd.read_csv(
                report["tables"][table_name]["path"], usecols=["gender"]
            )["gender"]
            actual = gender_series.value_counts(normalize=True)
            st.caption("Resulting gender split in this generated table: " + ", ".join(f"{k} {v:.0%}" for k, v in actual.items()))

        comparison_rows = []
        for override_table, cols in (st.session_state.db_category_overrides or {}).items():
            for override_col, requested in cols.items():
                entry = report["tables"].get(override_table)
                if not entry:
                    continue
                try:
                    generated_series = pd.read_csv(entry["path"], usecols=[override_col])[override_col]
                    generated = generated_series.value_counts(normalize=True).to_dict()
                    learned = adapters_by_table[override_table].get_category_distribution(override_col)
                except Exception:
                    continue
                for category in sorted(set(learned) | set(requested) | set(generated)):
                    comparison_rows.append(
                        {
                            "table": override_table,
                            "column": override_col,
                            "category": category,
                            "learned %": round(float(learned.get(category, 0.0)) * 100, 2),
                            "requested %": round(float(requested.get(category, 0.0)) * 100, 2),
                            "generated %": round(float(generated.get(category, 0.0)) * 100, 2),
                        }
                    )
        if comparison_rows:
            with st.expander("Category distribution result", expanded=False):
                st.dataframe(comparison_rows, hide_index=True, width="stretch")

if st.session_state.db_relational_report is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 10 - Validate
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "validate")
    step_header(10, step.get("title", "Validate"), st.session_state.db_qa_report is not None)
    step_guide(
        what=step.get("what", "Confirm relational integrity and available quality signals."),
        next_step=step.get("next", "Export the synthetic database."),
    )

    if st.session_state.db_qa_report is None:
        if st.button("Validate results", icon=":material/fact_check:"):
            manifest = _manifest_for_new_outputs("qa_report.json")
            result = run_qa_validation(
                contract, st.session_state.db_relational_report, manifest,
                relational_report_reference="relational_generation_report.json",
                reference_profile=st.session_state.db_profile,
            )
            if not result.is_success():
                show_user_error(
                    "Validation failed.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_qa_report = load_qa_report(manifest.output_path("qa_report.json"))
            st.rerun()
    else:
        qa = st.session_state.db_qa_report
        if qa["hard_checks_passed"]:
            validation_badge(True, passed_label="Validation passed")
        else:
            validation_badge(False)
        integrity = qa.get("report", {}).get("integrity", {})
        fk = integrity.get("fk_validity", {}).get("overall_fk_validity")
        pk_checks = integrity.get("primary_key_checks") or {}
        pk_ok = all(
            check.get("is_unique", True) for check in pk_checks.values() if check.get("checked")
        ) if pk_checks else None
        render_integrity_metrics(
            hard_checks_passed=bool(qa["hard_checks_passed"]),
            fk_label=f"{float(fk):.0%}" if fk is not None else "-",
            pk_label=("unique" if pk_ok else "duplicates") if pk_ok is not None else "-",
        )
        fidelity_rows = fidelity_drift_rows(qa, limit=12)
        render_data_drift_panel(
            mode="database",
            rows=fidelity_rows or None,
            footnote=(
                "Scores come from QA fidelity checks against privacy-safe profile bounds / vocabularies."
                if fidelity_rows
                else "No fidelity columns available yet - integrity checks above are still authoritative."
            ),
        )

if st.session_state.db_qa_report is None or not st.session_state.db_qa_report["hard_checks_passed"]:
    st.stop()

# ---------------------------------------------------------------------------
# Step 11 - Export to target database
# ---------------------------------------------------------------------------
with st.container(border=True):
    step = ui_step("database", "export")
    step_header(11, step.get("title", "Export"), st.session_state.db_write_report is not None)
    step_guide(
        what=step.get("what", "Write the synthetic database and download it."),
        next_step=step.get("next", "Done - you have a validated synthetic database."),
    )

    if st.session_state.db_write_report is None:
        if st.button("Export to target database", icon=":material/database_upload:", type="primary"):
            target_path = st.session_state.db_workdir / "database_b.db"
            manifest = _manifest_for_new_outputs("target_write_report.json")
            result = run_target_write(
                target_path, contract, st.session_state.db_relational_report, st.session_state.db_qa_report, manifest,
                qa_report_reference="qa_report.json",
            )
            if not result.is_success():
                show_user_error(
                    "Export to the target database failed.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_target_path = target_path
            st.session_state.db_write_report = load_target_write_report(
                manifest.output_path("target_write_report.json")
            )
            st.rerun()

    if st.session_state.db_write_report is not None:
        write_report = st.session_state.db_write_report
        show_success(f"Synthetic database exported to `{st.session_state.db_target_path.name}`.")
        confirmed = write_report["validation_report"]["all_writes_confirmed"]
        st.metric("Writes confirmed", "yes" if confirmed else "no")

        conn = sqlite3.connect(str(st.session_state.db_target_path))
        for table_name in st.session_state.db_relational_report["tables"]:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            st.caption(f"`{table_name}`: {count} rows")
        conn.close()

        with open(st.session_state.db_target_path, "rb") as f:
            st.download_button("Download Database B", f, file_name="database_b.db", icon=":material/download:")
        mode_outcomes(
            [
                "Portable trained twin artifact",
                "Synthetic relational database",
                "Validation / QA report",
                "Source-free generation after disconnect (this session)",
            ]
        )
        st.caption(f"Intent: {st.session_state.db_intent}. Privacy safeguards applied during training.")
