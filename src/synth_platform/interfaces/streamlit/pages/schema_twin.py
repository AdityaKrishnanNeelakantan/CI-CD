"""Schema Twin — Streamlit presentation layer for Schema Mode.

UI-only: collect intent/schema/settings, call the Schema Twin workflow facade,
and present preview / validation / download. No generation algorithms here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from synth_platform.application.dto.tool_commands import (
    CompleteSessionCommand,
    GetResultCommand,
    ReadPreferencesCommand,
    StartSessionCommand,
)
from synth_platform.application.dto.workspace import ProgressStatus, WorkflowKind
from synth_platform.application.services.result_presentation import schema_result_view
from synth_platform.application.workflows.schema_twin import (
    generate_from_schema,
    list_llm_text_columns,
    load_schema,
    load_schema_bytes,
    package_download,
    schema_column_details,
    summarize_schema,
    validation_highlights,
)
from synth_platform.bootstrap import build_guarded_chat_model
from synth_platform.interfaces.streamlit.components.common.ux import (
    measure_generation,
    mode_outcomes,
    platform_intro,
    render_data_drift_panel,
    render_integrity_metrics,
    render_pk_fk_legend,
    render_progress,
    render_run_metrics,
    schema_integrity_signal_rows,
    show_user_error,
    step_guide,
    step_header,
)
from synth_platform.interfaces.streamlit.workspace import (
    ensure_workflow_session,
    fingerprint_configuration,
    get_workspace_tools,
    record_progress_once,
    render_backend_progress,
    render_project_save,
    render_result_summary,
    workflow_run_dir,
)

st.title("Schema Mode")
platform_intro()
st.caption(
    "You have the structure of the data you need — generate synthetic relational data "
    "without production records."
)

defaults = {
    "schema_intent": "Development & testing",
    "schema_config": None,
    "schema_summary": None,
    "schema_file_id": None,
    "schema_workdir": None,
    "schema_result": None,
    "schema_zip_bytes": None,
    "schema_run_metrics": None,
    "schema_llm_text": False,
    "schema_workspace_session_id": None,
    "schema_configuration_hash": None,
    "schema_result_view_id": None,
    "schema_template_definition": None,
    "schema_template_id": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)
workspace_tools = get_workspace_tools()
preferences = workspace_tools.get_preferences(ReadPreferencesCommand())


def _progress_steps() -> list[tuple[str, bool]]:
    return [
        ("Intent", True),
        ("Provide schema", st.session_state.schema_config is not None),
        ("Review", st.session_state.schema_config is not None),
        ("Configure", st.session_state.schema_config is not None),
        ("Generate", st.session_state.schema_result is not None),
        ("Preview", st.session_state.schema_result is not None),
        ("Validate", st.session_state.schema_result is not None),
        ("Download", st.session_state.schema_result is not None),
    ]


render_progress(_progress_steps())


def _reset_generation() -> None:
    st.session_state.schema_result = None
    st.session_state.schema_zip_bytes = None
    st.session_state.schema_run_metrics = None
    st.session_state.schema_result_view_id = None


# ---------------------------------------------------------------------------
# 1. Intent
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(1, "Intent", True)
    step_guide(
        what="Tell us how you plan to use the output (informational — does not change generation).",
        next_step="Upload a schema file.",
    )
    options = [
        "Development & testing",
        "QA / automated tests",
        "Demonstrations",
        "Data pipeline development",
        "Early project environments",
    ]
    st.session_state.schema_intent = st.selectbox(
        "What will you use the synthetic data for?",
        options=options,
        index=options.index(st.session_state.schema_intent)
        if st.session_state.schema_intent in options
        else 0,
        help="Informational only — does not change generation behavior.",
    )
    st.info("Schema Mode creates synthetic datasets from metadata alone — no production records required.")


# ---------------------------------------------------------------------------
# 2. Provide schema
# ---------------------------------------------------------------------------
if st.session_state.schema_config is None and st.session_state.schema_template_definition:
    template_payload = st.session_state.schema_template_definition
    schema = load_schema(template_payload)
    template_bytes = json.dumps(template_payload, sort_keys=True).encode("utf-8")
    source_hash = hashlib.sha256(template_bytes).hexdigest()
    st.session_state.schema_config = schema
    st.session_state.schema_summary = summarize_schema(schema)
    st.session_state.schema_file_id = source_hash

with st.container(border=True):
    step_header(2, "Provide schema", st.session_state.schema_config is not None)
    step_guide(
        what="Upload SQL DDL, JSON, or YAML that describes tables, columns, and relationships.",
        next_step="Review the detected structure.",
    )
    uploaded = st.file_uploader(
        "Schema file",
        type=["json", "yaml", "yml", "sql"],
        help="JSON, YAML, or SQL DDL.",
    )
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        file_id = hashlib.sha256(uploaded_bytes).hexdigest()
        if file_id != st.session_state.schema_file_id:
            try:
                schema = load_schema_bytes(uploaded_bytes, uploaded.name)
                st.session_state.schema_config = schema
                st.session_state.schema_summary = summarize_schema(schema)
                st.session_state.schema_file_id = file_id
                st.session_state.schema_template_definition = None
                st.session_state.schema_template_id = None
                _reset_generation()
            except Exception as exc:
                st.session_state.schema_config = None
                st.session_state.schema_summary = None
                st.session_state.schema_file_id = None
                _reset_generation()
                show_user_error(
                    "We couldn't load your schema file. Check the format and try again.",
                    technical=exc,
                    next_action="Use a .json, .yaml/.yml, or .sql schema, then re-upload.",
                )
                st.stop()
        st.caption(f"Loaded schema from `{uploaded.name}`.")

if st.session_state.schema_config is None:
    st.warning("Upload a schema to continue.")
    with st.expander("What is Schema Mode?"):
        st.write(
            "Schema Mode generates synthetic relational data from a schema definition. "
            "Useful for development, QA, demos, and pipeline work when you should not use production records."
        )
    st.stop()

schema = st.session_state.schema_config
summary = st.session_state.schema_summary or summarize_schema(schema)
st.session_state.schema_summary = summary

# ---------------------------------------------------------------------------
# 3. Review schema
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(3, "Review schema", True, help_text="Confirm tables, keys, and relationships before generating.")
    step_guide(what="This is the structure the twin will follow.", next_step="Configure row count and seed.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tables", summary.table_count)
    c2.metric("Columns", summary.column_count)
    c3.metric("Relationships", summary.relationship_count)
    c4.metric("Source records required", "No")
    st.caption(f"Dataset: **{summary.name}**")
    render_pk_fk_legend()
    if summary.tables:
        st.dataframe(summary.tables, width="stretch", hide_index=True)
    with st.expander("Schema details (columns, PK, FK)", expanded=False):
        details = schema_column_details(schema)
        if details:
            st.dataframe(details, width="stretch", hide_index=True)
        else:
            st.caption("No columns found in this schema.")
        if schema.relationships:
            st.markdown("**Relationships**")
            rel_rows = [
                {
                    "from": f"{r.child_table}.{r.child_key}",
                    "to": f"{r.parent_table}.{r.parent_key}",
                }
                for r in schema.relationships
            ]
            st.dataframe(rel_rows, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 4. Configure generation
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(4, "Configure generation", True)
    step_guide(what="Choose how much data to create and how to reproduce it.", next_step="Generate.")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        row_count = st.number_input(
            "Rows per table",
            min_value=1,
            max_value=1_000_000,
            value=preferences.default_row_count,
            step=10,
            help="Applied to each table in the schema.",
        )
    with cfg2:
        locale = st.selectbox(
            "Locale",
            options=["en_US", "en_GB", "en_IN", "de_DE", "fr_FR"],
            index=(
                ["en_US", "en_GB", "en_IN", "de_DE", "fr_FR"].index(
                    preferences.locale
                )
                if preferences.locale
                in ["en_US", "en_GB", "en_IN", "de_DE", "fr_FR"]
                else 0
            ),
            help="Locale-aware synthetic values where supported by the generator.",
        )
    with cfg3:
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            help="Same seed reproduces the same synthetic output.",
        )
    export_format = st.selectbox(
        "Export format",
        options=["csv", "parquet"],
        index=["csv", "parquet"].index(preferences.default_export_format),
    )

    llm_cols = list_llm_text_columns(schema)
    st.markdown("**Text-heavy columns (optional LLM)**")
    st.caption(
        "Only long notes-style columns can use optional LLM text. "
        "IDs, names, emails, and similar fields stay on privacy-safe generators."
    )
    if llm_cols:
        st.dataframe(llm_cols, width="stretch", hide_index=True)
        st.session_state.schema_llm_text = st.toggle(
            "Use LLM for text-heavy columns only",
            value=bool(st.session_state.schema_llm_text),
            help="Requires a configured LLM provider. Falls back to local text generators if unavailable.",
        )
    else:
        st.session_state.schema_llm_text = False
        st.caption("No text-heavy columns detected in this schema — LLM option is hidden.")

schema_configuration_hash = fingerprint_configuration(
    {
        "row_count": int(row_count),
        "locale": locale,
        "seed": int(seed),
        "export_format": export_format,
        "llm_text_enabled": bool(st.session_state.schema_llm_text),
    }
)
if st.session_state.schema_configuration_hash not in {None, schema_configuration_hash}:
    _reset_generation()
st.session_state.schema_configuration_hash = schema_configuration_hash
schema_session = ensure_workflow_session(
    state_key="schema_workspace_session_id",
    workflow=WorkflowKind.SCHEMA,
    title="Schema Twin",
    source_fingerprint=st.session_state.schema_file_id,
    configuration_fingerprint=schema_configuration_hash,
    current_step="review",
)
st.session_state.schema_workdir = workflow_run_dir(
    schema_session.session_id, WorkflowKind.SCHEMA
)
for macro_step, stage_id, stage_label in (
    ("input", "schema_input", "Provide schema"),
    ("review", "schema_review", "Review schema"),
    ("review", "generation_configuration", "Configure generation"),
):
    record_progress_once(
        schema_session.session_id,
        macro_step=macro_step,
        stage_id=stage_id,
        stage_label=stage_label,
        status=ProgressStatus.SUCCEEDED,
    )

# ---------------------------------------------------------------------------
# 5. Generate
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(5, "Generate", st.session_state.schema_result is not None)
    step_guide(what="Create the synthetic relational dataset from the schema.", next_step="Preview and validate.")
    generate_clicked = st.button("Generate Synthetic Data", type="primary", width="stretch")
    if generate_clicked:
        workspace_tools.start_session(
            StartSessionCommand(session_id=schema_session.session_id)
        )
        record_progress_once(
            schema_session.session_id,
            macro_step="generate",
            stage_id="generation",
            stage_label="Generate synthetic data",
            status=ProgressStatus.RUNNING,
            message="Running the specialized Schema pipeline.",
        )
        with st.status("Generating your synthetic dataset...", expanded=True) as status:
            try:
                st.write("Reading schema...")
                st.write("Preparing generation configuration...")
                text_model = (
                    build_guarded_chat_model("synthetic_text")
                    if st.session_state.schema_llm_text
                    else None
                )
                st.write("Running schema-driven pipeline...")
                with measure_generation() as run_stats:
                    result = generate_from_schema(
                        schema,
                        row_count=int(row_count),
                        seed=int(seed),
                        locale=locale,
                        output_dir=Path(st.session_state.schema_workdir) / "output",
                        export_format=export_format,
                        preview_rows=min(100, int(row_count)),
                        llm_text_enabled=bool(st.session_state.schema_llm_text),
                        text_model=text_model,
                    )
                st.session_state.schema_result = result
                zip_bytes = package_download(result)
                st.session_state.schema_zip_bytes = zip_bytes
                package_path = Path(st.session_state.schema_workdir) / "schema_twin_package.zip"
                package_path.write_bytes(zip_bytes)
                perf = getattr(result.pipeline, "performance", None)
                st.session_state.schema_run_metrics = {
                    "seconds": float(getattr(perf, "total_time_seconds", None) or run_stats["seconds"]),
                    "peak_memory_mb": float(
                        getattr(perf, "peak_memory_mb", None) or run_stats["peak_memory_mb"]
                    ),
                    "rows_per_second": float(getattr(perf, "rows_per_second", 0) or 0) or None,
                }
                record_progress_once(
                    schema_session.session_id,
                    macro_step="generate",
                    stage_id="generation",
                    stage_label="Generate synthetic data",
                    status=ProgressStatus.SUCCEEDED,
                    counts={"rows": sum(int(value) for value in result.row_counts.values())},
                )
                result_view = schema_result_view(
                    workspace_tools,
                    schema_session.session_id,
                    result,
                    package_path=package_path,
                )
                workspace_tools.complete_session(
                    CompleteSessionCommand(result=result_view)
                )
                st.session_state.schema_result_view_id = result_view.result_id
                status.update(label="Synthetic dataset generated", state="complete", expanded=False)
            except Exception as exc:
                st.session_state.schema_result = None
                st.session_state.schema_zip_bytes = None
                st.session_state.schema_run_metrics = None
                st.session_state.schema_result_view_id = None
                record_progress_once(
                    schema_session.session_id,
                    macro_step="generate",
                    stage_id="generation",
                    stage_label="Generate synthetic data",
                    status=ProgressStatus.FAILED,
                    message=f"Generation failed ({type(exc).__name__}).",
                )
                status.update(label="Generation failed", state="error")
                show_user_error(
                    "The synthetic dataset could not be generated. Review the schema and settings.",
                    technical=exc,
                    next_action="Check relationships and column types, then try again.",
                )

result = st.session_state.schema_result
render_backend_progress(schema_session.session_id)
if result is None:
    st.stop()

# ---------------------------------------------------------------------------
# 6. Preview
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(6, "Preview", True)
    st.caption("Your synthetic dataset is ready to preview.")
    counts = result.row_counts or {name: len(df) for name, df in result.preview_tables.items()}
    total_rows = sum(int(v) for v in counts.values())
    perf = getattr(result.pipeline, "performance", None)
    streaming = bool(
        perf
        and (
            getattr(perf, "output_streamed", False)
            or any("streaming=True" in str(n) for n in (getattr(perf, "notes", None) or []))
        )
    )
    chunk_size = int(getattr(perf, "chunk_size", 0) or 0) if perf else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tables", len(counts))
    m2.metric("Total rows", f"{total_rows:,}")
    m3.metric("Export files", len(result.export_paths or {}))
    if streaming and chunk_size:
        m4.metric("Generation", f"Streaming ({chunk_size:,}/chunk)")
    else:
        m4.metric("Generation", "In memory")

    metrics = st.session_state.schema_run_metrics or {}
    render_run_metrics(
        seconds=metrics.get("seconds"),
        peak_memory_mb=metrics.get("peak_memory_mb"),
        rows_per_second=metrics.get("rows_per_second"),
        extra={"scope": "schema generation", "note": "Time and memory for this Schema Twin run."},
    )

    table_names = list(result.preview_tables.keys())
    if table_names:
        selected = st.selectbox("Preview table", options=table_names)
        preview_df = result.preview_tables[selected]
        st.dataframe(preview_df.head(100), width="stretch", hide_index=True)
        st.caption(
            f"`{selected}`: {int(counts.get(selected, len(preview_df))):,} rows "
            f"(showing first {min(100, len(preview_df))})"
        )

# ---------------------------------------------------------------------------
# 7. Validate
# ---------------------------------------------------------------------------
highlights = validation_highlights(result)
with st.container(border=True):
    step_header(7, "Validate", True)
    if highlights["hard_checks_passed"]:
        st.success(f"Validation status: {highlights['status']}")
    else:
        st.error(f"Validation status: {highlights['status']}")

    export_val = ((result.validation_report or {}).get("guarantees") or {}).get("export_validation") or {}
    if not export_val:
        export_val = getattr(result.pipeline, "export_validation", None) or {}
    fk_passed = export_val.get("fk_passed")
    pk_passed = export_val.get("pk_passed")
    render_integrity_metrics(
        hard_checks_passed=bool(highlights["hard_checks_passed"]),
        fk_label="n/a" if fk_passed is None else ("valid" if fk_passed else "failed"),
        pk_label="n/a" if pk_passed is None else ("unique" if pk_passed else "duplicates"),
    )
    render_data_drift_panel(
        mode="schema",
        rows=schema_integrity_signal_rows(
            hard_checks_passed=bool(highlights["hard_checks_passed"]),
            fk_passed=None if fk_passed is None else bool(fk_passed),
            pk_passed=None if pk_passed is None else bool(pk_passed),
        ),
    )
    with st.expander("Validation report"):
        st.json(result.validation_report)

# ---------------------------------------------------------------------------
# 8. Download
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(8, "Download", True)
    zip_bytes = st.session_state.schema_zip_bytes
    if zip_bytes is None:
        zip_bytes = package_download(result)
        st.session_state.schema_zip_bytes = zip_bytes
    st.download_button(
        "Download synthetic dataset (ZIP)",
        data=zip_bytes,
        file_name=f"{summary.name.replace(' ', '_').lower()}_synthetic.zip",
        mime="application/zip",
        width="stretch",
    )
    mode_outcomes(
        [
            "Synthetic relational tables (CSV or Parquet)",
            "Validation report in the package",
            "Repeatable output via the same seed",
        ]
    )
    st.caption(
        f"Intent: {st.session_state.schema_intent}. "
        "Generated from schema metadata — not from production records."
    )


result_view_id = st.session_state.schema_result_view_id
if result_view_id:
    persisted_result = workspace_tools.get_result(
        GetResultCommand(
            session_id=schema_session.session_id,
            result_id=result_view_id,
        )
    )
    if persisted_result is not None:
        render_result_summary(persisted_result)
        render_project_save(schema_session.session_id)
