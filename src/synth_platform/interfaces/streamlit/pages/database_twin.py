"""Database twin: Database A -> discover -> profile -> infer semantics ->
train -> export -> (disconnect) -> load -> generate (with distribution
tweaking) -> write to Database B -> validate.

UI-focused: every step below calls straight into the already-tested
src/ pipeline modules. No business logic lives in this file.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from synth_platform.application.dto.tool_commands import (
    CompleteSessionCommand,
    GetResultCommand,
    GetSessionCommand,
    SetConfigurationFingerprintCommand,
    StartSessionCommand,
)
from synth_platform.application.dto.workspace import (
    PreviewDescriptor,
    ProgressStatus,
    WorkflowKind,
)
from synth_platform.application.services.result_presentation import (
    stage_workflow_result_view,
)
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
)
from synth_platform.bootstrap import workspace_root_path
from synth_platform.domain.validation.models import Status
from synth_platform.infrastructure.storage.source_staging import (
    create_source_staging,
    remove_source_staging,
    touch_source_staging,
)
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
    show_success,
    show_user_error,
    step_guide,
    step_header,
    validation_badge,
)
from synth_platform.interfaces.streamlit.workspace import (
    ensure_workflow_session,
    fingerprint_configuration,
    get_workspace_tools,
    record_failure,
    record_progress_once,
    render_backend_progress,
    render_project_save,
    render_result_summary,
    workflow_run_dir,
)

st.title("Database Twin")
platform_intro()
st.caption(
    "Learn from an existing database, download a portable trained twin, disconnect the source, "
    "then generate and validate synthetic data from the artifact alone."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "db_workdir": None,
    "db_source_path": None,
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
    "db_category_overrides": {},
    "db_relational_report": None,
    "db_qa_report": None,
    "db_target_path": None,
    "db_write_report": None,
    "db_chunk_size": None,
    "db_run_metrics": None,
    "db_llm_text": False,
    "db_llm_evidence": None,
    "db_intent": "Development & testing",
    "db_source_hash": None,
    "db_workspace_session_id": None,
    "db_result_view_id": None,
    "db_draft_id": None,
    "db_source_staging_dir": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

if st.session_state.db_draft_id is None:
    st.session_state.db_draft_id = uuid4().hex
if st.session_state.db_workdir is None:
    st.session_state.db_workdir = (
        workspace_root_path() / "drafts" / "database" / st.session_state.db_draft_id
    )
    st.session_state.db_workdir.mkdir(parents=True, exist_ok=True)
if st.session_state.db_source_staging_dir is None:
    st.session_state.db_source_staging_dir = create_source_staging("database")
else:
    touch_source_staging(st.session_state.db_source_staging_dir)


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
    current_hint="Source → Understand → Train twin → Download artifact → Disconnect → Generate → Validate → Export",
)

def _reset_downstream(from_key: str) -> None:
    """Clear every session-state key that depends on `from_key`, so
    re-running an earlier step doesn't leave stale later-step results
    displayed alongside inconsistent new ones.
    """
    order = [
        "db_source_path", "db_discovery", "db_profile", "db_candidates", "db_contract",
        "db_training_report", "db_artifact_path", "db_source_disconnected", "db_loaded_artifact",
        "db_relational_report", "db_qa_report", "db_target_path", "db_write_report",
    ]
    if from_key not in order:
        return
    for key in order[order.index(from_key) + 1 :]:
        st.session_state[key] = defaults.get(key)
    st.session_state.db_result_view_id = None
    # Stage outputs are immutable per run — a redo must start a new run dir.
    st.session_state.db_manifest = None


def _remove_protected_profile_files() -> None:
    """Delete exact source-derived profiles before the source is disconnected."""

    workdir = Path(st.session_state.db_workdir).resolve()
    failures: list[Path] = []
    for profile_path in workdir.rglob(PROFILE_FILENAME):
        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            failures.append(profile_path)
    if failures:
        raise OSError("protected Database profile cleanup failed")


def _active_manifest() -> RunManifest:
    """Return the session run, creating one if needed."""
    manifest = st.session_state.db_manifest
    workdir = st.session_state.db_workdir
    if manifest is None or manifest.run_dir.parent.parent != workdir:
        manifest = RunManifest.create(runs_dir=workdir / "runs")
        st.session_state.db_manifest = manifest
    return manifest


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
    step_header(1, "Intent", True)
    step_guide(
        what="How you plan to use the twin (informational — does not change training).",
        next_step="Connect a SQLite source database.",
    )
    intent_options = [
        "Development & testing",
        "QA / automated tests",
        "Demonstrations",
        "Analytics prototyping",
        "Safe sharing with partners",
    ]
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
    step_header(2, "Connect", st.session_state.db_source_path is not None)
    step_guide(
        what="Provide a SQLite database to learn from (upload or generate a sample).",
        next_step="Discover tables and relationships.",
    )

    source_choice = st.radio(
        "Source",
        ["Use a generated sample database", "Upload a SQLite file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source_choice == "Use a generated sample database":
        customer_count = st.slider("Customers to generate", 50, 2000, 400, step=50)
        if st.button("Generate sample database", icon=":material/auto_awesome:"):
            path = st.session_state.db_source_staging_dir / "database_a.db"
            build_sample_database(path, seed=42, customer_count=customer_count)
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            st.session_state.db_source_path = path
            st.session_state.db_source_hash = source_hash
            st.session_state.db_manifest = None
            _reset_downstream("db_source_path")
            db_session = ensure_workflow_session(
                state_key="db_workspace_session_id",
                workflow=WorkflowKind.DATABASE,
                title="Database Twin",
                source_fingerprint=source_hash,
                current_step="connect",
            )
            record_progress_once(
                db_session.session_id,
                macro_step="connect",
                stage_id="database_connect",
                stage_label="Connect source database",
                status=ProgressStatus.SUCCEEDED,
                message="Local source fingerprint recorded; credentials were not persisted.",
            )
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
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()
            source_hash = hashlib.sha256(uploaded_bytes).hexdigest()
        else:
            uploaded_bytes = None
            source_hash = None
        if uploaded is not None and source_hash != st.session_state.db_source_hash:
            path = st.session_state.db_source_staging_dir / "database_a.db"
            path.write_bytes(uploaded_bytes)
            st.session_state.db_source_path = path
            st.session_state.db_source_hash = source_hash
            st.session_state.db_uploaded_file_id = source_hash
            st.session_state.db_manifest = None
            _reset_downstream("db_source_path")
            db_session = ensure_workflow_session(
                state_key="db_workspace_session_id",
                workflow=WorkflowKind.DATABASE,
                title="Database Twin",
                source_fingerprint=source_hash,
                current_step="connect",
            )
            record_progress_once(
                db_session.session_id,
                macro_step="connect",
                stage_id="database_connect",
                stage_label="Connect source database",
                status=ProgressStatus.SUCCEEDED,
                message="Local source fingerprint recorded; credentials were not persisted.",
            )

    if (
        st.session_state.db_source_path is not None
        and not st.session_state.db_source_disconnected
    ):
        conn = sqlite3.connect(str(st.session_state.db_source_path))
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        conn.close()
        show_success(
            f"Connected to a local SQLite source — tables: {', '.join(tables)}"
        )
    elif st.session_state.db_source_disconnected:
        st.info("Source database is disconnected and its staged copy was removed.")

if st.session_state.db_source_path is None:
    st.stop()

workspace_tools = get_workspace_tools()
db_session = workspace_tools.get_session(
    GetSessionCommand(session_id=st.session_state.db_workspace_session_id or "missing")
)
if db_session is None:
    source_hash = st.session_state.db_source_hash or hashlib.sha256(
        Path(st.session_state.db_source_path).read_bytes()
    ).hexdigest()
    st.session_state.db_source_hash = source_hash
    db_session = ensure_workflow_session(
        state_key="db_workspace_session_id",
        workflow=WorkflowKind.DATABASE,
        title="Database Twin",
        source_fingerprint=source_hash,
        current_step="connect",
    )

source_adapter = SQLiteSourceAdapter({"path": str(st.session_state.db_source_path)})
manifest = _active_manifest()
metadata_dir = st.session_state.db_workdir / "metadata"


def _start_database_stage(
    *,
    macro_step: str,
    stage_id: str,
    stage_label: str,
    active_manifest: RunManifest,
) -> None:
    """Mark specialized execution as running without owning the stage itself."""
    workspace_tools.start_session(
        StartSessionCommand(
            session_id=db_session.session_id,
            run_id=active_manifest.run_id,
        )
    )
    record_progress_once(
        db_session.session_id,
        macro_step=macro_step,
        stage_id=stage_id,
        stage_label=stage_label,
        status=ProgressStatus.RUNNING,
        message="Running the specialized Database pipeline stage.",
    )


def _generation_configuration_fingerprint() -> str:
    return fingerprint_configuration(
        {
            "row_counts": {
                table: int(count)
                for table, count in st.session_state.db_row_counts.items()
            },
            "model_type": st.session_state.db_model_type,
            "dp_epsilon": (
                float(st.session_state.db_dp_epsilon)
                if st.session_state.db_model_type == "dp_gaussian_copula"
                else None
            ),
            "dp_bounds": st.session_state.db_dp_bounds,
            "category_overrides": st.session_state.db_category_overrides,
            "llm_text_enabled": bool(st.session_state.db_llm_text),
        }
    )


def _sync_workspace_progress() -> None:
    stages = (
        ("connect", "database_connect", "Connect source database", "db_source_path"),
        ("configure", "database_discovery", "Discover structure", "db_discovery"),
        ("configure", "database_profiling", "Understand values", "db_profile"),
        ("configure", "database_inference", "Infer semantics", "db_candidates"),
        ("configure", "database_approval", "Approve data contract", "db_contract"),
        ("configure", "database_training", "Train twin", "db_training_report"),
        ("configure", "database_artifact", "Export trained twin", "db_artifact_path"),
        ("configure", "database_disconnect", "Disconnect source", "db_source_disconnected"),
        ("generate", "database_generation", "Generate relational data", "db_relational_report"),
        ("results", "database_validation", "Validate results", "db_qa_report"),
        ("delivery", "database_delivery", "Deliver target database", "db_write_report"),
    )
    for macro_step, stage_id, label, state_key in stages:
        if st.session_state.get(state_key):
            progress_status = ProgressStatus.SUCCEEDED
            if state_key == "db_qa_report" and not st.session_state.db_qa_report.get(
                "hard_checks_passed", False
            ):
                progress_status = ProgressStatus.BLOCKED
            record_progress_once(
                db_session.session_id,
                macro_step=macro_step,
                stage_id=stage_id,
                stage_label=label,
                status=progress_status,
            )


def _publish_database_result() -> None:
    qa_report = st.session_state.db_qa_report
    if not qa_report:
        return
    hard_checks_passed = bool(qa_report.get("hard_checks_passed"))
    relational_tables = (st.session_state.db_relational_report or {}).get("tables") or {}
    integrity = qa_report.get("report", {}).get("integrity", {})
    fk_validity = integrity.get("fk_validity", {}).get("overall_fk_validity")
    artifact_paths = [
        ("Portable trained twin", st.session_state.db_artifact_path, True),
        ("Database QA report", manifest.output_path("qa_report.json"), True),
    ]
    if st.session_state.db_target_path:
        artifact_paths.extend(
            [
                ("Synthetic target database", st.session_state.db_target_path, True),
                (
                    "Target write report",
                    manifest.output_path("target_write_report.json"),
                    True,
                ),
            ]
        )
    result_view = stage_workflow_result_view(
        workspace_tools,
        db_session.session_id,
        workflow=WorkflowKind.DATABASE,
        manifest=manifest,
        validation_status=Status.PASS if hard_checks_passed else Status.FAIL,
        release_verdict=None,
        blockers=[] if hard_checks_passed else ["Database hard checks failed"],
        metrics={
            "table_count": len(relational_tables),
            "total_rows": sum(
                int(table.get("row_count") or 0)
                for table in relational_tables.values()
            ),
            "fk_validity": fk_validity,
            "writes_confirmed": (st.session_state.db_write_report or {})
            .get("validation_report", {})
            .get("all_writes_confirmed"),
        },
        previews=[
            PreviewDescriptor(
                label=table_name,
                kind="table",
                row_count=int(table.get("row_count") or 0),
                description=(
                    "Synthetic table preview is available in the active workflow."
                ),
            )
            for table_name, table in relational_tables.items()
        ],
        artifacts=artifact_paths,
    )
    if st.session_state.db_result_view_id:
        result_view = result_view.model_copy(
            update={"result_id": st.session_state.db_result_view_id}
        )
    workspace_tools.complete_session(CompleteSessionCommand(result=result_view))
    st.session_state.db_result_view_id = result_view.result_id


_sync_workspace_progress()
render_backend_progress(db_session.session_id)

# ---------------------------------------------------------------------------
# Step 2 - Discover
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(3, "Discover", st.session_state.db_discovery is not None)
    step_guide(
        what="Inspect tables, columns, primary keys, and foreign keys.",
        next_step="Understand values and meaning.",
    )
    if st.session_state.db_discovery is None:
        if st.button("Discover structure", icon=":material/travel_explore:"):
            manifest = _manifest_for_new_outputs("discovery.json")
            _start_database_stage(
                macro_step="configure",
                stage_id="database_discovery",
                stage_label="Discover structure",
                active_manifest=manifest,
            )
            result = run_discovery(source_adapter, manifest, config_path="config/project.yaml")
            if not result.is_success():
                record_failure(
                    db_session.session_id,
                    macro_step="configure",
                    stage_id="database_discovery",
                    stage_label="Discover structure",
                )
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
                    f"{fk.get('column')} → {fk.get('references_table')}.{fk.get('references_column')}"
                )
            overview_rows.append(
                {
                    "table": table_name,
                    "rows (source)": table.get("estimated_row_count") if table.get("estimated_row_count") is not None else "—",
                    "columns": len(table.get("columns") or {}),
                    "primary_key": ", ".join(pks) if pks else "-",
                    "foreign_keys": "; ".join(fk_bits) if fk_bits else "-",
                    "description": friendly_table_description(table_name),
                }
            )
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
    step_header(4, "Understand values", st.session_state.db_profile is not None)
    step_guide(
        what="Learn safe shape statistics (missingness, cardinality, distributions).",
        next_step="Approve how each column should be treated.",
    )
    if st.session_state.db_profile is None:
        with st.expander("Advanced: large tables"):
            chunked = st.checkbox(
                "Process large tables in chunks",
                help="Streams each table instead of loading the full sample at once. "
                "Recommended only for tables too large to fit in memory as a single sample.",
            )
            st.session_state.db_chunk_size = (
                st.number_input("Chunk size (rows)", min_value=100, value=10_000, step=100)
                if chunked
                else None
            )
        if st.button("Understand values", icon=":material/query_stats:"):
            manifest = _manifest_for_new_outputs("profile.json")
            _start_database_stage(
                macro_step="configure",
                stage_id="database_profiling",
                stage_label="Understand values",
                active_manifest=manifest,
            )
            result = run_profiling(
                source_adapter, st.session_state.db_discovery, manifest, "discovery.json", sample_limit=3000,
                chunk_size=st.session_state.db_chunk_size,
            )
            if not result.is_success():
                record_failure(
                    db_session.session_id,
                    macro_step="configure",
                    stage_id="database_profiling",
                    stage_label="Understand values",
                )
                show_user_error(
                    "We couldn't analyze values in your database.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_profile = load_profile(manifest.output_path(PROFILE_FILENAME))
            st.rerun()
    else:
        table_name = st.selectbox("Table", list(st.session_state.db_profile["tables"]), key="profile_table_select")
        columns = st.session_state.db_profile["tables"][table_name]["columns"]
        rows = [
            {
                "column": name,
                "dtype": col["inferred_dtype"],
                "null %": col["null_percentage"],
                "distinct": col["distinct_count"],
                "warnings": ", ".join(col["warnings"]) or "-",
            }
            for name, col in columns.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

if st.session_state.db_profile is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 4 - Review semantics
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(5, "Understand data", st.session_state.db_contract is not None)
    step_guide(
        what="Review column meanings, then approve the understanding used for training.",
        next_step="Train the twin.",
    )
    if st.session_state.db_candidates is None:
        if st.button("Understand data", icon=":material/psychology:"):
            _start_database_stage(
                macro_step="configure",
                stage_id="database_inference",
                stage_label="Infer semantics",
                active_manifest=manifest,
            )
            result = run_inference(
                source_adapter, st.session_state.db_discovery, st.session_state.db_profile, manifest,
                "discovery.json", "profile.json", sample_limit=5000,
                chunk_size=st.session_state.db_chunk_size,
            )
            if not result.is_success():
                record_failure(
                    db_session.session_id,
                    macro_step="configure",
                    stage_id="database_inference",
                    stage_label="Infer semantics",
                )
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
            _start_database_stage(
                macro_step="configure",
                stage_id="database_approval",
                stage_label="Approve data contract",
                active_manifest=manifest,
            )
            approval_result = run_contract_approval(
                dataset_id="database_a", source_fingerprint=st.session_state.db_discovery["source_fingerprint"],
                discovery_data=st.session_state.db_discovery, candidates_by_table=candidates["tables"],
                manifest=manifest, metadata_dir=metadata_dir, candidates_reference="semantic_candidates.json",
                decisions=decisions,
            )
            if not approval_result.is_success():
                record_failure(
                    db_session.session_id,
                    macro_step="configure",
                    stage_id="database_approval",
                    stage_label="Approve data contract",
                )
                show_user_error(
                    "We couldn't save the approved data understanding.",
                    next_action="Check local output storage and retry approval.",
                )
                st.stop()
            st.session_state.db_contract = load_dataset_contract(metadata_dir)
            st.rerun()
    elif st.session_state.db_contract is not None:
        show_success("Data understanding approved — ready to train the twin.")

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
    step_header(6, "Train twin", st.session_state.db_training_report is not None)
    step_guide(
        what="Train a portable twin from the approved understanding (privacy safeguards applied before fit).",
        next_step="Download the trained twin artifact.",
    )

    if st.session_state.db_training_report is None:
        _labels_by_model_type = {
            "safe_gaussian_copula": "Standard twin (fidelity-focused)",
            "dp_gaussian_copula": "Twin with differential privacy",
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
        with st.expander("Technical details — twin type", expanded=False):
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
                        upper = c2.number_input(f"{table_name}.{col_name} - upper bound", value=1_000_000.0, key=f"ub_{table_name}_{col_name}")
                        bounds[col_name] = (lower, upper)
            st.session_state.db_dp_bounds = bounds
            model_kwargs["epsilon_budget"] = st.session_state.db_dp_epsilon
            model_kwargs["column_bounds"] = bounds
        else:
            st.session_state.db_dp_bounds = {}

        if st.button("Train", icon=":material/model_training:"):
            manifest = _manifest_for_new_outputs("training_report.json")
            _start_database_stage(
                macro_step="configure",
                stage_id="database_training",
                stage_label="Train twin",
                active_manifest=manifest,
            )
            result = run_training_and_sampling(
                source_adapter, contract, manifest, "dataset_contract.json", sample_limit=5000,
                num_rows_to_generate=5, seed=1, model_type=st.session_state.db_model_type, model_kwargs=model_kwargs,
            )
            if not result.is_success():
                record_failure(
                    db_session.session_id,
                    macro_step="configure",
                    stage_id="database_training",
                    stage_label="Train twin",
                )
                show_user_error(
                    "Training the twin failed.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_training_report = load_training_report(Path(result.output_references[0]))
            st.rerun()
    else:
        for table_name, table_report in st.session_state.db_training_report["tables"].items():
            evidence = table_report["fit_evidence"]
            st.markdown(f"**{table_name}** — trained {len(evidence['trained_columns'])} columns")
            if "privacy_summary" in evidence:
                st.caption(
                    f":material/shield_lock: epsilon spent: {evidence['privacy_summary']['total_epsilon_spent']:.3f} "
                    f"/ {evidence['privacy_summary']['epsilon_budget']:.3f}"
                )

if st.session_state.db_training_report is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 6 - Export trained twin
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(7, "Download trained twin", st.session_state.db_artifact_path is not None)
    step_guide(
        what="Package the trained twin as a portable artifact you can move without the source.",
        next_step="Disconnect the source database.",
    )
    if st.session_state.db_artifact_path is None:
        if st.button("Download trained twin", icon=":material/archive:"):
            artifact_path = _active_manifest().run_dir / "artifacts" / "generator-database_a-1.0.0.zip"
            manifest = _manifest_for_new_outputs(extra_paths=[artifact_path])
            _start_database_stage(
                macro_step="configure",
                stage_id="database_artifact",
                stage_label="Export trained twin",
                active_manifest=manifest,
            )
            result = run_artifact_export(
                "database_a", "1.0.0", st.session_state.db_discovery, contract, st.session_state.db_profile,
                st.session_state.db_training_report, manifest.run_id, manifest.code_version, manifest,
                "training_report.json",
            )
            if not result.is_success():
                record_failure(
                    db_session.session_id,
                    macro_step="configure",
                    stage_id="database_artifact",
                    stage_label="Export trained twin",
                )
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
    step_header(8, "Disconnect source", st.session_state.db_source_disconnected)
    step_guide(
        what="Stop using the source so generation runs from the trained twin only.",
        next_step="Generate synthetic data from the artifact.",
    )
    if not st.session_state.db_source_disconnected:
        st.caption("Disconnect before generating so the twin runs without source access.")
        if st.button("Disconnect source", icon=":material/link_off:"):
            try:
                _remove_protected_profile_files()
                remove_source_staging(st.session_state.db_source_staging_dir)
            except (OSError, ValueError):
                st.error(
                    "The source was not disconnected because private source/profile "
                    "cleanup could not be verified."
                )
                st.stop()
            st.session_state.db_source_staging_dir = None
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
    step_header(9, "Generate", st.session_state.db_relational_report is not None)
    step_guide(
        what="Create synthetic tables from the portable artifact. Tune category mixes if needed.",
        next_step="Validate integrity and quality.",
    )

    artifact_only_banner(
        disconnected=st.session_state.db_source_disconnected,
        has_artifact=bool(st.session_state.db_artifact_path or st.session_state.db_loaded_artifact),
    )

    row_counts = {}
    for table_name in loaded_artifact.manifest["tables"]:
        default = st.session_state.db_row_counts.get(table_name, 200)
        row_counts[table_name] = st.number_input(
            f"Rows to generate — {table_name}", min_value=1, value=default, step=50, key=f"rowcount_{table_name}"
        )
    st.session_state.db_row_counts = row_counts

    st.markdown("**Tweak a category's distribution (no retraining)**")
    category_overrides: dict[str, dict[str, dict[str, float]]] = {}
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

            with st.expander(f"{table_name}.{col_name} — learned: " + ", ".join(f"{k} {v:.0%}" for k, v in learned.items())):
                chart_df = pd.DataFrame({"category": list(learned), "proportion": list(learned.values())})
                st.bar_chart(chart_df, x="category", y="proportion", horizontal=True)

                enable = st.checkbox("Override this distribution", key=f"override_enable_{table_name}_{col_name}")
                if enable:
                    weights = {}
                    for category in learned:
                        weights[category] = st.slider(
                            category, 0.0, 1.0, float(learned[category]), key=f"override_{table_name}_{col_name}_{category}"
                        )
                    total = sum(weights.values()) or 1.0
                    normalized = {k: v / total for k, v in weights.items()}
                    st.caption("Normalized target: " + ", ".join(f"{k} {v:.0%}" for k, v in normalized.items()))
                    category_overrides.setdefault(table_name, {})[col_name] = normalized

    st.session_state.db_category_overrides = category_overrides

    free_text_cols = list_contract_free_text_columns(contract)
    st.markdown("**Text-heavy columns (optional LLM)**")
    if free_text_cols:
        st.dataframe(free_text_cols, hide_index=True, width="stretch")
        st.session_state.db_llm_text = st.toggle(
            "Use LLM for text-heavy columns only",
            value=bool(st.session_state.db_llm_text),
            help="After relational generation, rewrites free_text columns with the shared "
            "text engine (LLM when configured, otherwise local templates). Never uses source text.",
        )
    else:
        st.session_state.db_llm_text = False
        st.caption("No free_text columns in this contract — LLM option is hidden.")

    if st.button("Generate synthetic database", icon=":material/auto_fix_high:", type="primary"):
        generation_fingerprint = _generation_configuration_fingerprint()
        is_regeneration = st.session_state.db_relational_report is not None
        if is_regeneration:
            # Preserve the approved contract and trained artifact, but keep the
            # previous completed session/result immutable by rotating run identity.
            st.session_state.db_workspace_session_id = None
            db_session = ensure_workflow_session(
                state_key="db_workspace_session_id",
                workflow=WorkflowKind.DATABASE,
                title="Database Twin",
                source_fingerprint=st.session_state.db_source_hash,
                configuration_fingerprint=generation_fingerprint,
                current_step="generate",
            )
            st.session_state.db_workdir = workflow_run_dir(
                db_session.session_id, WorkflowKind.DATABASE
            )
            manifest = RunManifest.create(
                runs_dir=st.session_state.db_workdir / "runs"
            )
            st.session_state.db_manifest = manifest
        else:
            db_session = workspace_tools.set_configuration_fingerprint(
                SetConfigurationFingerprintCommand(
                    session_id=db_session.session_id,
                    configuration_fingerprint=generation_fingerprint,
                )
            )

        for key in (
            "db_relational_report",
            "db_qa_report",
            "db_target_path",
            "db_write_report",
            "db_result_view_id",
            "db_run_metrics",
            "db_llm_evidence",
        ):
            st.session_state[key] = defaults[key]

        max_rows = max(int(v) for v in row_counts.values()) if row_counts else 0
        # Streaming activates for large runs; small/E2E paths keep in-memory default.
        gen_batch_size = 10_000 if max_rows >= 10_000 else None
        manifest = _manifest_for_new_outputs("relational_generation_report.json")
        _start_database_stage(
            macro_step="generate",
            stage_id="database_generation",
            stage_label="Generate relational data",
            active_manifest=manifest,
        )
        with measure_generation() as run_stats:
            result = run_relational_generation(
                contract, adapters_by_table, row_counts, manifest, contract_reference="dataset_contract.json",
                seed=7, category_overrides_by_table=category_overrides or None,
                batch_size=gen_batch_size,
            )
            if result.is_success() and st.session_state.db_llm_text and free_text_cols:
                report_path = manifest.output_path("relational_generation_report.json")
                interim = load_relational_generation_report(report_path)
                st.session_state.db_llm_evidence = apply_llm_free_text_to_tables(
                    interim,
                    free_text_cols,
                    seed=7,
                    max_llm_rows=50,
                )
            else:
                st.session_state.db_llm_evidence = None
        if not result.is_success():
            record_failure(
                db_session.session_id,
                macro_step="generate",
                stage_id="database_generation",
                stage_label="Generate relational data",
            )
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
        # Preview only — never reload the full frame into session_state.
        generated_df = pd.read_csv(report["tables"][table_name]["path"], nrows=50)
        st.dataframe(generated_df, width="stretch")
        st.caption(
            f"`{table_name}`: {int(report['tables'][table_name].get('row_count') or 0):,} rows "
            "(showing first 50)"
        )

        drift_rows = profile_vs_synthetic_drift_rows(
            st.session_state.db_profile,
            table_name,
            pd.read_csv(report["tables"][table_name]["path"], nrows=500),
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

if st.session_state.db_relational_report is None:
    st.stop()

# ---------------------------------------------------------------------------
# Step 10 - Validate
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(10, "Validate", st.session_state.db_qa_report is not None)
    step_guide(
        what="Confirm relational integrity and available quality signals.",
        next_step="Export the synthetic database.",
    )

    if st.session_state.db_qa_report is None:
        if st.button("Validate results", icon=":material/fact_check:"):
            manifest = _manifest_for_new_outputs("qa_report.json")
            _start_database_stage(
                macro_step="results",
                stage_id="database_validation",
                stage_label="Validate results",
                active_manifest=manifest,
            )
            result = run_qa_validation(
                contract, st.session_state.db_relational_report, manifest,
                relational_report_reference="relational_generation_report.json",
                reference_profile=st.session_state.db_profile,
            )
            if not result.is_success():
                record_failure(
                    db_session.session_id,
                    macro_step="results",
                    stage_id="database_validation",
                    stage_label="Validate results",
                )
                show_user_error(
                    "Validation failed.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_qa_report = load_qa_report(manifest.output_path("qa_report.json"))
            _publish_database_result()
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
            fk_label=f"{float(fk):.0%}" if fk is not None else "—",
            pk_label=("unique" if pk_ok else "duplicates") if pk_ok is not None else "—",
        )
        fidelity_rows = fidelity_drift_rows(qa, limit=12)
        render_data_drift_panel(
            mode="database",
            rows=fidelity_rows or None,
            footnote=(
                "Scores come from QA fidelity checks against privacy-safe profile bounds / vocabularies."
                if fidelity_rows
                else "No fidelity columns available yet — integrity checks above are still authoritative."
            ),
        )

if st.session_state.db_qa_report is None or not st.session_state.db_qa_report[
    "hard_checks_passed"
]:
    if st.session_state.db_result_view_id:
        failed_result = workspace_tools.get_result(
            GetResultCommand(
                session_id=db_session.session_id,
                result_id=st.session_state.db_result_view_id,
            )
        )
        if failed_result is not None:
            render_result_summary(failed_result)
            render_project_save(db_session.session_id)
    st.stop()

# ---------------------------------------------------------------------------
# Step 11 - Export to target database
# ---------------------------------------------------------------------------
with st.container(border=True):
    step_header(11, "Export", st.session_state.db_write_report is not None)
    step_guide(
        what="Write the synthetic database and download it.",
        next_step="Done — you have a validated synthetic database.",
    )

    if st.session_state.db_write_report is None:
        if st.button("Export to target database", icon=":material/database_upload:", type="primary"):
            target_path = st.session_state.db_workdir / "database_b.db"
            manifest = _manifest_for_new_outputs("target_write_report.json")
            # This optional write happens after QA has completed the workspace
            # session. It must not reopen specialized execution as RUNNING.
            result = run_target_write(
                target_path, contract, st.session_state.db_relational_report, st.session_state.db_qa_report, manifest,
                qa_report_reference="qa_report.json",
            )
            if not result.is_success():
                record_progress_once(
                    db_session.session_id,
                    macro_step="delivery",
                    stage_id="database_delivery",
                    stage_label="Deliver target database",
                    status=ProgressStatus.WARN,
                    message=(
                        "Validated generation succeeded, but post-completion target "
                        "delivery failed. See its in-session details."
                    ),
                )
                show_user_error(
                    "Export to the target database failed.",
                    technical=result.errors,
                )
                st.stop()
            st.session_state.db_target_path = target_path
            st.session_state.db_write_report = load_target_write_report(
                manifest.output_path("target_write_report.json")
            )
            _publish_database_result()
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


result_view_id = st.session_state.db_result_view_id
if result_view_id:
    persisted_result = workspace_tools.get_result(
        GetResultCommand(
            session_id=db_session.session_id,
            result_id=result_view_id,
        )
    )
    if persisted_result is not None:
        render_result_summary(persisted_result)
        render_project_save(db_session.session_id)
