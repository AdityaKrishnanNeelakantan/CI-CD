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

from synth_platform.application.services.transfer_service import TransferService
from synth_platform.errors import TransferBlockedError
from synth_platform.domain.privacy.llm_policy import LlmPolicyError
from synth_platform.infrastructure.persistence.platform_db import get_platform_db
from synth_platform.domain.product_settings import read_generation_defaults
from synth_platform.engine.discovery.database.adapters.errors import SourceAdapterError
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
from synth_platform.application.workflows.database_twin import (
    PROFILE_FILENAME,
    RunManifest,
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
from synth_platform.interfaces.streamlit.database_source_ui import (
    PostgresConnectionDetails,
    build_source_adapter,
    postgres_connection_config,
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
    "db_source_type": None,
    "db_source_config": None,
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
    current_hint="Source → Understand → Train twin → Download artifact → Disconnect → Generate → Validate → Export",
)
product_defaults = read_generation_defaults(get_platform_db())

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
    # Stage outputs are immutable per run — a redo must start a new run dir.
    st.session_state.db_manifest = None


def _set_source(source_type: str, connection_config: dict, source_path: Path | None = None) -> None:
    st.session_state.db_source_type = source_type
    st.session_state.db_source_config = connection_config
    st.session_state.db_source_path = source_path or source_type
    st.session_state.db_manifest = None
    _reset_downstream("db_source_path")
    st.session_state.db_source_type = source_type
    st.session_state.db_source_config = connection_config
    st.session_state.db_source_path = source_path or source_type


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
        what="Provide a SQLite or PostgreSQL database to learn from.",
        next_step="Discover tables and relationships.",
    )

    source_choice = st.radio(
        "Source",
        ["Use a generated sample database", "Upload a SQLite file", "Connect to PostgreSQL"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source_choice == "Use a generated sample database":
        customer_count = st.slider("Customers to generate", 50, 2000, 400, step=50)
        if st.button("Generate sample database", icon=":material/auto_awesome:"):
            path = st.session_state.db_workdir / "database_a.db"
            build_sample_database(path, seed=42, customer_count=customer_count)
            _set_source("sqlite", {"path": str(path)}, source_path=path)
    elif source_choice == "Upload a SQLite file":
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
            st.session_state.db_uploaded_file_id = uploaded.file_id
            _set_source("sqlite", {"path": str(path)}, source_path=path)
    else:
        pg_col1, pg_col2 = st.columns([2, 1])
        with pg_col1:
            pg_host = st.text_input("Host", placeholder="db.example.com")
            pg_database = st.text_input("Database name")
            pg_username = st.text_input("Username")
            pg_password = st.text_input("Password", type="password")
        with pg_col2:
            pg_port = st.number_input("Port", min_value=1, max_value=65535, value=5432, step=1)
            pg_sslmode = st.selectbox("SSL mode", ["require", "verify-full", "verify-ca", "disable"])
            pg_schemas = st.text_input("Allowed schemas", value="public")
            pg_tables = st.text_input("Allowed tables", placeholder="optional: public.customers, public.orders")

        if st.button("Connect to PostgreSQL", icon=":material/database:"):
            schemas = tuple(s.strip() for s in pg_schemas.split(",") if s.strip())
            tables = tuple(t.strip() for t in pg_tables.split(",") if t.strip()) or None
            details = PostgresConnectionDetails(
                host=pg_host,
                port=int(pg_port),
                database=pg_database,
                username=pg_username,
                password=pg_password,
                sslmode=pg_sslmode,
                allowed_schemas=schemas,
                allowed_tables=tables,
            )
            config = postgres_connection_config(details)
            try:
                adapter = build_source_adapter("postgresql", config)
                health = adapter.test_connection()
                if not health.get("healthy", False):
                    raise SourceAdapterError(health.get("error", "PostgreSQL health check failed"))
                close = getattr(adapter, "close", None)
                if close is not None:
                    close()
            except Exception as exc:
                show_user_error(
                    "We couldn't connect to PostgreSQL.",
                    technical=[str(exc)],
                )
            else:
                _set_source("postgresql", config)
                show_success(
                    f"Connected to PostgreSQL database `{pg_database}` as `{pg_username}`."
                )

    if st.session_state.db_source_path is not None:
        if st.session_state.db_source_type == "sqlite":
            conn = sqlite3.connect(str(st.session_state.db_source_path))
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            conn.close()
            show_success(f"Connected to `{st.session_state.db_source_path.name}` — tables: {', '.join(tables)}")
        elif st.session_state.db_source_type == "postgresql":
            show_success("Connected to PostgreSQL source.")

if st.session_state.db_source_path is None:
    st.stop()

source_adapter = build_source_adapter(
    st.session_state.db_source_type,
    st.session_state.db_source_config,
)
manifest = _active_manifest()
metadata_dir = st.session_state.db_workdir / "metadata"

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
            result = run_profiling(
                source_adapter, st.session_state.db_discovery, manifest, "discovery.json", sample_limit=3000,
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
            result = run_inference(
                source_adapter, st.session_state.db_discovery, st.session_state.db_profile, manifest,
                "discovery.json", "profile.json", sample_limit=5000,
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
            model_kwargs["epsilon_budget"] = st.session_state.db_dp_epsilon
            model_kwargs["column_bounds"] = bounds

        if st.button("Train", icon=":material/model_training:"):
            manifest = _manifest_for_new_outputs("training_report.json")
            result = run_training_and_sampling(
                source_adapter, contract, manifest, "dataset_contract.json", sample_limit=5000,
                num_rows_to_generate=5, seed=1, model_type=st.session_state.db_model_type, model_kwargs=model_kwargs,
            )
            if not result.is_success():
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
    step_header(8, "Disconnect source", st.session_state.db_source_disconnected)
    step_guide(
        what="Stop using the source so generation runs from the trained twin only.",
        next_step="Generate synthetic data from the artifact.",
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
        default = st.session_state.db_row_counts.get(table_name, int(product_defaults["default_record_count"]))
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
        max_rows = max(int(v) for v in row_counts.values()) if row_counts else 0
        # Streaming activates for large runs; small/E2E paths keep in-memory default.
        gen_batch_size = 10_000 if max_rows >= 10_000 else None
        manifest = _manifest_for_new_outputs("relational_generation_report.json")
        with measure_generation() as run_stats:
            result = run_relational_generation(
                contract, adapters_by_table, row_counts, manifest, contract_reference="dataset_contract.json",
                seed=7, category_overrides_by_table=category_overrides or None,
                batch_size=gen_batch_size,
            )
            if result.is_success() and st.session_state.db_llm_text and free_text_cols:
                try:
                    report_path = manifest.output_path("relational_generation_report.json")
                    interim = load_relational_generation_report(report_path)
                    st.session_state.db_llm_evidence = apply_llm_free_text_to_tables(
                        interim,
                        free_text_cols,
                        seed=7,
                        max_llm_rows=50,
                    )
                except LlmPolicyError as exc:
                    show_user_error(
                        "External LLM providers are disabled in air-gapped mode.",
                        technical=exc,
                        next_action="Enable local Ollama or set SYNTH_ALLOW_EXTERNAL_LLM=true to use OpenAI/Groq.",
                    )
                    st.stop()
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
            get_platform_db().record_run(
                workflow_type="database",
                project_name=st.session_state.db_intent,
                status="completed" if st.session_state.db_qa_report["hard_checks_passed"] else "failed",
                validation_status="PASS" if st.session_state.db_qa_report["hard_checks_passed"] else "FAIL",
                validation_passed=bool(st.session_state.db_qa_report["hard_checks_passed"]),
                output_id="database_b.db",
                metadata={
                    "intent": st.session_state.db_intent,
                    "source_type": st.session_state.db_source_type,
                    "row_counts_by_table": {
                        table: data.get("row_count")
                        for table, data in st.session_state.db_relational_report.get("tables", {}).items()
                    },
                    "manifest_run_id": manifest.run_id,
                },
            )
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

if st.session_state.db_qa_report is None:
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

    if not st.session_state.db_qa_report["hard_checks_passed"]:
        try:
            TransferService(transfer_recorder=get_platform_db()).downloadable_bytes(
                workflow="database_twin",
                output_id="database_b.db",
                validation_report=st.session_state.db_qa_report,
                data=b"",
                metadata={"file_name": "database_b.db", "intent": st.session_state.db_intent},
            )
        except TransferBlockedError as exc:
            block_reason = str(exc)
        st.download_button(
            "Download Database B",
            b"",
            file_name="database_b.db",
            icon=":material/download:",
            disabled=True,
        )
        st.warning(block_reason)
        st.stop()

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

        try:
            transfer = TransferService(transfer_recorder=get_platform_db()).downloadable_file(
                workflow="database_twin",
                output_id="database_b.db",
                validation_report=st.session_state.db_qa_report,
                path=st.session_state.db_target_path,
                metadata={"file_name": "database_b.db", "intent": st.session_state.db_intent},
            )
            with transfer.path.open("rb") as f:
                st.download_button("Download Database B", f, file_name="database_b.db", icon=":material/download:")
        except TransferBlockedError as exc:
            st.download_button(
                "Download Database B",
                b"",
                file_name="database_b.db",
                icon=":material/download:",
                disabled=True,
            )
            st.warning(str(exc))
        mode_outcomes(
            [
                "Portable trained twin artifact",
                "Synthetic relational database",
                "Validation / QA report",
                "Source-free generation after disconnect (this session)",
            ]
        )
        st.caption(f"Intent: {st.session_state.db_intent}. Privacy safeguards applied during training.")
