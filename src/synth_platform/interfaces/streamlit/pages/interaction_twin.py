"""Customer Interaction Twin: transcript -> sanitized structured SSOT package."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import streamlit as st

from synth_platform.application.dto.tool_commands import (
    CompleteSessionCommand,
    GetResultCommand,
    ReadRuntimeSettingsCommand,
    StartSessionCommand,
)
from synth_platform.application.dto.workspace import ProgressStatus, WorkflowKind
from synth_platform.application.services.result_presentation import (
    interaction_result_view,
)
from synth_platform.application.workflows.interaction_twin import (
    InteractionWorkflowResult,
    run_interaction_twin,
)
from synth_platform.bootstrap import build_guarded_chat_model
from synth_platform.domain.validation.models import Status
from synth_platform.engine.interactions.service import render_sanitized_source
from synth_platform.infrastructure.storage.source_staging import (
    create_source_staging,
    remove_source_staging,
)
from synth_platform.interfaces.streamlit.components.common.ux import platform_intro
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

st.title("Customer Interaction Twin")
platform_intro()
st.caption(
    "Paste or upload a customer interaction. The workflow sanitizes it before any "
    "optional local-model call, builds one structured SSOT, validates privacy and "
    "source replay, and packages only sanitized artifacts."
)

_DEFAULTS = {
    "interaction_workdir": None,
    "interaction_source_hash": None,
    "interaction_result": None,
    "interaction_error": None,
    "interaction_seed": 42,
    "interaction_locale": "en_US",
    "interaction_model_enabled": False,
    "interaction_model_name": "qwen3:8b",
    "interaction_model_host": "http://localhost:11434",
    "interaction_configuration_hash": None,
    "interaction_workspace_session_id": None,
    "interaction_result_view_id": None,
    "interaction_guardrail_reports": [],
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)


def _promote_released_result(
    result: InteractionWorkflowResult,
    durable_runs_dir: Path,
) -> InteractionWorkflowResult:
    """Move only release-admitted outputs into durable workspace storage."""
    if not result.released:
        return result
    source_run_dir = result.run_manifest.run_dir
    target_run_dir = durable_runs_dir / result.run_manifest.run_id
    durable_runs_dir.mkdir(parents=True, exist_ok=True)

    def _target(path: Path) -> Path:
        return target_run_dir / path.relative_to(source_run_dir)

    try:
        shutil.move(str(source_run_dir), str(target_run_dir))
    except Exception:
        shutil.rmtree(target_run_dir, ignore_errors=True)
        raise

    result.run_manifest.run_dir = target_run_dir
    return replace(
        result,
        sanitized_source_path=_target(result.sanitized_source_path),
        ssot_path=_target(result.ssot_path),
        validation_path=_target(result.validation_path),
        artifact_manifest_path=_target(result.artifact_manifest_path),
        package_path=(
            _target(result.package_path) if result.package_path is not None else None
        ),
    )


with st.container(border=True):
    st.markdown("### 1. Provide the interaction")
    input_mode = st.radio(
        "Input method",
        ["Paste transcript", "Upload TXT/LOG"],
        horizontal=True,
    )
    source_name = "pasted_transcript.txt"
    source_text = ""
    if input_mode == "Paste transcript":
        source_text = st.text_area(
            "Transcript",
            height=240,
            placeholder=(
                "Customer: I cannot access my account.\n"
                "Agent: I can help you reset the credentials."
            ),
        )
    else:
        uploaded = st.file_uploader("Transcript file", type=["txt", "log"])
        if uploaded is not None:
            source_name = uploaded.name
            try:
                source_text = uploaded.getvalue().decode("utf-8-sig")
            except UnicodeDecodeError:
                st.error("The transcript must be UTF-8 text.")

    current_hash = (
        hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        if source_text.strip()
        else None
    )
    if current_hash != st.session_state.interaction_source_hash:
        st.session_state.interaction_source_hash = current_hash
        st.session_state.interaction_result = None
        st.session_state.interaction_error = None
        st.session_state.interaction_result_view_id = None
        st.session_state.interaction_guardrail_reports = []

with st.container(border=True):
    st.markdown("### 2. Configure the SSOT")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.interaction_seed = st.number_input(
            "Deterministic seed",
            min_value=0,
            max_value=2_147_483_647,
            value=int(st.session_state.interaction_seed),
        )
    with c2:
        st.session_state.interaction_locale = st.selectbox(
            "Locale",
            ["en_US"],
            help="The initial deterministic privacy rules are US/English-centric.",
        )

    st.session_state.interaction_model_enabled = st.checkbox(
        "Use one local Ollama call for semantic enrichment",
        value=bool(st.session_state.interaction_model_enabled),
        help=(
            "The model receives only the sanitized transcript and can return enum codes only. "
            "If unavailable or invalid, the workflow uses its deterministic fallback."
        ),
    )
    runtime = get_workspace_tools().runtime_settings(ReadRuntimeSettingsCommand())
    st.session_state.interaction_model_host = runtime.local_model_host
    if st.session_state.interaction_model_enabled:
        st.session_state.interaction_model_name = st.text_input(
            "Ollama model",
            value=st.session_state.interaction_model_name,
        )
        st.caption(
            f"Guardrails `{runtime.guardrail_policy_version}` enforced · "
            f"model network scope: {runtime.local_model_network_scope}."
        )

    interaction_configuration_hash = fingerprint_configuration(
        {
            "seed": int(st.session_state.interaction_seed),
            "locale": st.session_state.interaction_locale,
            "model_enabled": bool(st.session_state.interaction_model_enabled),
            "model_name": (
                st.session_state.interaction_model_name
                if st.session_state.interaction_model_enabled
                else None
            ),
            "model_host": (
                st.session_state.interaction_model_host
                if st.session_state.interaction_model_enabled
                else None
            ),
        }
    )
    if (
        st.session_state.interaction_configuration_hash
        not in {None, interaction_configuration_hash}
    ):
        st.session_state.interaction_result = None
        st.session_state.interaction_error = None
        st.session_state.interaction_result_view_id = None
        st.session_state.interaction_guardrail_reports = []
    st.session_state.interaction_configuration_hash = interaction_configuration_hash

    interaction_session = None
    if current_hash:
        interaction_session = ensure_workflow_session(
            state_key="interaction_workspace_session_id",
            workflow=WorkflowKind.INTERACTION,
            title="Customer Interaction Twin",
            source_fingerprint=current_hash,
            configuration_fingerprint=interaction_configuration_hash,
            current_step="upload",
        )
        st.session_state.interaction_workdir = workflow_run_dir(
            interaction_session.session_id, WorkflowKind.INTERACTION
        )
        record_progress_once(
            interaction_session.session_id,
            macro_step="upload",
            stage_id="interaction_input",
            stage_label="Upload interaction",
            status=ProgressStatus.SUCCEEDED,
            message="Source fingerprint recorded; raw text remains in memory only.",
        )
        record_progress_once(
            interaction_session.session_id,
            macro_step="configure",
            stage_id="interaction_configuration",
            stage_label="Configure interaction SSOT",
            status=ProgressStatus.SUCCEEDED,
        )

    create_clicked = st.button(
        "Create Interaction Twin",
        type="primary",
        icon=":material/record_voice_over:",
        disabled=not bool(source_text.strip()),
        use_container_width=True,
    )
    if create_clicked and interaction_session is not None:
        model = None
        if st.session_state.interaction_model_enabled:
            model = build_guarded_chat_model(
                "interaction_semantics",
                model=st.session_state.interaction_model_name,
            )
        workspace_tools = get_workspace_tools()
        workspace_tools.start_session(
            StartSessionCommand(session_id=interaction_session.session_id)
        )
        record_progress_once(
            interaction_session.session_id,
            macro_step="generate",
            stage_id="interaction_generation",
            stage_label="Sanitize, structure, validate, and package",
            status=ProgressStatus.RUNNING,
            message="Running the specialized Interaction workflow.",
        )
        staging_dir = create_source_staging("interaction")
        promoted_run_dir: Path | None = None
        try:
            with st.spinner("Sanitizing, structuring, validating, and packaging..."):
                workflow_result = run_interaction_twin(
                    source_text,
                    source_name=source_name,
                    runs_dir=staging_dir / "runs",
                    seed=int(st.session_state.interaction_seed),
                    locale=st.session_state.interaction_locale,
                    model=model,
                )
                if workflow_result.released:
                    workflow_result = _promote_released_result(
                        workflow_result,
                        st.session_state.interaction_workdir / "runs",
                    )
                    promoted_run_dir = workflow_result.run_manifest.run_dir
            record_progress_once(
                interaction_session.session_id,
                macro_step="generate",
                stage_id="interaction_generation",
                stage_label="Sanitize, structure, validate, and package",
                status=(
                    ProgressStatus.SUCCEEDED
                    if workflow_result.released
                    else ProgressStatus.BLOCKED
                ),
                counts={"turns": len(workflow_result.sanitized_transcript.turns)},
            )
            result_view = interaction_result_view(
                workspace_tools, interaction_session.session_id, workflow_result
            )
            workspace_tools.complete_session(
                CompleteSessionCommand(result=result_view)
            )
            st.session_state.interaction_result = workflow_result
            st.session_state.interaction_result_view_id = result_view.result_id
            st.session_state.interaction_error = None
        except Exception as exc:  # noqa: BLE001 - UI boundary
            if promoted_run_dir is not None:
                shutil.rmtree(promoted_run_dir, ignore_errors=True)
            st.session_state.interaction_result = None
            st.session_state.interaction_error = type(exc).__name__
            st.session_state.interaction_result_view_id = None
            record_progress_once(
                interaction_session.session_id,
                macro_step="generate",
                stage_id="interaction_generation",
                stage_label="Sanitize, structure, validate, and package",
                status=ProgressStatus.FAILED,
                message=f"Workflow failed ({type(exc).__name__}).",
            )
        finally:
            reports = []
            if model is not None:
                for attribute in ("last_input_report", "last_output_report"):
                    report = getattr(model, attribute, None)
                    if report is not None:
                        reports.append(report.model_dump(mode="json"))
            st.session_state.interaction_guardrail_reports = reports
            remove_source_staging(staging_dir)

if interaction_session is not None:
    render_backend_progress(interaction_session.session_id)

if st.session_state.interaction_error:
    st.error(
        "The interaction could not be created. No raw transcript was persisted. "
        f"Error type: {st.session_state.interaction_error}."
    )

result = st.session_state.interaction_result
if result is not None:
    with st.container(border=True):
        st.markdown("### 3. Review sanitized source")
        st.caption(
            "Speaker labels and detected sensitive values are replaced before optional model use."
        )
        st.code(render_sanitized_source(result.sanitized_transcript), language="text")

    with st.container(border=True):
        st.markdown("### 4. Structured SSOT")
        st.json(result.ssot.model_dump(mode="json"))
        extraction = result.ssot.extraction
        if extraction.model_requested and not extraction.model_used:
            st.warning(
                "Local model enhancement was not accepted; the deterministic SSOT was used. "
                f"Reason: {extraction.fallback_reason}."
            )

    with st.container(border=True):
        st.markdown("### 5. Privacy and replay validation")
        verdict = result.validation_report.release.verdict
        if verdict == Status.PASS:
            st.success("Validation passed. The package is ready.")
        elif verdict == Status.WARN:
            st.warning("Critical validation passed with non-blocking review warnings.")
        else:
            st.error(
                "Validation blocked release: "
                + ", ".join(result.validation_report.release.blocking)
            )
        check_rows = [
            {
                "check": check.name,
                "dimension": check.dimension,
                "status": check.status.value,
                "critical": check.is_critical,
                "metric": check.metric,
                "threshold": check.threshold,
            }
            for check in result.validation_report.validation.checks
        ]
        st.dataframe(check_rows, use_container_width=True, hide_index=True)
        with st.expander("Validation metrics and limitations"):
            st.json(result.validation_report.model_dump(mode="json"))
        if st.session_state.interaction_guardrail_reports:
            with st.expander("Model guardrail evidence", expanded=False):
                st.caption(
                    "Category/count evidence only; prompts and matched values are never stored."
                )
                st.json(st.session_state.interaction_guardrail_reports)

    with st.container(border=True):
        st.markdown("### 6. Download")
        if result.package_path is not None:
            st.download_button(
                "Download Interaction Twin package",
                data=result.package_path.read_bytes(),
                file_name=result.package_path.name,
                mime="application/zip",
                icon=":material/download:",
                use_container_width=True,
            )
            st.caption(
                "Package contents: sanitized_source.txt, interaction_ssot.json, "
                "validation_report.json, and manifest.json."
            )
        else:
            st.error(
                "No package was created because a critical validation check failed."
            )


    result_view_id = st.session_state.interaction_result_view_id
    if result_view_id and interaction_session is not None:
        persisted_result = get_workspace_tools().get_result(
            GetResultCommand(
                session_id=interaction_session.session_id,
                result_id=result_view_id,
            )
        )
        if persisted_result is not None:
            render_result_summary(persisted_result)
            render_project_save(interaction_session.session_id)
