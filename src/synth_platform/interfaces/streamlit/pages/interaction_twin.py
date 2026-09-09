"""Customer Interaction Twin: transcript -> sanitized structured SSOT package."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from synth_platform.application.workflows.interaction_twin import run_interaction_twin
from synth_platform.domain.validation.models import Status
from synth_platform.interfaces.streamlit.components.common.ux import platform_intro

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
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)

if st.session_state.interaction_workdir is None:
    st.session_state.interaction_workdir = Path(
        tempfile.mkdtemp(prefix="interaction_twin_demo_")
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
    if st.session_state.interaction_model_enabled:
        model_col, host_col = st.columns(2)
        with model_col:
            st.session_state.interaction_model_name = st.text_input(
                "Ollama model",
                value=st.session_state.interaction_model_name,
            )
        with host_col:
            st.session_state.interaction_model_host = st.text_input(
                "Ollama host",
                value=st.session_state.interaction_model_host,
            )

    create_clicked = st.button(
        "Create Interaction Twin",
        type="primary",
        icon=":material/record_voice_over:",
        disabled=not bool(source_text.strip()),
        use_container_width=True,
    )
    if create_clicked:
        model = None
        if st.session_state.interaction_model_enabled:
            from synth_platform.infrastructure.llm.ollama_chat import OllamaChatModel

            model = OllamaChatModel(
                model=st.session_state.interaction_model_name,
                host=st.session_state.interaction_model_host,
            )
        try:
            with st.spinner("Sanitizing, structuring, validating, and packaging..."):
                st.session_state.interaction_result = run_interaction_twin(
                    source_text,
                    source_name=source_name,
                    runs_dir=st.session_state.interaction_workdir / "runs",
                    seed=int(st.session_state.interaction_seed),
                    locale=st.session_state.interaction_locale,
                    model=model,
                )
            st.session_state.interaction_error = None
        except Exception as exc:  # noqa: BLE001 - UI boundary
            st.session_state.interaction_result = None
            st.session_state.interaction_error = type(exc).__name__

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
        st.code(
            result.sanitized_source_path.read_text(encoding="utf-8"), language="text"
        )

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
