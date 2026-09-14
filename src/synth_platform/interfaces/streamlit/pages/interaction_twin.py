"""Customer Interaction Twin Streamlit page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from synth_platform.application.services.transfer_service import TransferService
from synth_platform.application.workflows.interaction_twin import parse_transcript, run_interaction_twin
from synth_platform.errors import TransferBlockedError
from synth_platform.domain.product_settings import (
    privacy_level_removes_sensitive_information,
    read_generation_defaults,
)
from synth_platform.infrastructure.persistence.platform_db import get_platform_db
from synth_platform.interfaces.streamlit.components.common.ux import (
    measure_generation,
    mode_outcomes,
    platform_intro,
    render_progress,
    render_run_metrics,
    show_user_error,
    step_guide,
    step_header,
)

st.title("Customer Interaction Twin")
platform_intro()
st.caption("Create a synthetic customer-service transcript from an uploaded .txt or .log conversation.")

defaults = {
    "interaction_file_id": None,
    "interaction_source_text": None,
    "interaction_source_name": None,
    "interaction_preview": None,
    "interaction_result": None,
    "interaction_workdir": None,
    "interaction_run_metrics": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

if st.session_state.interaction_workdir is None:
    st.session_state.interaction_workdir = Path(tempfile.mkdtemp(prefix="interaction_twin_"))


def _reset_generation() -> None:
    st.session_state.interaction_result = None
    st.session_state.interaction_run_metrics = None


render_progress(
    [
        ("Upload", st.session_state.interaction_source_text is not None),
        ("Configure", st.session_state.interaction_source_text is not None),
        ("Generate", st.session_state.interaction_result is not None),
        ("Results", st.session_state.interaction_result is not None),
    ]
)
product_defaults = read_generation_defaults(get_platform_db())

with st.container(border=True):
    step_header(1, "Upload", st.session_state.interaction_source_text is not None)
    step_guide(what="Upload a speaker-labeled transcript.", next_step="Choose interaction settings.")
    uploaded = st.file_uploader("Transcript", type=["txt", "log"])
    if uploaded is not None:
        file_id = f"{uploaded.name}:{uploaded.size}"
        if file_id != st.session_state.interaction_file_id:
            try:
                text = uploaded.getvalue().decode("utf-8-sig")
                parsed = parse_transcript(text)
                st.session_state.interaction_source_text = text
                st.session_state.interaction_source_name = uploaded.name
                st.session_state.interaction_preview = parsed
                st.session_state.interaction_file_id = file_id
                _reset_generation()
            except Exception as exc:
                st.session_state.interaction_source_text = None
                st.session_state.interaction_preview = None
                st.session_state.interaction_file_id = None
                _reset_generation()
                show_user_error(
                    "We couldn't read that transcript.",
                    technical=exc,
                    next_action="Upload a .txt or .log transcript with speaker-labeled turns.",
                )
                st.stop()
        st.caption(f"Loaded `{uploaded.name}`.")

if st.session_state.interaction_source_text is None:
    st.warning("Upload a transcript to continue.")
    st.stop()

preview = st.session_state.interaction_preview or parse_transcript(st.session_state.interaction_source_text)

with st.container(border=True):
    step_header(2, "Configure", True)
    step_guide(what="Set the transcript domain and export shape.", next_step="Generate the synthetic interaction.")
    c1, c2, c3 = st.columns(3)
    with c1:
        interaction_type = st.selectbox(
            "Interaction Type",
            options=["Customer Support", "Billing Support", "Account Support", "Benefit Activation"],
            index=0,
        )
    with c2:
        output_format = st.selectbox(
            "Output Format",
            options=["Structured JSON + Synthetic Logs"],
            index=0,
        )
    with c3:
        seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42)
    remove_sensitive = st.toggle(
        "Remove sensitive information",
        value=privacy_level_removes_sensitive_information(str(product_defaults["privacy_level"])),
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Turns", len(preview.turns))
    m2.metric("Speakers", len(preview.speaker_counts))
    m3.metric("Source", st.session_state.interaction_source_name or "Transcript")
    st.dataframe([turn.to_dict() for turn in preview.turns[:10]], hide_index=True, width="stretch")

with st.container(border=True):
    step_header(3, "Generate", st.session_state.interaction_result is not None)
    step_guide(what="Create a synthetic conversation and validation report.", next_step="Review and download.")
    if st.button("Generate Interaction Twin", type="primary", width="stretch"):
        with st.status("Generating synthetic interaction...", expanded=True) as status:
            try:
                st.write("Parsing transcript...")
                st.write("Applying privacy settings...")
                st.write("Synthesizing structured turns...")
                with measure_generation() as run_stats:
                    result = run_interaction_twin(
                        st.session_state.interaction_source_text,
                        interaction_type=interaction_type,
                        output_format=output_format,
                        remove_sensitive_information=bool(remove_sensitive),
                        seed=int(seed),
                        output_dir=Path(st.session_state.interaction_workdir) / "output",
                        project_name=Path(st.session_state.interaction_source_name or "Interaction Twin").stem,
                        history=get_platform_db(),
                        product_settings=get_platform_db(),
                    )
                st.session_state.interaction_result = result
                st.session_state.interaction_run_metrics = run_stats
                status.update(label="Synthetic interaction generated", state="complete", expanded=False)
            except Exception as exc:
                st.session_state.interaction_result = None
                st.session_state.interaction_run_metrics = None
                status.update(label="Generation failed", state="error")
                show_user_error(
                    "The synthetic interaction could not be generated.",
                    technical=exc,
                    next_action="Check the transcript structure and try again.",
                )

result = st.session_state.interaction_result
if result is None:
    st.stop()

with st.container(border=True):
    step_header(4, "Results", True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Synthetic turns", len(result.synthetic.turns))
    c2.metric("Validation", result.validation_report.get("status", "UNKNOWN"))
    c3.metric("Redactions", result.redaction_report.get("finding_count", 0))
    metrics = st.session_state.interaction_run_metrics or {}
    render_run_metrics(
        seconds=metrics.get("seconds"),
        peak_memory_mb=metrics.get("peak_memory_mb"),
        rows_per_second=None,
        extra={"scope": "interaction generation", "note": "Time and memory for this Interaction Twin run."},
    )
    st.text_area("Synthetic log preview", value="\n".join(f"{t.speaker}: {t.text}" for t in result.synthetic.turns[:20]), height=260)
    with st.expander("Structured JSON"):
        st.json(result.synthetic.to_dict())
    with st.expander("Validation report"):
        st.json(result.validation_report)

    filename = "interaction_twin.zip"
    try:
        package_bytes = result.package_bytes
        if package_bytes is None:
            package_bytes = b""
        transfer = TransferService(transfer_recorder=get_platform_db()).downloadable_bytes(
            workflow="interaction_twin",
            output_id=filename,
            validation_report=result.validation_report,
            data=package_bytes,
            metadata={
                "file_name": filename,
                "source_file": st.session_state.interaction_source_name,
                "interaction_type": interaction_type,
            },
        )
        st.download_button(
            "Download interaction twin (ZIP)",
            data=transfer.data,
            file_name=filename,
            mime="application/zip",
            width="stretch",
        )
    except TransferBlockedError as exc:
        st.download_button(
            "Download interaction twin (ZIP)",
            data=b"",
            file_name=filename,
            mime="application/zip",
            width="stretch",
            disabled=True,
        )
        st.warning(str(exc))

    mode_outcomes(
        [
            "Structured synthetic turns",
            "Synthetic log export",
            "Validation and redaction reports",
        ]
    )
