"""Transcript SSOT input path for the unified synthetic-data platform."""
from __future__ import annotations

import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

from synth_platform.settings import Settings
from synth_platform.engine.transcripts import (
    build_transcript_contract,
    summarize_transcript_preview,
    validate_transcript_non_replay,
)
from synth_platform.engine.generation.backends import GeneratorFactory
from synth_platform.infrastructure.integrations.nvidia_nemo import inspect_nvidia_nemo_environment
from synth_platform.interfaces.streamlit.components.common.ux import platform_intro, step_guide, step_header
from synth_platform.interfaces.streamlit.ui_config import ui_step, ui_value

SETTINGS = Settings.from_env()
TRANSCRIPT_UI = ui_value("transcript", default={})
NVIDIA_ENV = inspect_nvidia_nemo_environment()
NVIDIA_READY = any(
    status.installed and getattr(status, "python_supported", True)
    for status in NVIDIA_ENV.packages
)

st.title(TRANSCRIPT_UI.get("title", "Customer Interactions Twin"))
platform_intro()
st.caption(
    TRANSCRIPT_UI.get(
        "caption",
        "Create synthetic customer interactions from a real conversation example.",
    )
)

defaults = {
    "transcript_text": "",
    "transcript_source_name": "",
    "transcript_preview": None,
    "transcript_contract": None,
    "transcript_synthetic": None,
    "transcript_validation": None,
    "transcript_requested_turns": None,
    "transcript_twin_zip_bytes": None,
    "transcript_generation_error": "",
    "transcript_use_nvidia_nemo": SETTINGS.nvidia_nemo_enabled and NVIDIA_READY,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def _reset_transcript_source() -> None:
    st.session_state.transcript_text = ""
    st.session_state.transcript_source_name = ""
    st.session_state.transcript_preview = None
    st.session_state.transcript_contract = None
    st.session_state.transcript_synthetic = None
    st.session_state.transcript_validation = None
    st.session_state.transcript_requested_turns = None
    st.session_state.transcript_twin_zip_bytes = None
    st.session_state.transcript_generation_error = ""


def _package_transcript_twin() -> bytes:
    contract = st.session_state.transcript_contract
    synthetic = st.session_state.transcript_synthetic or []
    validation = st.session_state.transcript_validation or {}
    synthetic_df = pd.DataFrame(synthetic)
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("canonical_contract.json", json.dumps(contract.model_dump(mode="json"), indent=2))
        archive.writestr("synthetic_transcript.json", json.dumps(synthetic, indent=2))
        archive.writestr("synthetic_transcript.csv", synthetic_df.to_csv(index=False))
        archive.writestr("validation_report.json", json.dumps(validation, indent=2))
        archive.writestr(
            "README.txt",
            "Customer Interactions Twin artifact\n"
            "Contains the twin contract, source-free synthetic customer interactions, "
            "and validation evidence. Raw source interaction text is not included.\n",
        )
    return buffer.getvalue()


def _display_transcript_rows(rows: list[dict]) -> pd.DataFrame:
    display_rows = []
    for row in rows:
        speaker = str(row.get("speaker") or "")
        speaker = speaker.replace("speaker_1", "Customer").replace("speaker_2", "Agent")
        speaker = speaker.replace("speaker_", "Person ")
        display_rows.append(
            {
                "Message #": row.get("turn"),
                "Who": speaker,
                "What they said": row.get("text") or "",
            }
        )
    return pd.DataFrame(display_rows)


def _display_preview_rows(rows: list[dict]) -> pd.DataFrame:
    return _display_transcript_rows(
        [
            {
                "turn": row.get("turn"),
                "speaker": row.get("speaker"),
                "text": row.get("text"),
            }
            for row in rows
        ]
    )


def _validation_evidence(report: dict) -> dict:
    max_self_similarity = report.get("max_self_similarity")
    if max_self_similarity is None:
        max_self_similarity = report.get("max_synthetic_similarity")
    warning_threshold = report.get("variety_warning_threshold")
    if warning_threshold is None:
        warning_threshold = report.get("similarity_threshold")
    return {
        "passed": report.get("passed"),
        "source_replay": {
            "exact_replay_count": report.get("exact_replay_count"),
            "ngram_replay_count": report.get("ngram_replay_count"),
            "raw_source_text_used": report.get("raw_source_text_used"),
        },
        "synthetic_variety": {
            "max_self_similarity": max_self_similarity,
            "warning_threshold": warning_threshold,
            "repeated_phrase_warning": report.get("repeated_phrase_warning"),
        },
        "nvidia_nemo_guardrails": report.get("nvidia_nemo_guardrails"),
    }


def _nvidia_options() -> dict[str, str | bool]:
    return {
        "enabled": bool(st.session_state.transcript_use_nvidia_nemo and NVIDIA_READY),
        "curator_base_url": SETTINGS.nvidia_curator_base_url,
        "curator_api_key": SETTINGS.nvidia_curator_api_key,
        "curator_model": SETTINGS.nvidia_curator_model,
        "guardrails_config_path": SETTINGS.nvidia_guardrails_config_path,
    }


with st.container(border=True):
    step = ui_step("transcript", "add")
    step_header(1, step.get("title", "Add customer interaction"), bool(st.session_state.transcript_text))
    step_guide(
        what=step.get("what", "Upload a TXT/LOG file or paste a customer conversation."),
        next_step=step.get("next", "Review the safe preview."),
    )
    input_locked = st.session_state.transcript_contract is not None
    if input_locked and st.button("Start new interaction", icon=":material/restart_alt:"):
        _reset_transcript_source()
        st.rerun()
    uploaded = st.file_uploader("Interaction file", type=["txt", "log"], disabled=input_locked)
    pasted = st.text_area(
        "Or paste the interaction",
        height=180,
        value="" if input_locked else st.session_state.transcript_text,
        disabled=input_locked,
        help="Start a new interaction to replace this source."
        if input_locked
        else None,
    )
    if uploaded is not None and not input_locked:
        st.session_state.transcript_text = uploaded.getvalue().decode("utf-8", errors="replace")
        st.session_state.transcript_source_name = uploaded.name
        st.session_state.transcript_preview = None
        st.session_state.transcript_contract = None
        st.session_state.transcript_synthetic = None
        st.session_state.transcript_validation = None
        st.session_state.transcript_twin_zip_bytes = None
        st.session_state.transcript_generation_error = ""
    elif (pasted or "").strip() and pasted != st.session_state.transcript_text and not input_locked:
        st.session_state.transcript_text = pasted
        st.session_state.transcript_source_name = "pasted_transcript"
        st.session_state.transcript_preview = None
        st.session_state.transcript_contract = None
        st.session_state.transcript_synthetic = None
        st.session_state.transcript_validation = None
        st.session_state.transcript_twin_zip_bytes = None
        st.session_state.transcript_generation_error = ""

if not st.session_state.transcript_text and st.session_state.transcript_contract is None:
    st.stop()

with st.container(border=True):
    step = ui_step("transcript", "preview")
    step_header(2, step.get("title", "Preview"), st.session_state.transcript_preview is not None)
    if st.session_state.transcript_preview is None and st.session_state.transcript_text:
        st.session_state.transcript_preview = summarize_transcript_preview(
            st.session_state.transcript_text,
            source_name=st.session_state.transcript_source_name,
        )
    preview = st.session_state.transcript_preview
    if preview is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Turns", preview.record_count or 0)
        c2.metric("Speakers", preview.entity_count or 0)
        c3.metric("Fields", preview.field_count or 0)
        c4.metric("Source type", "customer interaction")
        if preview.items:
            item = preview.items[0]
            if item.sample:
                st.dataframe(
                    _display_preview_rows(item.sample),
                    hide_index=True,
                    width="stretch",
                    column_config={"What they said": st.column_config.TextColumn("What they said", width="large")},
                )
            speakers = item.metadata.get("speakers") or []
            st.caption(
                f"Speakers found: {len(speakers)}. "
                "Direct identifiers and common sensitive references are hidden in the preview."
            )

with st.container(border=True):
    step = ui_step("transcript", "contract")
    step_header(3, step.get("title", "Build twin contract"), st.session_state.transcript_contract is not None)
    step_guide(
        what=step.get(
            "what",
            "Turn the interaction into a reusable twin contract. Raw interaction text is dropped after this step.",
        ),
        next_step=step.get("next", "Create synthetic customer interactions."),
    )
    if st.button("Build twin contract", icon=":material/hub:"):
        st.session_state.transcript_contract = build_transcript_contract(
            st.session_state.transcript_text,
            source_name=st.session_state.transcript_source_name,
            nvidia_options=_nvidia_options(),
        )
        # Privacy boundary: after contract build, keep only sanitized preview
        # and hashed non-replay evidence inside the contract metadata.
        st.session_state.transcript_text = ""
        st.session_state.transcript_synthetic = None
        st.session_state.transcript_validation = None
        st.session_state.transcript_twin_zip_bytes = None
        st.rerun()
    if st.session_state.transcript_contract is not None:
        contract = st.session_state.transcript_contract
        entity_meta = contract.entities[0].metadata if contract.entities else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Twin type", contract.source_type)
        c2.metric("Fields", len(contract.fields))
        c3.metric("Speakers", entity_meta.get("speaker_count", 0))
        c4.metric("Source text kept", "no")
        with st.expander("Contract fields", expanded=False):
            st.dataframe(
                [
                    {
                        "field": field.name,
                        "type": field.semantic_type,
                        "nullable": field.nullable,
                    }
                    for field in contract.fields
                ],
                hide_index=True,
                width="stretch",
            )
        with st.expander("Technical JSON", expanded=False):
            st.json(contract.model_dump(mode="json"))

if st.session_state.transcript_contract is None:
    st.stop()

with st.container(border=True):
    step = ui_step("transcript", "generate")
    step_header(4, step.get("title", "Create customer conversation"), st.session_state.transcript_synthetic is not None)
    step_guide(
        what=step.get("what", "Choose how many messages you want in the new conversation."),
        next_step=step.get("next", "Check the conversation before downloading."),
    )
    contract = st.session_state.transcript_contract
    entity_meta = contract.entities[0].metadata if contract.entities else {}
    default_turns = int(entity_meta.get("turn_count") or SETTINGS.transcript_default_turns)
    if st.session_state.transcript_requested_turns is None:
        st.session_state.transcript_requested_turns = max(1, default_turns)
    requested_turns = st.number_input(
        "Number of messages",
        min_value=1,
        value=int(st.session_state.transcript_requested_turns),
        step=1,
        help="This can be different from the original conversation.",
    )
    st.session_state.transcript_requested_turns = int(requested_turns)
    if st.button("Create conversation", icon=":material/auto_fix_high:", type="primary"):
        try:
            st.session_state.transcript_synthetic = GeneratorFactory.create(
                SETTINGS.transcript_generation_backend
            ).generate_transcript(
                contract,
                turn_count=int(requested_turns),
                seed=SETTINGS.transcript_generation_seed,
            )
            st.session_state.transcript_generation_error = ""
            st.session_state.transcript_validation = None
            st.session_state.transcript_twin_zip_bytes = None
            st.rerun()
        except RuntimeError as exc:
            st.session_state.transcript_synthetic = None
            st.session_state.transcript_validation = None
            st.session_state.transcript_twin_zip_bytes = None
            st.session_state.transcript_generation_error = str(exc)
    if st.session_state.transcript_generation_error:
        st.error(st.session_state.transcript_generation_error)
    if st.session_state.transcript_synthetic:
        st.dataframe(
            _display_transcript_rows(st.session_state.transcript_synthetic),
            hide_index=True,
            width="stretch",
            column_config={"What they said": st.column_config.TextColumn("What they said", width="large")},
        )

if st.session_state.transcript_synthetic is None:
    st.stop()

with st.container(border=True):
    step = ui_step("transcript", "validate")
    step_header(5, step.get("title", "Validate interaction"), st.session_state.transcript_validation is not None)
    step_guide(
        what=step.get("what", "Check that the synthetic interaction does not replay the source."),
        next_step=step.get("next", "Download the finished twin."),
    )
    if st.button("Validate interaction", icon=":material/verified:"):
        st.session_state.transcript_validation = validate_transcript_non_replay(
            st.session_state.transcript_contract,
            st.session_state.transcript_synthetic,
            nvidia_options=_nvidia_options(),
        )
        st.session_state.transcript_twin_zip_bytes = None
        st.rerun()
    if st.session_state.transcript_validation is not None:
        report = st.session_state.transcript_validation
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Replay check", "passed" if report["passed"] else "review")
        c2.metric("Exact replays", report["exact_replay_count"])
        c3.metric("N-gram replays", report["ngram_replay_count"])
        c4.metric("Raw source retained", "no" if not report["raw_source_text_used"] else "yes")
        c5.metric("Variety warning", "yes" if report.get("repeated_phrase_warning") else "no")
        if report["passed"]:
            st.success("Passed. No source replay or raw interaction retention was detected.")
            if report.get("repeated_phrase_warning"):
                st.info("Some generated turns are similar to each other. This is a variety warning, not source replay.")
        else:
            st.warning("Needs review. Check replay counts before downloading.")
        with st.expander("Validation details", expanded=False):
            st.json(_validation_evidence(report))

if st.session_state.transcript_validation is None:
    st.stop()

with st.container(border=True):
    step = ui_step("transcript", "download")
    step_header(6, step.get("title", "Download twin"), st.session_state.transcript_twin_zip_bytes is not None)
    step_guide(
        what=step.get("what", "Review the generated twin, then download it as a ZIP."),
        next_step=step.get("next", "Done."),
    )
    if st.session_state.transcript_twin_zip_bytes is None:
        st.session_state.transcript_twin_zip_bytes = _package_transcript_twin()
    c1, c2, c3 = st.columns(3)
    c1.metric("Messages", len(st.session_state.transcript_synthetic or []))
    c2.metric("Validation", "passed" if (st.session_state.transcript_validation or {}).get("passed") else "review")
    c3.metric("Raw source included", "no")

    st.markdown("**Generated twin**")
    contract = st.session_state.transcript_contract
    synthetic = st.session_state.transcript_synthetic or []
    validation = st.session_state.transcript_validation or {}
    st.dataframe(
        _display_transcript_rows(synthetic).head(20),
        hide_index=True,
        width="stretch",
        column_config={"What they said": st.column_config.TextColumn("What they said", width="large")},
    )

    with st.expander("What's included in the ZIP", expanded=False):
        st.write("Synthetic interaction CSV/JSON, twin contract, validation report, and README.")

    with st.expander("Validation evidence", expanded=False):
        st.json(_validation_evidence(validation))

    st.download_button(
        "Download customer interactions twin (ZIP)",
        data=st.session_state.transcript_twin_zip_bytes,
        file_name="customer_interactions_twin_artifact.zip",
        mime="application/zip",
        icon=":material/download:",
        type="primary",
    )
    st.caption(
        "The ZIP does not include the raw source interaction."
    )
