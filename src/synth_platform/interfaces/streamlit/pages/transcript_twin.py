"""Transcript SSOT input path for the unified synthetic-data platform."""
from __future__ import annotations

import json
import os
from importlib import util
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

from synth_platform.settings import Settings
from synth_platform.engine.transcripts import (
    build_transcript_contract,
    build_transcript_ssot_from_contract_evidence,
    validate_transcript_non_replay,
    summarize_transcript_preview,
)
from synth_platform.engine.generation.backends import GeneratorFactory, generate_transcript_twin_with_data_designer
from synth_platform.infrastructure.integrations.nvidia_nemo import inspect_nvidia_nemo_environment
from synth_platform.interfaces.streamlit.components.common.ux import platform_intro, step_guide, step_header
from synth_platform.interfaces.streamlit.ui_config import ui_step, ui_value
from synth_platform.production import is_production_profile

SETTINGS = Settings.from_env()
TRANSCRIPT_UI = ui_value("transcript", default={})
TRANSCRIPT_LABELS = dict(TRANSCRIPT_UI.get("labels") or {})
TRANSCRIPT_ARTIFACT = dict(TRANSCRIPT_UI.get("artifact") or {})
NVIDIA_ENV = inspect_nvidia_nemo_environment()
DATA_DESIGNER_READY = (
    NVIDIA_ENV.data_designer.installed
    and getattr(NVIDIA_ENV.data_designer, "python_supported", True)
) or util.find_spec("data_designer") is not None
SDK_READY = DATA_DESIGNER_READY or os.getenv("SP_NEMO_DATA_DESIGNER_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}

st.title(TRANSCRIPT_UI.get("title", ""))
platform_intro()
st.caption(TRANSCRIPT_UI.get("caption", ""))


def _ui_label(key: str, default: str = "") -> str:
    return str(TRANSCRIPT_LABELS.get(key) or default)


def _artifact_value(key: str, default: str = "") -> str:
    return str(TRANSCRIPT_ARTIFACT.get(key) or default)

defaults = {
    "transcript_text": "",
    "transcript_source_name": "",
    "transcript_preview": None,
    "transcript_contract": None,
    "transcript_synthetic": None,
    "transcript_validation": None,
    "transcript_twin_zip_bytes": None,
    "transcript_generation_error": "",
    "transcript_use_nvidia_nemo": SETTINGS.nvidia_nemo_enabled and SDK_READY,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)
if SETTINGS.nvidia_nemo_enabled and SDK_READY:
    st.session_state.transcript_use_nvidia_nemo = True


def _reset_transcript_source() -> None:
    st.session_state.transcript_text = ""
    st.session_state.transcript_source_name = ""
    st.session_state.transcript_preview = None
    st.session_state.transcript_contract = None
    st.session_state.transcript_synthetic = None
    st.session_state.transcript_validation = None
    st.session_state.transcript_twin_zip_bytes = None
    st.session_state.transcript_generation_error = ""


def _package_transcript_twin() -> bytes:
    contract = st.session_state.transcript_contract
    synthetic_twin = st.session_state.transcript_synthetic or {}
    turns = list(synthetic_twin.get("turns") or []) if isinstance(synthetic_twin, dict) else []
    ssot = dict(synthetic_twin.get("structured_ssot") or {}) if isinstance(synthetic_twin, dict) else {}
    validation = st.session_state.transcript_validation or {}
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("canonical_contract.json", json.dumps(contract.model_dump(mode="json"), indent=2))
        archive.writestr("synthetic_interaction.json", json.dumps(synthetic_twin, indent=2))
        archive.writestr("synthetic_interaction.csv", pd.DataFrame(turns).to_csv(index=False))
        archive.writestr("transcript_twin_ssot.json", json.dumps(ssot or synthetic_twin, indent=2))
        archive.writestr("validation_report.json", json.dumps(validation, indent=2))
        archive.writestr(
            "README.txt",
            f"{_artifact_value('readme_title')}\n{_artifact_value('readme_body')}\n",
        )
    return buffer.getvalue()


def _structured_ssot_from_contract() -> dict:
    contract = st.session_state.transcript_contract
    metadata = contract.entities[0].metadata if contract and contract.entities else {}
    ssot = metadata.get("structured_ssot")
    return dict(ssot) if isinstance(ssot, dict) else {}


def _contract_metadata(contract) -> dict:
    return dict(contract.entities[0].metadata) if contract and contract.entities else {}


def _default_synthetic_turns(contract) -> int:
    source_turns = int(_contract_metadata(contract).get("turn_count") or SETTINGS.transcript_default_turns)
    return max(4, min(source_turns, int(os.getenv("SP_TRANSCRIPT_UI_DEFAULT_TURNS", "12"))))


def _sample_transcript_files() -> list[Path]:
    sample_config = TRANSCRIPT_UI.get("sample_loader", {})
    if is_production_profile() or not bool(sample_config.get("enabled", False)):
        return []
    data_dir = Path(str(sample_config.get("data_dir") or ""))
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.glob("*.txt") if path.is_file())


def _build_synthetic_interaction_artifact(contract, ssot: dict, *, turn_count: int) -> dict:
    metadata = _contract_metadata(contract)
    generation_metadata = {}
    mode = _transcript_twin_mode()
    if mode == "full_sdk":
        try:
            backend = GeneratorFactory.create(SETTINGS.transcript_generation_backend)
            result = generate_transcript_twin_with_data_designer(
                contract,
                turn_count=int(turn_count),
                seed=SETTINGS.transcript_generation_seed,
            )
            turns = result.turns
            ssot = result.structured_ssot
            generation_metadata = result.metadata
            if contract and contract.entities:
                contract.entities[0].metadata["structured_ssot"] = ssot
        except Exception as exc:
            backend = GeneratorFactory.create("current")
            ssot = _ensure_structured_ssot(contract)
            turns = backend.generate_transcript(
                contract,
                turn_count=int(turn_count),
                seed=SETTINGS.transcript_generation_seed,
            )
            generation_metadata = {
                "mode": "full_sdk_fallback_to_platform_fast",
                "full_sdk_error": str(exc)[:500],
                "ssot_builder": "nemo_data_designer_local_slm",
                "turn_generation": "platform_materializer",
                "sdk_preview_calls": 1,
            }
    else:
        backend_name = "current" if mode == "fast" else SETTINGS.transcript_generation_backend
        backend = GeneratorFactory.create(backend_name if st.session_state.transcript_use_nvidia_nemo else "current")
        turns = backend.generate_transcript(
            contract,
            turn_count=int(turn_count),
            seed=SETTINGS.transcript_generation_seed,
        )
        if mode == "fast":
            generation_metadata = {
                "mode": "ssot_sdk_plus_platform_turns",
                "ssot_builder": "nemo_data_designer_local_slm" if st.session_state.transcript_use_nvidia_nemo else "disabled",
                "turn_generation": "platform_materializer",
                "sdk_preview_calls": 1 if ssot.get("status") == "generated" and st.session_state.transcript_use_nvidia_nemo else 0,
            }
    turns = _friendly_synthetic_turns(turns, metadata)
    return {
        "status": "generated",
        "artifact_type": "synthetic_customer_interaction",
        "generation": {
            "backend": backend.name,
            "turn_count": len(turns),
            "source_turn_count": metadata.get("turn_count"),
            "source_speaker_count": metadata.get("speaker_count"),
            "seed": SETTINGS.transcript_generation_seed,
            "data_designer": generation_metadata,
        },
        "summary": {
            "issue_type": (ssot.get("support_context") or {}).get("issue_type", "customer_support_request"),
            "topic_terms": (ssot.get("support_context") or {}).get("topic_terms")
            or ", ".join(metadata.get("topic_terms") or []),
            "privacy": _ui_label("source_free_privacy"),
        },
        "turns": turns,
        "structured_ssot": ssot,
    }


def _friendly_synthetic_turns(turns: list[dict], metadata: dict) -> list[dict]:
    speaker_roles = {str(key): str(value) for key, value in dict(metadata.get("speaker_roles") or {}).items()}
    friendly: list[dict] = []
    for index, row in enumerate(turns):
        next_row = dict(row)
        speaker = str(next_row.get("speaker") or "")
        lower = speaker.strip().lower()
        role = speaker_roles.get(speaker, "").lower()
        if lower.startswith("speaker_"):
            suffix = lower.rsplit("_", 1)[-1]
            if suffix == "1" or role == "customer":
                next_row["speaker"] = "Customer"
            elif suffix == "2" or role == "agent":
                next_row["speaker"] = "Agent"
            else:
                next_row["speaker"] = f"Participant {suffix}" if suffix.isdigit() else f"Participant {index + 1}"
        friendly.append(next_row)
    return friendly


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


def _label_from_key(value: str) -> str:
    return str(value or "").replace("_", " ").strip().title()


def _display_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{_label_from_key(str(key))}: {item}" for key, item in value.items())
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value or "")


def _records_from_mapping(mapping: dict, *, key_label: str, value_label: str) -> pd.DataFrame:
    rows = [
        {
            key_label: _label_from_key(str(key)),
            value_label: _display_value(value),
        }
        for key, value in (mapping or {}).items()
    ]
    return pd.DataFrame(rows)


def _records_from_list(records: list, *, fallback_label: str) -> pd.DataFrame:
    rows: list[dict] = []
    for index, record in enumerate(records or [], start=1):
        if isinstance(record, dict):
            rows.append({_label_from_key(str(key)): _display_value(value) for key, value in record.items()})
        else:
            rows.append({"#": index, fallback_label: _display_value(record)})
    return pd.DataFrame(rows)


def _section_text(mapping: dict, key: str, fallback: str = "") -> str:
    return str((mapping or {}).get(key) or fallback)


def _show_synthetic_interaction(artifact: dict) -> None:
    generation = dict(artifact.get("generation") or {})
    summary = dict(artifact.get("summary") or {})
    turns = list(artifact.get("turns") or [])
    st.markdown(f"**{_ui_label('synthetic_section')}**")
    c1, c2, c3 = st.columns(3)
    c1.metric(_ui_label("synthetic_status_metric"), _label_from_key(str(artifact.get("status") or "unknown")))
    c2.metric(_ui_label("synthetic_turns_metric"), int(generation.get("turn_count") or len(turns)))
    c3.metric(_ui_label("source_speakers_metric"), generation.get("source_speaker_count") or "unknown")
    issue_type = str(summary.get("issue_type") or "customer_support_request")
    st.caption(f"{_ui_label('issue_type_prefix')} {_label_from_key(issue_type)}")
    if summary.get("topic_terms"):
        st.caption(f"{_ui_label('grounding_topics_prefix')} {summary['topic_terms']}")

    if turns:
        st.dataframe(
            _display_transcript_rows(turns),
            hide_index=True,
            width="stretch",
            column_config={"What they said": st.column_config.TextColumn("What they said", width="large")},
        )

    with st.expander(_ui_label("structured_evidence_expander"), expanded=False):
        ssot = dict(artifact.get("structured_ssot") or {})
        support_context = dict(ssot.get("support_context") or {})
        e1, e2, e3 = st.columns(3)
        e1.metric("SSOT status", _label_from_key(str(ssot.get("status") or "unknown")))
        e2.metric("Source turns", _section_text(support_context, "turn_count", "unknown"))
        e3.metric("Source speakers", _section_text(support_context, "speaker_count", "unknown"))
        if support_context.get("issue_summary"):
            st.write(support_context["issue_summary"])
        st.json(ssot)

    with st.expander(_ui_label("technical_json_expander"), expanded=False):
        st.json(artifact)


def _show_structured_twin(ssot: dict, *, validation: dict | None = None) -> None:
    metadata = dict(ssot.get("metadata") or {})
    support_context = dict(ssot.get("support_context") or {})
    sentiment = dict(ssot.get("sentiment_analysis") or {})
    privacy = dict(ssot.get("privacy_validation") or {})

    st.markdown("**Interaction summary**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", _label_from_key(str(ssot.get("status") or "unknown")))
    m2.metric("Issue type", _label_from_key(_section_text(support_context, "issue_type", "support request")))
    m3.metric("Turns", _section_text(support_context, "turn_count", "unknown"))
    m4.metric("Speakers", _section_text(support_context, "speaker_count", "unknown"))
    summary = _section_text(support_context, "issue_summary")
    if summary:
        st.write(summary)
    topic_terms = _section_text(support_context, "topic_terms")
    if topic_terms:
        st.caption(f"Topics: {topic_terms}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Participants**")
        entities_df = _records_from_mapping(dict(ssot.get("entities") or {}), key_label="Participant", value_label="Role")
        st.dataframe(entities_df, hide_index=True, width="stretch")
    with c2:
        st.markdown("**Sentiment**")
        s1, s2 = st.columns(2)
        s1.metric("Start", _label_from_key(_section_text(sentiment, "initial_customer_sentiment", "unknown")))
        s2.metric("End", _label_from_key(_section_text(sentiment, "final_customer_sentiment", "unknown")))

    st.markdown("**Issues and outcomes**")
    issues_df = _records_from_list(list(ssot.get("resolved_issues") or []), fallback_label="Issue")
    st.dataframe(issues_df, hide_index=True, width="stretch")

    st.markdown("**Actions captured**")
    actions_df = _records_from_list(list(ssot.get("actions_taken") or []), fallback_label="Action")
    st.dataframe(actions_df, hide_index=True, width="stretch")

    account_mutations = list(ssot.get("account_mutations") or [])
    if account_mutations:
        st.markdown("**State updates**")
        st.dataframe(_records_from_list(account_mutations, fallback_label="Update"), hide_index=True, width="stretch")

    st.markdown("**Privacy checks**")
    p1, p2, p3 = st.columns(3)
    p1.metric("Raw source used", _display_value(privacy.get("raw_source_text_used")))
    p2.metric("Identifiers removed", _display_value(privacy.get("source_identifiers_removed")))
    p3.metric("Generated from sanitized turns", _display_value(privacy.get("generated_from_sanitized_turns")))
    if validation is not None:
        with st.expander("Validation details", expanded=False):
            st.json(validation)

    with st.expander(_ui_label("technical_json_expander"), expanded=False):
        st.json(
            {
                "metadata": metadata,
                "structured_twin": ssot,
            }
        )


def _validate_structured_ssot_twin(contract, ssot: dict) -> dict:
    status = ssot.get("status")
    missing = [
        key
        for key in (
            "metadata",
            "entities",
            "support_context",
            "resolved_issues",
            "actions_taken",
            "sentiment_analysis",
            "privacy_validation",
        )
        if key not in ssot
    ]
    serialized = json.dumps(ssot, sort_keys=True, default=str)
    raw_hashes_present = "source_turn_hashes" in serialized or "source_ngram_hashes" in serialized
    raw_source_text_used = False
    if contract and contract.entities:
        source_hashes = set(contract.entities[0].metadata.get("source_turn_hashes") or [])
        raw_source_text_used = any(source_hash and source_hash in serialized for source_hash in source_hashes)
    placeholder_values = [
        value
        for value in (
            "synthetic_value",
            "synthetic_resolution",
            "account_support_guidance",
            "synthetic_next_step_confirmed",
        )
        if value in serialized.lower()
    ]
    passed = (
        status == "generated"
        and not missing
        and not raw_hashes_present
        and not raw_source_text_used
        and not placeholder_values
    )
    return {
        "passed": passed,
        "status": "passed" if passed else "review",
        "structured_twin_status": status,
        "missing_required_sections": missing,
        "raw_source_text_used": raw_source_text_used,
        "raw_hashes_present": raw_hashes_present,
        "placeholder_values": placeholder_values,
    }


def _nvidia_options() -> dict[str, str | bool]:
    return {
        "enabled": bool(st.session_state.transcript_use_nvidia_nemo and SDK_READY),
        "build_structured_ssot": False,
        "curator_enabled": os.getenv("SP_TRANSCRIPT_CURATOR_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"},
        "curator_base_url": SETTINGS.nvidia_curator_base_url,
        "curator_api_key": SETTINGS.nvidia_curator_api_key,
        "curator_model": SETTINGS.nvidia_curator_model,
        "guardrails_config_path": SETTINGS.nvidia_guardrails_config_path,
        "guardrails_model": SETTINGS.nvidia_guardrails_model,
    }


def _ensure_structured_ssot(contract) -> dict:
    metadata = _contract_metadata(contract)
    ssot = dict(metadata.get("structured_ssot") or {})
    if ssot.get("status") == "generated":
        return ssot
    ssot = build_transcript_ssot_from_contract_evidence(
        contract,
        enabled=bool(st.session_state.transcript_use_nvidia_nemo and SDK_READY),
    )
    if contract and contract.entities:
        contract.entities[0].metadata["structured_ssot"] = ssot
    return ssot


def _use_single_call_transcript_twin() -> bool:
    return _transcript_twin_mode() == "full_sdk"


def _transcript_twin_mode() -> str:
    if not bool(st.session_state.transcript_use_nvidia_nemo and SDK_READY):
        return "platform"
    output_mode = os.getenv("SP_TRANSCRIPT_TWIN_OUTPUT_MODE", "ssot").strip().lower()
    if output_mode == "ssot":
        return "fast"
    raw = os.getenv("SP_TRANSCRIPT_TWIN_MODE", "").strip().lower()
    full_sdk_allowed = os.getenv("SP_TRANSCRIPT_TWIN_ALLOW_FULL_SDK", "").strip().lower() in {"1", "true", "yes", "on"}
    if raw in {"full", "full_sdk", "sdk", "single_call"} and full_sdk_allowed:
        return "full_sdk"
    if raw in {"row", "per_turn", "per-turn"}:
        return "row"
    if raw in {"platform", "current"}:
        return "platform"
    return "fast"


with st.container(border=True):
    step = ui_step("transcript", "add")
    step_header(1, step.get("title", ""), bool(st.session_state.transcript_text))
    step_guide(
        what=step.get("what", "Upload a TXT/LOG file or paste a customer conversation."),
        next_step=step.get("next", "Review the safe preview."),
    )
    input_locked = st.session_state.transcript_contract is not None
    if input_locked and st.button(_ui_label("start_new_button"), icon=":material/restart_alt:"):
        _reset_transcript_source()
        st.rerun()
    sample_files = _sample_transcript_files()
    if sample_files and not input_locked:
        sample_config = TRANSCRIPT_UI.get("sample_loader", {})
        selected_sample = st.selectbox(
            str(sample_config.get("label") or ""),
            sample_files,
            format_func=lambda path: path.name,
            index=0,
        )
        if st.button(str(sample_config.get("button_label") or ""), icon=":material/database:"):
            st.session_state.transcript_text = selected_sample.read_text(encoding="utf-8", errors="replace")
            st.session_state.transcript_source_name = selected_sample.name
            st.session_state.transcript_preview = None
            st.session_state.transcript_contract = None
            st.session_state.transcript_synthetic = None
            st.session_state.transcript_validation = None
            st.session_state.transcript_twin_zip_bytes = None
            st.session_state.transcript_generation_error = ""
            st.rerun()
    uploaded = st.file_uploader(_ui_label("interaction_file_label"), type=["txt", "log"], disabled=input_locked)
    pasted = st.text_area(
        _ui_label("paste_label"),
        height=180,
        value="" if input_locked else st.session_state.transcript_text,
        disabled=input_locked,
        help=_ui_label("locked_input_help")
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
        c4.metric(_ui_label("source_type_metric"), _ui_label("source_type"))
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
                f"{_ui_label('speakers_preview_suffix')}"
            )

with st.container(border=True):
    step = ui_step("transcript", "contract")
    step_header(3, step.get("title", "Build twin contract"), st.session_state.transcript_contract is not None)
    step_guide(
        what=step.get(
            "what",
            "Turn the interaction into a reusable twin contract. Raw interaction text is dropped after this step.",
        ),
        next_step=step.get("next", ""),
    )
    if st.button(_ui_label("build_button"), icon=":material/hub:"):
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
        c4.metric(_ui_label("source_text_kept_metric"), "no")
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
        with st.expander(_ui_label("technical_json_expander"), expanded=False):
            st.json(contract.model_dump(mode="json"))

if st.session_state.transcript_contract is None:
    st.stop()

with st.container(border=True):
    step = ui_step("transcript", "generate")
    step_header(4, step.get("title", ""), st.session_state.transcript_synthetic is not None)
    step_guide(
        what=step.get("what", ""),
        next_step=step.get("next", ""),
    )
    contract = st.session_state.transcript_contract
    ssot = _structured_ssot_from_contract()
    source_turns = int(_contract_metadata(contract).get("turn_count") or SETTINGS.transcript_default_turns)
    max_synthetic_turns = max(2, min(source_turns, 200))
    target_turns = int(
        st.number_input(
            _ui_label("synthetic_turns_input"),
            min_value=2,
            max_value=max_synthetic_turns,
            value=min(_default_synthetic_turns(contract), max_synthetic_turns),
            step=1,
        )
    )
    if st.button(_ui_label("generate_button"), icon=":material/forum:", type="primary"):
        ssot = _structured_ssot_from_contract()
        if _transcript_twin_mode() == "fast" and ssot.get("status") != "generated":
            ssot = _ensure_structured_ssot(contract)
        if _use_single_call_transcript_twin() or ssot.get("status") == "generated":
            try:
                st.session_state.transcript_synthetic = _build_synthetic_interaction_artifact(
                    contract,
                    ssot,
                    turn_count=target_turns,
                )
                st.session_state.transcript_generation_error = ""
                st.session_state.transcript_validation = None
                st.session_state.transcript_twin_zip_bytes = None
                st.rerun()
            except Exception as exc:
                st.session_state.transcript_synthetic = None
                st.session_state.transcript_validation = None
                st.session_state.transcript_twin_zip_bytes = None
                st.session_state.transcript_generation_error = str(exc)
        else:
            ssot = _ensure_structured_ssot(contract)
            if ssot.get("status") == "generated":
                st.rerun()
            else:
                st.session_state.transcript_synthetic = None
                st.session_state.transcript_validation = None
                st.session_state.transcript_twin_zip_bytes = None
                st.session_state.transcript_generation_error = (
                    f"{_ui_label('contract_not_generated')} Status: {ssot.get('status') or 'missing'}."
                )
    if st.session_state.transcript_generation_error:
        st.error(st.session_state.transcript_generation_error)
    if st.session_state.transcript_synthetic:
        _show_synthetic_interaction(st.session_state.transcript_synthetic)

if st.session_state.transcript_synthetic is None:
    st.stop()

with st.container(border=True):
    step = ui_step("transcript", "validate")
    step_header(5, step.get("title", ""), st.session_state.transcript_validation is not None)
    step_guide(
        what=step.get("what", ""),
        next_step=step.get("next", ""),
    )
    if st.button(_ui_label("validate_button"), icon=":material/verified:"):
        artifact = st.session_state.transcript_synthetic or {}
        transcript_report = validate_transcript_non_replay(
            st.session_state.transcript_contract,
            list(artifact.get("turns") or []),
            nvidia_options=_nvidia_options(),
        )
        ssot_report = _validate_structured_ssot_twin(
            st.session_state.transcript_contract,
            dict(artifact.get("structured_ssot") or {}),
        )
        st.session_state.transcript_validation = {
            "passed": bool(transcript_report.get("passed")) and bool(ssot_report.get("passed")),
            "status": "passed"
            if bool(transcript_report.get("passed")) and bool(ssot_report.get("passed"))
            else "review",
            "synthetic_turns": len(artifact.get("turns") or []),
            "raw_source_text_used": bool(transcript_report.get("raw_source_text_used")),
            "transcript_non_replay": transcript_report,
            "structured_ssot": ssot_report,
        }
        st.session_state.transcript_twin_zip_bytes = None
        st.rerun()
    if st.session_state.transcript_validation is not None:
        report = st.session_state.transcript_validation
        c1, c2, c3 = st.columns(3)
        c1.metric(_ui_label("validation_metric"), "passed" if report["passed"] else "review")
        c2.metric(_ui_label("turns_checked_metric"), report.get("synthetic_turns", 0))
        c3.metric(_ui_label("raw_source_retained_metric"), "no" if not report["raw_source_text_used"] else "yes")
        if report["passed"]:
            st.success(_ui_label("passed_message"))
        else:
            st.warning(_ui_label("review_message"))
        with st.expander(_ui_label("validation_details_expander"), expanded=False):
            st.json(report)

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
    c1.metric(_ui_label("twin_status_metric"), (st.session_state.transcript_synthetic or {}).get("status", "unknown"))
    c2.metric(_ui_label("validation_status_metric"), "passed" if (st.session_state.transcript_validation or {}).get("passed") else "review")
    c3.metric(_ui_label("raw_source_included_metric"), "no")

    validation = st.session_state.transcript_validation or {}

    with st.expander(_ui_label("zip_contents_expander"), expanded=False):
        st.write(_ui_label("zip_contents"))

    with st.expander(_ui_label("validation_evidence_expander"), expanded=False):
        st.json(validation)

    st.download_button(
        _ui_label("download_button"),
        data=st.session_state.transcript_twin_zip_bytes,
        file_name=_artifact_value("file_name"),
        mime="application/zip",
        icon=":material/download:",
        type="primary",
    )
    st.caption(_ui_label("zip_caption"))
