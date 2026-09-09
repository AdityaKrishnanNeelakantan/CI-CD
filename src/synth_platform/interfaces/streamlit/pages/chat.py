"""Chat-first entry point for routing into specialized twin workflows."""

from __future__ import annotations

import streamlit as st

from synth_platform.application.coordinator import (
    CAPABILITIES,
    CapabilityId,
    RouteDecision,
    RouteRequest,
    get_capability,
    route_request,
)
from synth_platform.interfaces.streamlit.capabilities import capability_page_path
from synth_platform.interfaces.streamlit.state import (
    ChatMessage,
    ChatState,
    load_chat_state,
    save_chat_state,
)

_WELCOME = (
    "Describe what you want to create or attach a schema, SQLite database, or PDF. "
    "I route the request; each specialized workflow still owns generation, validation, "
    "and artifacts."
)


def _apply_decision(
    state: ChatState, decision: RouteDecision, user_content: str
) -> None:
    state.messages.append(ChatMessage(role="user", content=user_content))
    state.messages.append(ChatMessage(role="assistant", content=decision.user_message))
    state.selected_capability = (
        decision.capability_id.value
        if decision.can_invoke and decision.capability_id
        else None
    )


def _route_explicit(state: ChatState, capability_id: CapabilityId) -> None:
    capability = get_capability(capability_id)
    decision = route_request(RouteRequest(explicit_capability=capability_id))
    _apply_decision(state, decision, f"Use {capability.display_name}.")
    save_chat_state(st.session_state, state)
    st.rerun()


def _render_capability_choices(state: ChatState) -> None:
    st.markdown("### What can I create?")
    columns = st.columns(len(CAPABILITIES))
    for column, capability in zip(columns, CAPABILITIES, strict=True):
        with column:
            status = (
                "Available" if capability.status.value == "available" else "Planned"
            )
            st.markdown(f"**{capability.display_name}**")
            st.caption(f"{capability.description}  \n{status} · `{capability.command}`")
            if st.button(
                f"Choose {capability.display_name}",
                key=f"chat_choose_{capability.capability_id.value}",
                use_container_width=True,
            ):
                _route_explicit(state, capability.capability_id)


def _render_selected_capability(state: ChatState) -> None:
    if not state.selected_capability:
        return
    try:
        capability_id = CapabilityId(state.selected_capability)
    except ValueError:
        state.selected_capability = None
        save_chat_state(st.session_state, state)
        return

    capability = get_capability(capability_id)
    page_path = capability_page_path(capability_id)
    if page_path is None:
        return

    with st.container(border=True):
        st.markdown(f"### Ready for {capability.display_name}")
        st.write(
            "Continue to the existing workflow. Its review gates, deterministic or statistical "
            "generation, validation, and artifact handling remain unchanged."
        )
        st.caption(
            "Phase 1 uses a safe page hand-off. Uploaded files are not copied across workflow "
            "boundaries yet, so provide the source again in the selected workflow."
        )
        st.page_link(
            page_path,
            label=f"Continue in {capability.display_name}",
            icon=":material/arrow_forward:",
            use_container_width=True,
        )
        if st.button("Choose a different capability", key="chat_clear_capability"):
            state.selected_capability = None
            save_chat_state(st.session_state, state)
            st.rerun()


def run() -> None:
    st.title("Synthetic Data Twin")
    st.caption(
        "One conversational entry point. Specialized workflows. Deterministic routing."
    )

    state = load_chat_state(st.session_state)
    if not state.messages:
        state.messages.append(ChatMessage(role="assistant", content=_WELCOME))
        save_chat_state(st.session_state, state)

    _render_capability_choices(state)
    st.divider()

    for message in state.messages:
        with st.chat_message(message.role):
            st.write(message.content)

    uploaded_files = st.file_uploader(
        "Attach source files for routing",
        accept_multiple_files=True,
        help=(
            "Schema: SQL/JSON/YAML · Database: SQLite · Document: PDF · "
            "Interaction: TXT/LOG (planned)"
        ),
    )
    if uploaded_files and st.button(
        "Route attached files",
        key="chat_route_attachments",
        type="primary",
        use_container_width=True,
    ):
        names = tuple(item.name for item in uploaded_files)
        decision = route_request(RouteRequest(attachment_names=names))
        _apply_decision(
            state,
            decision,
            f"Route attachments: {', '.join(names)}",
        )
        save_chat_state(st.session_state, state)
        st.rerun()

    prompt = st.chat_input(
        "Describe what you want, or use /schema, /database, /document, or /interaction"
    )
    if prompt:
        names = tuple(item.name for item in uploaded_files) if uploaded_files else ()
        user_content = prompt
        if names:
            user_content = f"{prompt}\n\nAttachments: {', '.join(names)}"
        decision = route_request(RouteRequest(message=prompt, attachment_names=names))
        _apply_decision(state, decision, user_content)
        save_chat_state(st.session_state, state)
        st.rerun()

    _render_selected_capability(state)


run()
