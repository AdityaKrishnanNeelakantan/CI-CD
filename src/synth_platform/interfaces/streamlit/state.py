"""Streamlit session state contains identifiers and paths, never credentials."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DemoState:
    db_run_id: str | None = None
    db_artifact_path: str | None = None
    db_dataset_id: str | None = None
    db_report_id: str | None = None
    pdf_run_id: str | None = None
    pdf_artifact_path: str | None = None
    pdf_dataset_id: str | None = None
    pdf_report_id: str | None = None
    pdf_package_path: str | None = None


def load_state(session_state) -> DemoState:
    raw = session_state.get("demo_state") or {}
    return DemoState(**{k: raw.get(k) for k in DemoState.__dataclass_fields__})


def save_state(session_state, state: DemoState) -> None:
    session_state["demo_state"] = asdict(state)


@dataclass(frozen=True)
class ChatMessage:
    """A renderable conversation entry stored without workflow payloads."""

    role: str
    content: str


@dataclass
class ChatState:
    """Coordinator state kept separate from specialized workflow state."""

    messages: list[ChatMessage] = field(default_factory=list)
    selected_capability: str | None = None


def load_chat_state(session_state) -> ChatState:
    raw = session_state.get("chat_state") or {}
    messages = [
        ChatMessage(role=item["role"], content=item["content"])
        for item in raw.get("messages", [])
        if (
            isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
        )
    ]
    selected = raw.get("selected_capability")
    return ChatState(
        messages=messages,
        selected_capability=selected if isinstance(selected, str) else None,
    )


def save_chat_state(session_state, state: ChatState) -> None:
    session_state["chat_state"] = {
        "messages": [asdict(message) for message in state.messages],
        "selected_capability": state.selected_capability,
    }
