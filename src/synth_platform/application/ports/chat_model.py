"""ChatModel port: a local instruction-following LLM behind one interface.

The LLM is INJECTED, never hardcoded. Any adapter (Ollama, vLLM, a test fake)
satisfies this Protocol, so the extraction front-end never knows which model it
is talking to and the deterministic core never imports an LLM at all. Swapping
Qwen3-8B for gpt-oss is a bootstrap change, not a code change.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str, *, json_only: bool = True) -> str:
        """Return the model's completion. When json_only is True the adapter
        requests strict JSON output (e.g. Ollama's format=json)."""
        ...
