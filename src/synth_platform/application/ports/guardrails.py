"""Ports for deterministic model guardrails."""

from __future__ import annotations

from typing import Protocol

from synth_platform.domain.guardrails.models import (
    GuardedOutput,
    GuardedPrompt,
    GuardrailPolicy,
)


class ModelGuardrails(Protocol):
    def inspect_input(
        self,
        system: str,
        user: str,
        policy: GuardrailPolicy,
    ) -> GuardedPrompt: ...

    def inspect_output(
        self,
        output: str,
        policy: GuardrailPolicy,
        *,
        json_only: bool,
    ) -> GuardedOutput: ...
