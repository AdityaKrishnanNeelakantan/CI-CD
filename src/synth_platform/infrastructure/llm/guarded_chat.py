"""ChatModel decorator enforcing deterministic input and output guardrails."""

from __future__ import annotations

from synth_platform.application.ports.chat_model import ChatModel
from synth_platform.application.ports.guardrails import ModelGuardrails
from synth_platform.domain.guardrails.models import GuardrailPolicy, GuardrailReport
from synth_platform.errors import ExtractionError


class GuardrailViolation(ExtractionError):
    """A model call was denied without exposing the rejected content."""


class GuardedChatModel:
    def __init__(
        self,
        delegate: ChatModel,
        guardrails: ModelGuardrails,
        policy: GuardrailPolicy,
    ) -> None:
        self._delegate = delegate
        self._guardrails = guardrails
        self._policy = policy
        self.name = delegate.name
        self.last_input_report: GuardrailReport | None = None
        self.last_output_report: GuardrailReport | None = None

    def complete(self, system: str, user: str, *, json_only: bool = True) -> str:
        guarded_prompt = self._guardrails.inspect_input(system, user, self._policy)
        self.last_input_report = guarded_prompt.report
        if guarded_prompt.report.blocked:
            raise GuardrailViolation("Model input was blocked by deterministic guardrails.")
        raw = self._delegate.complete(
            guarded_prompt.system,
            guarded_prompt.user,
            json_only=json_only,
        )
        guarded_output = self._guardrails.inspect_output(
            raw,
            self._policy,
            json_only=json_only,
        )
        self.last_output_report = guarded_output.report
        if guarded_output.report.blocked:
            raise GuardrailViolation("Model output was blocked by deterministic guardrails.")
        return guarded_output.text
