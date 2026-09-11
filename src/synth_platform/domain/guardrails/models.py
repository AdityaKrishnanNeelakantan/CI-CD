"""Typed, source-free evidence for input, output, and tool guardrails."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class _GuardrailModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GuardrailStage(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    TOOL = "tool"


class GuardrailAction(StrEnum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class GuardrailFinding(_GuardrailModel):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_-]+$")
    category: str = Field(min_length=1, max_length=80)
    action: GuardrailAction
    count: int = Field(default=1, ge=1)
    message: str = Field(min_length=1, max_length=240)


class GuardrailReport(_GuardrailModel):
    policy_name: str = Field(min_length=1, max_length=80)
    policy_version: str = Field(min_length=1, max_length=32)
    stage: GuardrailStage
    action: GuardrailAction
    findings: tuple[GuardrailFinding, ...] = ()
    masked_value_count: int = Field(default=0, ge=0)
    checks: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.action == GuardrailAction.BLOCK


class GuardrailPolicy(_GuardrailModel):
    name: str = Field(min_length=1, max_length=80)
    version: str = Field(default="1.0", min_length=1, max_length=32)
    max_input_characters: int = Field(default=20_000, ge=1, le=1_000_000)
    max_output_characters: int = Field(default=20_000, ge=1, le=1_000_000)
    mask_sensitive_input: bool = True
    block_sensitive_json_output: bool = True
    block_prompt_injection: bool = True
    enforce_content_safety: bool = True
    require_json_object: bool = True
    required_input_markers: tuple[str, ...] = ()
    allowed_json_keys: tuple[str, ...] = ()


class GuardedPrompt(_GuardrailModel):
    system: str = Field(exclude=True, repr=False)
    user: str = Field(exclude=True, repr=False)
    report: GuardrailReport


class GuardedOutput(_GuardrailModel):
    text: str = Field(exclude=True, repr=False)
    report: GuardrailReport
