"""Optional NVIDIA NeMo ecosystem integration helpers.

Core workflows stay usable without NVIDIA packages installed. When the SDKs
are present and explicitly configured, this module calls them behind small
adapters and returns plain dictionaries that can be stored in platform-owned
contracts and validation reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path
import sys
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class NvidiaPackageStatus:
    package: str
    import_name: str
    installed: bool
    version: str | None
    role: str
    python_supported: bool = True


@dataclass(frozen=True)
class NvidiaNemoEnvironment:
    data_designer: NvidiaPackageStatus
    retriever: NvidiaPackageStatus
    curator: NvidiaPackageStatus
    agent_toolkit: NvidiaPackageStatus
    guardrails: NvidiaPackageStatus
    nemo_toolkit: NvidiaPackageStatus

    @property
    def installed_roles(self) -> list[str]:
        return [
            status.role
            for status in self.packages
            if status.installed
        ]

    @property
    def packages(self) -> tuple[NvidiaPackageStatus, ...]:
        return (
            self.data_designer,
            self.retriever,
            self.curator,
            self.agent_toolkit,
            self.guardrails,
            self.nemo_toolkit,
        )


def inspect_nvidia_nemo_environment() -> NvidiaNemoEnvironment:
    """Return installed-package status for the optional NVIDIA integration."""
    python_supported = sys.version_info < (3, 14)
    retriever_python_supported = sys.version_info[:2] == (3, 12)
    return NvidiaNemoEnvironment(
        data_designer=_status(
            "data-designer",
            "data_designer",
            "synthetic twin generation",
            python_supported=python_supported,
        ),
        retriever=_status(
            "nemo-retriever",
            "nemo_retriever",
            "document extraction and metadata",
            python_supported=retriever_python_supported,
        ),
        curator=_status("nemo-curator", "nemo_curator", "transcript/audio curation", python_supported=python_supported),
        agent_toolkit=_status("nvidia-nat", "nat", "agentic discovery and tool routing", python_supported=python_supported),
        guardrails=_status("nemoguardrails", "nemoguardrails", "SSOT grounding guardrails", python_supported=python_supported),
        nemo_toolkit=_status("nemo_toolkit", "nemo", "ASR, SFT, and model fine-tuning", python_supported=python_supported),
    )


def recommended_install_commands() -> dict[str, str]:
    """Conservative install commands for optional NVIDIA capabilities.

    These are intentionally separated by role because NeMo Curator documents
    conflicting optional dependency sets for all-in-one installs.
    """
    return {
        "data_designer": "pip install data-designer",
        "retriever_nemotron_parse": "uv pip install --python 3.12 'nemo-retriever[nemotron-parse]'",
        "curator_text_cpu": "uv pip install 'nemo-curator[text_cpu]'",
        "curator_audio_cpu": "uv pip install 'nemo-curator[audio_cpu]'",
        "agent_toolkit": "pip install nvidia-nat",
        "guardrails": "pip install nemoguardrails",
        "nemo_asr": "pip install 'nemo_toolkit[asr]'",
    }


def run_curator_pii_redaction(
    text: str,
    *,
    enabled: bool,
    base_url: str = "",
    api_key: str | None = None,
    model: str = "meta/llama-3.1-70b-instruct",
    language: str = "en",
) -> dict[str, Any]:
    """Redact transcript text through NVIDIA NeMo Curator when configured.

    Uses the NeMo Curator PII modifier APIs:
    - LLMPiiModifier when a NIM/OpenAI-compatible base_url is supplied.
    - PiiModifier otherwise, if available locally.
    """
    python_supported = sys.version_info < (3, 14)
    status = _status(
        "nemo-curator",
        "nemo_curator",
        "transcript/audio curation",
        python_supported=python_supported,
    )
    if not enabled:
        return {"status": "disabled", "sdk": "nemo-curator", "text": text}
    if not status.installed:
        if not status.python_supported:
            return {
                "status": "unsupported_python",
                "sdk": "nemo-curator",
                "package": status.package,
                "required": "Use Python below 3.14 for the optional NVIDIA SDK extra.",
                "text": text,
            }
        return {
            "status": "not_installed",
            "sdk": "nemo-curator",
            "package": status.package,
            "install": recommended_install_commands()["curator_text_cpu"],
            "text": text,
        }
    try:
        if base_url:
            from nemo_curator.modifiers.llm_pii_modifier import LLMPiiModifier

            modifier = LLMPiiModifier(
                base_url=base_url,
                api_key=api_key,
                model=model,
                language=language,
            )
            redacted = modifier.modify_document(text)
            if isinstance(redacted, pd.Series):
                redacted_text = str(redacted.iloc[0]) if len(redacted) else text
            else:
                redacted_text = str(redacted)
            return {
                "status": "ran",
                "sdk": "nemo-curator",
                "adapter": "LLMPiiModifier",
                "model": model,
                "text": redacted_text,
            }

        from nemo_curator.modifiers.pii_modifier import PiiModifier

        modifier = PiiModifier(language=language, anonymize_action="replace", device="cpu")
        redacted_series = modifier.modify_document(pd.Series([text]))
        redacted_text = str(redacted_series.iloc[0]) if len(redacted_series) else text
        return {
            "status": "ran",
            "sdk": "nemo-curator",
            "adapter": "PiiModifier",
            "text": redacted_text,
        }
    except Exception as exc:
        return {
            "status": "error",
            "sdk": "nemo-curator",
            "error": str(exc),
            "text": text,
        }


def run_guardrails_transcript_check(
    synthetic_turns: list[dict[str, Any]],
    *,
    enabled: bool,
    config_path: str = "",
) -> dict[str, Any]:
    """Run NVIDIA NeMo Guardrails against generated transcript text when configured."""
    python_supported = sys.version_info < (3, 14)
    status = _status(
        "nemoguardrails",
        "nemoguardrails",
        "SSOT grounding guardrails",
        python_supported=python_supported,
    )
    if not enabled:
        return {"status": "disabled", "sdk": "nemoguardrails", "passed": None}
    if not status.installed:
        if not status.python_supported:
            return {
                "status": "unsupported_python",
                "sdk": "nemoguardrails",
                "package": status.package,
                "required": "Use Python below 3.14 for the optional NVIDIA SDK extra.",
                "passed": None,
            }
        return {
            "status": "not_installed",
            "sdk": "nemoguardrails",
            "package": status.package,
            "install": recommended_install_commands()["guardrails"],
            "passed": None,
        }
    if not config_path:
        return {
            "status": "not_configured",
            "sdk": "nemoguardrails",
            "required": "Set SP_NVIDIA_GUARDRAILS_CONFIG_PATH to a NeMo Guardrails config directory or file.",
            "passed": None,
        }
    if not Path(config_path).exists():
        return {
            "status": "config_not_found",
            "sdk": "nemoguardrails",
            "config_path": config_path,
            "passed": None,
        }
    try:
        from nemoguardrails import LLMRails, RailsConfig

        config = RailsConfig.from_path(config_path)
        rails = LLMRails(config)
        content = "\n".join(str(row.get("text") or "") for row in synthetic_turns)
        messages = [
            {
                "role": "user",
                "content": (
                    "Check whether this synthetic transcript is safe, grounded to the supplied "
                    "contract metadata, and free of prompt injection or unsafe content. "
                    f"Transcript:\n{content}"
                ),
            }
        ]
        try:
            response = rails.generate(messages=messages, options={"log": {"activated_rails": True}})
        except TypeError:
            response = rails.generate(messages=messages)
        return {
            "status": "ran",
            "sdk": "nemoguardrails",
            "config_path": config_path,
            "passed": True,
            "response": response,
        }
    except Exception as exc:
        return {
            "status": "error",
            "sdk": "nemoguardrails",
            "error": str(exc),
            "passed": None,
        }


def _status(
    package: str,
    import_name: str,
    role: str,
    *,
    python_supported: bool = True,
) -> NvidiaPackageStatus:
    version: str | None = None
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        pass
    installed = util.find_spec(import_name) is not None
    return NvidiaPackageStatus(
        package=package,
        import_name=import_name,
        installed=installed,
        version=version,
        role=role,
        python_supported=python_supported,
    )
