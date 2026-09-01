from __future__ import annotations

from synth_platform.infrastructure.integrations.nvidia_nemo import (
    inspect_nvidia_nemo_environment,
    recommended_install_commands,
    run_curator_pii_redaction,
    run_guardrails_transcript_check,
)


def test_nvidia_nemo_probe_is_lazy_and_reports_known_packages():
    env = inspect_nvidia_nemo_environment()

    assert env.data_designer.package == "data-designer"
    assert env.retriever.package == "nemo-retriever"
    assert env.curator.package == "nemo-curator"
    assert env.agent_toolkit.import_name == "nat"
    assert env.guardrails.package == "nemoguardrails"
    assert env.nemo_toolkit.import_name == "nemo"


def test_nvidia_nemo_install_commands_are_separated_by_capability():
    commands = recommended_install_commands()

    assert commands["data_designer"] == "pip install data-designer"
    assert "nemo-retriever[nemotron-parse]" in commands["retriever_nemotron_parse"]
    assert "nemo-curator[text_cpu]" in commands["curator_text_cpu"]
    assert commands["agent_toolkit"] == "pip install nvidia-nat"
    assert commands["guardrails"] == "pip install nemoguardrails"


def test_curator_adapter_disabled_returns_original_text():
    result = run_curator_pii_redaction("Caller: jane@example.com", enabled=False)

    assert result["status"] == "disabled"
    assert result["sdk"] == "nemo-curator"
    assert result["text"] == "Caller: jane@example.com"


def test_guardrails_adapter_disabled_is_non_blocking():
    result = run_guardrails_transcript_check(
        [{"turn": 1, "speaker": "speaker_1", "text": "Synthetic content."}],
        enabled=False,
    )

    assert result["status"] == "disabled"
    assert result["sdk"] == "nemoguardrails"
    assert result["passed"] is None
