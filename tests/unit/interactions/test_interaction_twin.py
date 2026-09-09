from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from synth_platform.application.coordinator import (
    CapabilityId,
    RouteRequest,
    route_request,
)
from synth_platform.application.workflows.interaction_twin import run_interaction_twin
from synth_platform.domain.validation.models import Status
from synth_platform.engine.interactions.service import (
    build_interaction_ssot,
    parse_and_sanitize_transcript,
    render_sanitized_source,
    validate_interaction_ssot,
)

pytestmark = pytest.mark.unit


class RecordingModel:
    name = "recording-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, bool]] = []

    def complete(self, system: str, user: str, *, json_only: bool = False) -> str:
        self.calls.append((system, user, json_only))
        return self.response


def test_sensitive_fields_with_punctuation_are_sanitized() -> None:
    source = (
        "Customer: My password is abc:def/ghi+1 and account number is ABC/12345.\n"
        "Agent: I will reset it."
    )

    transcript = parse_and_sanitize_transcript(source)
    sanitized = render_sanitized_source(transcript)

    assert "abc:def/ghi+1" not in sanitized
    assert "ABC/12345" not in sanitized
    assert "[SECRET_001]" in sanitized
    assert "[ACCOUNT_ID_001]" in sanitized


def test_ordinary_secret_label_phrases_are_not_redacted() -> None:
    transcript = parse_and_sanitize_transcript(
        "Customer: I need a password reset because my token expired.\n"
        "Agent: I can explain the reset process."
    )
    sanitized = render_sanitized_source(transcript)

    assert "password reset" in sanitized
    assert "token expired" in sanitized
    assert "[SECRET_" not in sanitized


def test_sensitive_metadata_label_is_not_misparsed_as_a_speaker() -> None:
    transcript = parse_and_sanitize_transcript(
        "Password: abc:def/ghi+1\nCustomer: My account is locked."
    )
    sanitized = render_sanitized_source(transcript)

    assert "abc:def/ghi+1" not in sanitized
    assert "Password:" in transcript.turns[0].raw_text
    assert transcript.turns[-1].role.value == "customer"
    assert "locked" in transcript.turns[-1].text


def test_short_speaker_name_does_not_match_a_longer_semantic_label() -> None:
    transcript = parse_and_sanitize_transcript(
        "Bill: I have a billing question.\nAgent: I can help."
    )
    ssot = build_interaction_ssot(transcript)

    report = validate_interaction_ssot(transcript, ssot)

    assert report.metrics["raw_speaker_leak_count"] == 0
    assert report.release.verdict != Status.FAIL


def test_model_is_called_once_with_sanitized_source_only(tmp_path: Path) -> None:
    raw_email = "private.person@example.com"
    model = RecordingModel(
        json.dumps(
            {
                "topics": ["account_access"],
                "issue_codes": ["login_problem"],
                "action_codes": ["credential_reset"],
                "resolution_status": "resolved",
                "initial_customer_sentiment": "negative",
                "final_customer_sentiment": "positive",
            }
        )
    )

    result = run_interaction_twin(
        f"Customer: Email {raw_email}; my login is broken.\n"
        "Agent: I reset the password and it is working now.",
        runs_dir=tmp_path / "runs",
        model=model,
    )

    assert len(model.calls) == 1
    _system, user, json_only = model.calls[0]
    assert json_only is True
    assert raw_email not in user
    assert "[EMAIL_001]" in user
    assert result.ssot.extraction.model_used is True


def test_invalid_model_output_falls_back_deterministically(tmp_path: Path) -> None:
    model = RecordingModel('{"topics": ["not_an_enum"]}')

    result = run_interaction_twin(
        "Customer: My login is broken.\nAgent: Please try a password reset.",
        runs_dir=tmp_path / "runs",
        model=model,
    )

    assert len(model.calls) == 1
    assert result.ssot.extraction.mode == "deterministic_fallback"
    assert result.ssot.extraction.model_used is False


def test_released_zip_has_exact_contract_and_no_raw_source(tmp_path: Path) -> None:
    raw_email = "private.person@example.com"
    raw_secret = "abc:def/ghi+1"
    result = run_interaction_twin(
        f"Customer Jane Doe: Email {raw_email}; password {raw_secret}.\n"
        "Agent Bob Smith: I reset the password and it is working now.",
        source_name="support.log",
        runs_dir=tmp_path / "runs",
    )

    assert result.validation_report.release.verdict != Status.FAIL
    assert result.package_path is not None
    with zipfile.ZipFile(result.package_path) as archive:
        assert set(archive.namelist()) == {
            "sanitized_source.txt",
            "interaction_ssot.json",
            "validation_report.json",
            "manifest.json",
        }
        for name in archive.namelist():
            content = archive.read(name)
            assert raw_email.encode() not in content
            assert raw_secret.encode() not in content

    for path in result.run_manifest.run_dir.iterdir():
        if path.is_file():
            content = path.read_bytes()
            assert raw_email.encode() not in content
            assert raw_secret.encode() not in content


def test_participant_integrity_rejects_source_fact_drift() -> None:
    transcript = parse_and_sanitize_transcript(
        "Customer: My login is broken.\nAgent: I can help."
    )
    ssot = build_interaction_ssot(transcript)
    corrupted = ssot.model_copy(
        update={
            "participants": [ssot.participants[0], ssot.participants[0]],
            "participant_count": 2,
        }
    )

    report = validate_interaction_ssot(transcript, corrupted)

    assert report.release.verdict == Status.FAIL
    assert "participant_reference_integrity" in report.release.blocking


def test_interaction_route_is_available() -> None:
    decision = route_request(RouteRequest(attachment_names=("support.log",)))

    assert decision.capability_id == CapabilityId.INTERACTION_TWIN
    assert decision.can_invoke is True
