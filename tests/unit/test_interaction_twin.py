from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from synth_platform.application.services.transfer_service import TransferService
from synth_platform.application.workflows.interaction_twin import run_interaction_twin
from synth_platform.engine.generation.interaction import (
    InteractionGenerationConfig,
    generate_interaction_twin,
    parse_transcript,
)
from synth_platform.errors import TransferBlockedError
from synth_platform.infrastructure.persistence.platform_db import PlatformDB
from synth_platform.interfaces.streamlit.project_ui import load_project_summaries


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "local-data"
CARD_VALUE = "4199 4099 9799 8199"


def _fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def test_real_fixture_transcript_parses_into_structured_turns() -> None:
    transcript = parse_transcript(_fixture("Transcript1-HP.txt"))

    assert len(transcript.turns) > 10
    assert {"agent", "customer"}.issubset(transcript.speaker_counts)
    assert transcript.turns[0].speaker == "agent"
    assert transcript.turns[0].text


def test_card_number_fragments_from_real_billing_transcript_do_not_survive_output() -> None:
    source_text = _fixture("Transcript3-HP.txt")

    result = generate_interaction_twin(
        source_text,
        InteractionGenerationConfig(remove_sensitive_information=True),
    )
    synthetic_text = "\n".join(turn.text for turn in result.synthetic.turns)

    assert result.redaction_report["findings_by_type"]["credit_card_fragment"] >= 4
    assert CARD_VALUE not in synthetic_text
    for fragment in CARD_VALUE.split():
        assert fragment not in synthetic_text
    assert result.validation_report["hard_checks_passed"] is True


def test_end_to_end_generation_writes_json_log_and_validation_package(tmp_path: Path) -> None:
    result = run_interaction_twin(
        _fixture("Transcript2-HP.txt"),
        output_dir=tmp_path / "interaction",
        history=None,
    )

    assert result.hard_checks_passed is True
    assert result.package_bytes
    assert set(result.output_paths) == {"json", "log", "validation_report", "redaction_report"}
    package = tmp_path / "interaction_twin.zip"
    package.write_bytes(result.package_bytes)
    with zipfile.ZipFile(package) as zf:
        assert {"synthetic_interaction.json", "synthetic_interaction.log", "validation_report.json"}.issubset(
            set(zf.namelist())
        )


def test_interaction_transfer_blocks_missing_or_failed_validation_and_allows_passing() -> None:
    service = TransferService()

    with pytest.raises(TransferBlockedError):
        service.downloadable_bytes(
            workflow="interaction_twin",
            output_id="interaction_twin.zip",
            validation_report=None,
            data=b"payload",
        )

    with pytest.raises(TransferBlockedError):
        service.downloadable_bytes(
            workflow="interaction_twin",
            output_id="interaction_twin.zip",
            validation_report={"hard_checks_passed": False, "export_ready": True},
            data=b"payload",
        )

    transfer = service.downloadable_bytes(
        workflow="interaction_twin",
        output_id="interaction_twin.zip",
        validation_report={"hard_checks_passed": True, "export_ready": True},
        data=b"payload",
    )

    assert transfer.data == b"payload"
    assert transfer.approval.validation_status == "PASS"


def test_completed_interaction_run_appears_in_my_projects(tmp_path: Path) -> None:
    db = PlatformDB(tmp_path / "platform.db")

    result = run_interaction_twin(
        _fixture("Transcript1-HP-ssot-smoke.txt"),
        output_dir=tmp_path / "output",
        project_name="Interaction Fixture",
        history=db,
    )
    db.record_transfer_attempt(
        workflow="interaction_twin",
        output_id="interaction_twin.zip",
        allowed=True,
        validation_status="PASS",
        reason="interaction validation and export readiness passed",
    )

    [summary] = load_project_summaries(db, workflow_type="interaction")

    assert result.hard_checks_passed is True
    assert summary.name == "Interaction Fixture"
    assert summary.workflow_type == "interaction"
    assert summary.workflow_label == "Interaction"
    assert summary.validation == "Passed"
    assert summary.transfer == "Allowed"
    assert summary.records == f"{len(result.synthetic.turns):,}"
