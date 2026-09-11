"""End-to-end UI test for the Customer Interaction Twin Streamlit page."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/interaction_twin.py"
SAMPLE_TRANSCRIPT = Path("tests/fixtures/local-data/Transcript3-HP.txt")


def _click(at: AppTest, label: str) -> None:
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, [str(e) for e in at.exception]


def test_interaction_twin_upload_generate_validate_and_download(monkeypatch, tmp_path):
    monkeypatch.setenv("SP_PLATFORM_DB_PATH", str(tmp_path / "platform.db"))
    assert SAMPLE_TRANSCRIPT.exists(), f"local transcript fixture is missing: {SAMPLE_TRANSCRIPT}"

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload(SAMPLE_TRANSCRIPT.name, SAMPLE_TRANSCRIPT.read_bytes(), "text/plain")
    at.run()
    assert not at.exception

    _click(at, "Generate Interaction Twin")

    result = at.session_state["interaction_result"]
    assert result is not None
    assert result.validation_report["hard_checks_passed"] is True
    assert result.package_bytes[:2] == b"PK"
    synthetic_text = "\n".join(turn.text for turn in result.synthetic.turns)
    assert "4199" not in synthetic_text
    assert "4099" not in synthetic_text
    assert "9799" not in synthetic_text
    assert "8199" not in synthetic_text

    download = next(b for b in at.download_button if "Download interaction twin" in b.label)
    assert download.disabled is False
