"""End-to-end UI test for Customer Interactions Twin mode."""

from __future__ import annotations

import json
from io import BytesIO
import re
from zipfile import ZipFile

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/transcript_twin.py"


def _click(at: AppTest, label: str) -> None:
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, [str(e) for e in at.exception]


def test_transcript_twin_ui_preview_contract_generate_validate(monkeypatch):
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MOCK", "1")
    monkeypatch.setenv("SP_TRANSCRIPT_TWIN_MODE", "full")
    monkeypatch.delenv("SP_TRANSCRIPT_TWIN_ALLOW_FULL_SDK", raising=False)
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    source_text = (
        "Agent: Email me at alex@example.com about transformer attention.\n"
        "Customer: Call +1 555 123 4567 after the meeting."
    )
    at.text_area[0].set_value(source_text).run()
    assert not at.exception, [str(e) for e in at.exception]

    preview = at.session_state["transcript_preview"]
    assert preview is not None
    assert preview.record_count == 2
    preview_text = " ".join(row["text"] for row in preview.items[0].sample)
    assert "alex@example.com" not in preview_text
    assert "+1 555 123 4567" not in preview_text
    assert re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", preview_text)
    assert "[" not in preview_text
    assert "]" not in preview_text

    _click(at, "Build twin contract")
    contract = at.session_state["transcript_contract"]
    assert contract is not None
    assert contract.source_type == "transcript"
    assert at.session_state["transcript_text"] == ""

    _click(at, "Generate synthetic interaction")
    synthetic = at.session_state["transcript_synthetic"]
    assert synthetic is not None
    assert synthetic["status"] == "generated"
    assert synthetic["artifact_type"] == "synthetic_customer_interaction"
    assert len(synthetic["turns"]) > 0
    assert "structured_ssot" in synthetic
    assert synthetic["generation"]["data_designer"]["mode"] == "ssot_sdk_plus_platform_turns"

    _click(at, "Validate synthetic interaction")
    report = at.session_state["transcript_validation"]
    assert report is not None
    assert report["raw_source_text_used"] is False
    assert report["passed"] is True

    zip_bytes = at.session_state["transcript_twin_zip_bytes"]
    assert isinstance(zip_bytes, (bytes, bytearray))
    with ZipFile(BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
    assert {
        "canonical_contract.json",
        "synthetic_interaction.json",
        "synthetic_interaction.csv",
        "transcript_twin_ssot.json",
        "validation_report.json",
        "README.txt",
    } <= names

    body = "\n".join(str(markdown.value) for markdown in at.markdown)
    assert "Synthetic interaction" in body
    assert len(at.dataframe) >= 2

    download = next(b for b in at.download_button if "Download customer interactions twin" in b.label)
    assert download is not None


def test_transcript_twin_ui_rich_claim_context_reaches_generated_twin(monkeypatch):
    monkeypatch.setenv("SP_NEMO_DATA_DESIGNER_MOCK", "1")
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    source_text = (
        "[00:00:01.200] Agent: OmniMutual Claims Support, this is David. "
        "Can I get your auto policy number to bring up your details? "
        "[00:00:06.850] Caller: Hello, yes. My policy number is INS-441-B83. "
        "My name is Timothy Brooks. [00:00:13.100] Agent: Thank you, Mr. Brooks. "
        "To confirm identity, what is your date of birth, cell phone number, and the email address attached to the account? "
        "[00:00:20.450] Caller: My date of birth is February 14, 1988. Phone number is 555-012-9988, "
        "and my email is tim.brooks88@postbox.com. [00:00:30.900] Agent: Perfect, I have the system record open. "
        "Let's discuss the claim you started filing online. It looks like an incident involving a 2022 Honda Civic? "
        "[00:00:38.250] Caller: Yes. I was rear-ended at a stoplight on Fifth Ave near 42nd St, New York. "
        "The other driver's license plate was New York registration TX-R99-XPD. "
        "[00:00:49.700] Agent: Got it. Our system shows you checked a box regarding physical injury. Are you alright? "
        "[00:00:54.150] Caller: I have some severe concussion and a bruised ribs. "
        "I just left the emergency room at Saratoga Hospital, and they gave me a prescription for 400mg Ibuprofen. "
        "[00:01:04.600] Agent: Understood. I will log those medical updates under your active claim tracking ID #CLM-3349102."
    )
    raw_values = [
        "Timothy Brooks",
        "INS-441-B83",
        "February 14, 1988",
        "555-012-9988",
        "tim.brooks88@postbox.com",
        "TX-R99-XPD",
        "Saratoga Hospital",
        "CLM-3349102",
    ]

    at.text_area[0].set_value(source_text).run()
    assert not at.exception, [str(e) for e in at.exception]
    preview = at.session_state["transcript_preview"]
    preview_text = " ".join(row["text"] for row in preview.items[0].sample)
    assert preview.record_count == 9
    assert not any(raw in preview_text for raw in raw_values)

    _click(at, "Build twin contract")
    contract = at.session_state["transcript_contract"]
    context = contract.entities[0].metadata["synthetic_context"]
    assert {"customer_name", "policy_id", "email", "phone", "date_of_birth"} <= set(context)

    _click(at, "Generate synthetic interaction")
    synthetic = at.session_state["transcript_synthetic"]
    synthetic_text = json.dumps(synthetic, sort_keys=True)

    assert not any(raw in synthetic_text for raw in raw_values)
    assert synthetic["status"] == "generated"
    assert "turns" in synthetic
    assert "resolved_issues" in synthetic["structured_ssot"]
    assert "account_mutations" in synthetic["structured_ssot"]

    _click(at, "Validate synthetic interaction")
    assert at.session_state["transcript_validation"]["passed"] is True
