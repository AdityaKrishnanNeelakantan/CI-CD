"""End-to-end UI test for the PDF Twin Streamlit page."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/pdf_twin.py"
SAMPLE_PDF = Path("tests/fixtures/local-data/healthcare_patient_summary.pdf")


def _click(at: AppTest, label: str) -> None:
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, [str(e) for e in at.exception]


def test_full_pdf_twin_wizard_on_the_real_healthcare_summary():
    assert SAMPLE_PDF.exists(), f"local PDF fixture is missing: {SAMPLE_PDF}"
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload(SAMPLE_PDF.name, SAMPLE_PDF.read_bytes(), "application/pdf")
    at.run()
    assert not at.exception

    # This regression protects the default native/OCR path.
    extraction_engine = next(s for s in at.selectbox if s.label == "Extraction engine")
    extraction_engine.set_value("Automatic (native/OCR)").run()
    assert not at.exception

    _click(at, "Discover document")
    _click(at, "Build twin template")
    _click(at, "Understand fields")
    _click(at, "Generate values")
    _click(at, "Build twin document")
    pre_validation_download = next(b for b in at.download_button if b.label == "Download synthetic twin PDF")
    assert pre_validation_download.disabled is True

    _click(at, "Validate twin")

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Extraction method"] in {"native", "ocr"}
    assert metrics["Integrity checks"] == "passed"
    assert metrics["Field accuracy"] == "100%"
    assert metrics["Overflow failures"] == "0"


def test_pdf_download_is_disabled_after_render_until_validation_runs(tmp_path):
    rendered = tmp_path / "rendered.pdf"
    rendered.write_bytes(b"%PDF-1.4\n%%EOF\n")
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    at.session_state["pdf_workdir"] = tmp_path
    at.session_state["pdf_source_path"] = source
    at.session_state["pdf_doc_id"] = "doc1"
    at.session_state["pdf_profile"] = {
        "extraction_method": "native",
        "language": {"language": "en"},
        "pii_findings": [],
        "classification": {"document_type": "test", "status": "ok"},
    }
    at.session_state["pdf_template"] = {
        "pages": [
            {
                "page_number": 1,
                "regions": [
                    {
                        "region_id": "field_1",
                        "region_type": "field",
                        "label": "Name",
                        "value_type": "text",
                    }
                ],
            }
        ]
    }
    at.session_state["pdf_binding_map"] = {
        "bindings": [
            {
                "binding_type": "field",
                "label": "Name",
                "semantic_role": "person_name",
                "generator_strategy": "faker",
            }
        ]
    }
    at.session_state["pdf_values"] = {
        "fields": {"field_1": {"label": "Name", "value": "Ada Lovelace"}},
        "tables": {},
        "inline_spans": {},
    }
    at.session_state["pdf_ground_truth"] = {}
    at.session_state["pdf_rendered_path"] = rendered
    at.session_state["pdf_validation_report"] = None

    at.run()
    assert not at.exception
    download = next(b for b in at.download_button if b.label == "Download synthetic twin PDF")
    assert download.disabled is True
