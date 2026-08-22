"""End-to-end UI test for Schema Twin (Schema Mode).

Drives src/synth_platform/interfaces/streamlit/pages/schema_twin.py through Streamlit AppTest: upload schema,
configure, generate, preview, validate, and download — using the existing
the schema-driven engine via the Schema Mode facade.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/schema_twin.py"
SAMPLE_SCHEMA = Path("tests/fixtures/schema/schema_twin_minimal.json")


def _click(at: AppTest, label: str) -> None:
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, [str(e) for e in at.exception]


def test_schema_twin_end_to_end_generate_preview_validate_download():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload(
        SAMPLE_SCHEMA.name,
        SAMPLE_SCHEMA.read_bytes(),
        "application/json",
    )
    at.run()
    assert not at.exception
    assert at.session_state["schema_config"] is not None

    row_input = next(n for n in at.number_input if n.label == "Rows per table")
    row_input.set_value(20).run()
    assert not at.exception

    _click(at, "Generate Synthetic Data")
    result = at.session_state["schema_result"]
    assert result is not None
    assert set(result.preview_tables) == {"users", "orders"}
    assert result.row_counts["users"] == 20
    assert result.row_counts["orders"] == 20
    assert at.session_state["schema_zip_bytes"]

    preview = next(s for s in at.selectbox if s.label == "Preview table")
    preview.set_value("orders").run()
    assert not at.exception

    download = next(b for b in at.download_button if "Download synthetic dataset" in b.label)
    assert download is not None
    zip_bytes = at.session_state["schema_zip_bytes"]
    assert isinstance(zip_bytes, (bytes, bytearray))
    assert len(zip_bytes) > 0
    assert zip_bytes[:2] == b"PK"  # ZIP magic
