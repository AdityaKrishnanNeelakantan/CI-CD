"""AppTest: Schema Twin accepts generic SQL DDL uploads."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/schema_twin.py"
INPUT_SQL = Path("Input-Data/Schema/schema.sql")


def _click(at: AppTest, label: str) -> None:
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.skipif(not INPUT_SQL.exists(), reason="Input-Data schema.sql not present")
def test_schema_twin_sql_ddl_generate_validate_download():
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    assert not at.exception

    uploader = at.get("file_uploader")[0]
    uploader.upload(INPUT_SQL.name, INPUT_SQL.read_bytes(), "application/sql")
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["schema_config"] is not None
    assert at.session_state["schema_summary"].table_count >= 1

    row_input = next(n for n in at.number_input if n.label == "Base rows")
    row_input.set_value(15).run()
    assert not at.exception

    _click(at, "Generate Synthetic Data")
    result = at.session_state["schema_result"]
    assert result is not None
    assert result.preview_tables
    assert 15 in result.row_counts.values()
    if result.schema.relationships:
        assert len(set(result.row_counts.values())) > 1
    assert at.session_state["schema_zip_bytes"][:2] == b"PK"

    # Second configuration: different row count / seed via UI controls if present
    seed_inputs = [n for n in at.number_input if "seed" in n.label.lower()]
    if seed_inputs:
        seed_inputs[0].set_value(99).run()
    row_input = next(n for n in at.number_input if n.label == "Base rows")
    row_input.set_value(10).run()
    _click(at, "Generate Synthetic Data")
    result2 = at.session_state["schema_result"]
    assert 10 in result2.row_counts.values()
    if result2.schema.relationships:
        assert len(set(result2.row_counts.values())) > 1
