"""Home page reflects all supported SSOT workflows."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.e2e

APP_PATH = "src/synth_platform/interfaces/streamlit/pages/home.py"


def test_home_page_is_chat_first_and_lists_all_modes():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    body = "\n".join(
        [str(markdown.value) for markdown in at.markdown]
        + [str(caption.value) for caption in at.caption]
    )
    assert "Platform chat" in body
    assert "All modes" in body
    assert "Schema Mode" in body
    assert "Database Twin" in body
    assert "PDF Twin" in body
    assert "Customer Interactions Twin" in body
    assert "Production ready" in body
    assert "Work in progress" in body
    assert "Shared platform controls" not in body
