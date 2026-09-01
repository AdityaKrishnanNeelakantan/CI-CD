from __future__ import annotations

import pytest

from synth_platform.engine.discovery.database.adapters.registry import create_source_adapter
from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter

pytestmark = pytest.mark.unit


def test_registry_creates_sqlite_adapter_for_registered_type():
    adapter = create_source_adapter("sqlite", {"path": "irrelevant.db"})
    assert isinstance(adapter, SQLiteSourceAdapter)
