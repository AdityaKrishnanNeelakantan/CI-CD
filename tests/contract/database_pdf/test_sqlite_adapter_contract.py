from __future__ import annotations

from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from tests.contract.database_pdf.adapter_contract import SourceAdapterContract

pytestmark = pytest.mark.contract


class TestSQLiteAdapterContract(SourceAdapterContract):
    @pytest.fixture
    def adapter(self, temp_sqlite_db: Path) -> SQLiteSourceAdapter:
        return SQLiteSourceAdapter({"path": str(temp_sqlite_db)})

    @pytest.fixture
    def known_table_with_fk(self) -> str:
        return "orders"
