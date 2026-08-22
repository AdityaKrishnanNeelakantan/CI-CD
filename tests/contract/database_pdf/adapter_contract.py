"""Reusable contract suite every SourceAdapter implementation must pass.

Not named test_*.py on purpose: pytest must not collect this base class
directly (its fixtures raise NotImplementedError). Registering a new
adapter (PostgreSQL, CSV, ...) means adding one concrete test class in this
directory that supplies `adapter` and `known_table_with_fk` fixtures - the
assertions below must never need to change or special-case a source type.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.discovery.database.adapters.errors import UnknownTableError

REQUIRED_TABLE_KEYS = {"columns", "primary_key", "foreign_keys", "indexes", "estimated_row_count"}
REQUIRED_COLUMN_KEYS = {"name", "database_type", "nullable"}
REQUIRED_FK_KEYS = {"column", "references_table", "references_column"}
REQUIRED_DISCOVERY_KEYS = {"source_type", "discovered_at", "tables", "source_fingerprint"}


class SourceAdapterContract:
    """Subclass this in a concrete test module; do not instantiate directly."""

    @pytest.fixture
    def adapter(self):
        raise NotImplementedError("Concrete contract tests must provide an `adapter` fixture")

    @pytest.fixture
    def known_table_with_fk(self):
        """Name of a fixture table that has at least one foreign key."""
        raise NotImplementedError(
            "Concrete contract tests must provide a `known_table_with_fk` fixture"
        )

    def test_test_connection_returns_required_shape(self, adapter):
        result = adapter.test_connection()
        assert isinstance(result, dict)
        assert "healthy" in result
        assert "source_type" in result
        assert result["healthy"] is True
        # Credentials must never be echoed back in a health check result.
        serialized = str(result).lower()
        assert "password" not in serialized

    def test_list_tables_returns_list_of_strings(self, adapter):
        tables = adapter.list_tables()
        assert isinstance(tables, list)
        assert all(isinstance(name, str) for name in tables)

    def test_discover_returns_generic_contract_shape(self, adapter):
        result = adapter.discover()

        assert set(result) == REQUIRED_DISCOVERY_KEYS
        assert isinstance(result["tables"], dict)
        assert result["source_fingerprint"].startswith("sha256:")

        for table in result["tables"].values():
            assert REQUIRED_TABLE_KEYS.issubset(table)
            assert isinstance(table["primary_key"], list)
            assert isinstance(table["foreign_keys"], list)
            assert isinstance(table["estimated_row_count"], int)
            for column in table["columns"]:
                assert REQUIRED_COLUMN_KEYS.issubset(column)
                assert isinstance(column["nullable"], bool)
            for fk in table["foreign_keys"]:
                assert REQUIRED_FK_KEYS.issubset(fk)

    def test_read_sample_returns_bounded_dataframe(self, adapter, known_table_with_fk):
        df = adapter.read_sample(known_table_with_fk, limit=2)
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 2

    def test_unknown_table_raises_unknown_table_error(self, adapter):
        with pytest.raises(UnknownTableError):
            adapter.get_schema("__table_that_does_not_exist__")

    def test_fingerprint_is_deterministic_across_calls(self, adapter):
        first = adapter.discover()["source_fingerprint"]
        second = adapter.discover()["source_fingerprint"]
        assert first == second

    def test_foreign_keys_reference_a_discovered_table(self, adapter):
        discovery = adapter.discover()
        table_names = set(discovery["tables"])
        for table in discovery["tables"].values():
            for fk in table["foreign_keys"]:
                assert fk["references_table"] in table_names
