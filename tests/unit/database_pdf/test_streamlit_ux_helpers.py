"""Unit tests for Streamlit-facing helper filters."""

from __future__ import annotations

from synth_platform.interfaces.streamlit.components.common.ux import (
    list_contract_free_text_columns,
    relationship_volume_warnings,
)


def test_controlled_text_columns_are_not_llm_free_text_candidates():
    contract = {
        "tables": {
            "branches": {
                "columns": {
                    "country": {"semantic_type": "free_text"},
                    "city": {"semantic_type": "free_text"},
                    "town": {"semantic_type": "free_text"},
                    "merchant_name": {"semantic_type": "free_text"},
                    "company_name": {"semantic_type": "free_text"},
                    "notes": {"semantic_type": "free_text"},
                    "status": {"semantic_type": "free_text"},
                }
            }
        }
    }

    rows = list_contract_free_text_columns(contract)

    assert rows == [{"table": "branches", "column": "notes", "role": "free_text"}]


def test_child_rows_below_parent_is_ok_when_it_matches_ssot_ratio():
    contract = {
        "tables": {
            "customers": {"foreign_keys": []},
            "loans": {"foreign_keys": [{"references_table": "customers"}]},
        }
    }
    source_preview_tables = {
        "customers": {"row_count": 1000},
        "loans": {"row_count": 250},
    }

    warnings = relationship_volume_warnings(
        contract,
        {"customers": 100, "loans": 25},
        source_preview_tables,
    )

    assert warnings == []


def test_child_parent_volume_warning_only_fires_on_ssot_ratio_drift():
    contract = {
        "tables": {
            "customers": {"foreign_keys": []},
            "loans": {"foreign_keys": [{"references_table": "customers"}]},
        }
    }
    source_preview_tables = {
        "customers": {"row_count": 1000},
        "loans": {"row_count": 250},
    }

    warnings = relationship_volume_warnings(
        contract,
        {"customers": 100, "loans": 5},
        source_preview_tables,
    )

    assert len(warnings) == 1
    assert "lower than the SSOT shape" in warnings[0]
