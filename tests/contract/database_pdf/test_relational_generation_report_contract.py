"""Contract: run_relational_generation() always writes the same top-level
shape.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.generation.database.relational_service import (
    RELATIONAL_REPORT_FILENAME,
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.training.database.base import SynthesizerAdapter

pytestmark = pytest.mark.contract

REQUIRED_TOP_LEVEL_KEYS = {"generated_at", "generation_order", "fk_validity", "constraint_reports", "tables"}


class _FakeAdapter(SynthesizerAdapter):
    model_type = "fake"
    file_extension = ".json"
    serialization_format = "json"

    def __init__(self, pk_column: str) -> None:
        self._pk_column = pk_column

    def fit(self, df, table_name, table_contract, seed):
        return {}

    def sample(self, num_rows, seed=None):
        return pd.DataFrame({self._pk_column: [f"{self._pk_column}-{i}" for i in range(num_rows)]})

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls("id")


def test_relational_generation_report_shape(tmp_path):
    contract = {
        "tables": {
            "customers": {"primary_key": ["customer_id"], "foreign_keys": [], "columns": {}, "business_rules": []},
            "orders": {
                "primary_key": ["order_id"],
                "foreign_keys": [{"column": "customer_id", "references_table": "customers", "references_column": "customer_id"}],
                "columns": {},
                "business_rules": [],
            },
        }
    }
    adapters = {"customers": _FakeAdapter("customer_id"), "orders": _FakeAdapter("order_id")}
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    result = run_relational_generation(
        contract, adapters, {"customers": 5, "orders": 20}, manifest, contract_reference="contract.json", seed=1
    )
    assert result.is_success()

    report = load_relational_generation_report(manifest.output_path(RELATIONAL_REPORT_FILENAME))
    assert set(report) == REQUIRED_TOP_LEVEL_KEYS
    assert set(report["tables"]) == {"customers", "orders"}
    for table_entry in report["tables"].values():
        assert {"row_count", "path"} <= set(table_entry)
