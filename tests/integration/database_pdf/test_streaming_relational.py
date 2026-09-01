"""Integration: streaming relational generation preserves FK/PK integrity."""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.validation.database.qa_report import (
    build_qa_report_from_paths,
    check_primary_key_uniqueness_from_path,
)
from synth_platform.engine.generation.database.relational_generator import generate_relational_dataset
from synth_platform.engine.generation.database.relational_service import (
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.generation.database.streaming_generator import generate_relational_dataset_streaming
from synth_platform.engine.training.database.base import SynthesizerAdapter

pytestmark = pytest.mark.integration


class _OffsetAwareAdapter(SynthesizerAdapter):
    model_type = "fake_offset"
    file_extension = ".json"
    serialization_format = "json"

    def __init__(self, pk_column: str, extra: dict[str, list] | None = None) -> None:
        self._pk_column = pk_column
        self._extra = extra or {}

    def fit(self, df, table_name, table_contract, seed):
        return {}

    def sample(
        self,
        num_rows: int,
        seed: int | None = None,
        category_overrides=None,
        row_offset: int = 0,
    ) -> pd.DataFrame:
        start = int(row_offset)
        data = {self._pk_column: [f"{self._pk_column}-{start + i}" for i in range(num_rows)]}
        for name, values in self._extra.items():
            data[name] = [values[i % len(values)] for i in range(num_rows)]
        return pd.DataFrame(data)

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls("id")


def _schema_graph():
    return {
        "nodes": ["customers", "orders"],
        "edges": [
            {
                "parent": "customers",
                "child": "orders",
                "parent_key": "customer_id",
                "child_key": "customer_id",
            }
        ],
        "self_referencing_edges": [],
        "generation_order": ["customers", "orders"],
        "condensed_order": [["customers"], ["orders"]],
    }


def _contract():
    return {
        "tables": {
            "customers": {
                "primary_key": ["customer_id"],
                "foreign_keys": [],
                "business_rules": [],
            },
            "orders": {
                "primary_key": ["order_id"],
                "foreign_keys": [
                    {
                        "column": "customer_id",
                        "references_table": "customers",
                        "references_column": "customer_id",
                    }
                ],
                "business_rules": [],
            },
        }
    }


def test_streaming_relational_fk_and_pk(tmp_path: Path):
    adapters = {
        "customers": _OffsetAwareAdapter("customer_id"),
        "orders": _OffsetAwareAdapter("order_id", extra={"customer_id": ["placeholder"]}),
    }
    samples_dir = tmp_path / "relational_samples"
    result = generate_relational_dataset_streaming(
        _schema_graph(),
        adapters,
        {"customers": 40, "orders": 120},
        _contract(),
        samples_dir,
        tmp_path / "relational_keys.sqlite",
        batch_size=15,
        seed=11,
    )

    assert result["fk_validity"]["overall_fk_validity"] == 1.0
    assert result["tables"]["customers"]["row_count"] == 40
    assert result["tables"]["orders"]["row_count"] == 120
    assert Path(result["tables"]["customers"]["path"]).is_file()
    assert Path(result["tables"]["orders"]["path"]).is_file()

    customers = pd.read_csv(result["tables"]["customers"]["path"])
    orders = pd.read_csv(result["tables"]["orders"]["path"])
    assert set(orders["customer_id"]) <= set(customers["customer_id"])
    assert customers["customer_id"].is_unique
    assert orders["order_id"].is_unique

    pk_customers = check_primary_key_uniqueness_from_path(
        result["tables"]["customers"]["path"], ["customer_id"], chunk_size=10
    )
    assert pk_customers["is_unique"] is True
    pk_orders = check_primary_key_uniqueness_from_path(
        result["tables"]["orders"]["path"], ["order_id"], chunk_size=10
    )
    assert pk_orders["is_unique"] is True

    qa = build_qa_report_from_paths(
        result["tables"],
        _contract(),
        result["fk_validity"],
        result["constraint_reports"],
        chunk_size=10,
    )
    assert qa["hard_checks_passed"] is True
    assert qa["validation_mode"] == "streaming_paths"


def test_run_relational_generation_streaming_flag(tmp_path: Path):
    adapters = {
        "customers": _OffsetAwareAdapter("customer_id"),
        "orders": _OffsetAwareAdapter("order_id", extra={"customer_id": ["placeholder"]}),
    }
    contract = {
        "dataset_id": "stream-test",
        "tables": {
            "customers": {
                "primary_key": ["customer_id"],
                "columns": {"customer_id": {"physical_type": "TEXT", "semantic_type": "identifier"}},
                "foreign_keys": [],
                "business_rules": [],
            },
            "orders": {
                "primary_key": ["order_id"],
                "columns": {
                    "order_id": {"physical_type": "TEXT", "semantic_type": "identifier"},
                    "customer_id": {"physical_type": "TEXT", "semantic_type": "identifier"},
                },
                "foreign_keys": [
                    {
                        "column": "customer_id",
                        "references_table": "customers",
                        "references_column": "customer_id",
                    }
                ],
                "business_rules": [],
            },
        },
    }
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_relational_generation(
        contract,
        adapters,
        {"customers": 20, "orders": 50},
        manifest,
        contract_reference="dataset_contract.json",
        seed=3,
        batch_size=7,
    )
    assert result.is_success()
    report = load_relational_generation_report(manifest.output_path("relational_generation_report.json"))
    assert report["streaming"] is True
    assert report["batch_size"] == 7
    assert report["fk_validity"]["overall_fk_validity"] == 1.0


def test_streaming_memory_below_full_in_memory(tmp_path: Path):
    """Relative memory smoke: streaming peak should be materially below full path."""
    n_parent = 2_000
    n_child = 10_000
    batch_size = 2_000
    # Wide payload so O(N) frame retention dominates SQLite/CSV bookkeeping.
    wide = {f"c{i}": [f"{'x' * 40}-{i}"] for i in range(12)}
    adapters_stream = {
        "customers": _OffsetAwareAdapter("customer_id", extra=wide),
        "orders": _OffsetAwareAdapter("order_id", extra={**wide, "customer_id": ["x"]}),
    }
    adapters_full = {
        "customers": _OffsetAwareAdapter("customer_id", extra=wide),
        "orders": _OffsetAwareAdapter("order_id", extra={**wide, "customer_id": ["x"]}),
    }
    graph = _schema_graph()
    counts = {"customers": n_parent, "orders": n_child}

    tracemalloc.start()
    tables = generate_relational_dataset(graph, adapters_full, counts, seed=1)
    # Retain references through measurement so peak reflects held frames.
    retained = sum(df.memory_usage(deep=True).sum() for df in tables.values())
    _, peak_full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert retained > 0

    tracemalloc.start()
    _ = generate_relational_dataset_streaming(
        graph,
        adapters_stream,
        counts,
        _contract(),
        tmp_path / "samples",
        tmp_path / "keys.sqlite",
        batch_size=batch_size,
        seed=1,
    )
    _, peak_stream = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Streaming should use substantially less peak Python allocations than
    # holding both full frames (relative, not absolute GB).
    assert peak_stream < peak_full * 0.6, (
        f"streaming peak {peak_stream} not materially below full {peak_full}"
    )
