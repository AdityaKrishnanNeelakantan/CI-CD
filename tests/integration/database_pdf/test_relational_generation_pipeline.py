"""Integration: the full chain up through relational generation - discovery
-> profiling -> inference -> approval -> training (both tables) -> load
each table's saved model -> relational generation with real FK linking,
using the project's own real two-table fixture (customers -> orders).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.discovery.database.service import run_discovery
from synth_platform.engine.inference.database.contract import load_dataset_contract
from synth_platform.engine.inference.database.service import run_contract_approval, run_inference
from synth_platform.engine.profiling.database.service import run_profiling
from synth_platform.engine.generation.database.relational_service import (
    RELATIONAL_REPORT_FILENAME,
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling

pytestmark = pytest.mark.integration


def _run_up_to_training(temp_sqlite_db: Path, tmp_path: Path, model_type: str = "safe_gaussian_copula"):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    discovery_result = run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads(Path(discovery_result.output_references[0]).read_text())
    profiling_result = run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)
    profile_data = json.loads(Path(profiling_result.output_references[0]).read_text())
    inference_result = run_inference(
        adapter, discovery_data, profile_data, manifest, "discovery.json", "profile.json", sample_limit=100
    )
    candidates_data = json.loads(Path(inference_result.output_references[0]).read_text())
    decisions = {
        table: {col: cand["semantic_type"] for col, cand in cols.items()}
        for table, cols in candidates_data["tables"].items()
    }
    run_contract_approval(
        dataset_id="ds", source_fingerprint=discovery_data["source_fingerprint"],
        discovery_data=discovery_data, candidates_by_table=candidates_data["tables"],
        manifest=manifest, metadata_dir=metadata_dir, candidates_reference="semantic_candidates.json",
        decisions=decisions,
    )
    contract = load_dataset_contract(metadata_dir)

    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100,
        num_rows_to_generate=5, seed=1, model_type=model_type,
    )
    training_report = load_training_report(Path(training_result.output_references[0]))
    return manifest, contract, training_report


def _load_adapters(training_report):
    adapters = {}
    for table_name, table_report in training_report["tables"].items():
        adapter_cls = get_synthesizer_adapter_class(table_report["model_type"])
        adapters[table_name] = adapter_cls.load(table_report["model_path"])
    return adapters


def test_customers_orders_fixture_has_a_real_foreign_key_in_the_contract(temp_sqlite_db: Path, tmp_path: Path):
    _manifest, contract, _training_report = _run_up_to_training(temp_sqlite_db, tmp_path)
    orders_fks = contract["tables"]["orders"]["foreign_keys"]
    assert any(fk["references_table"] == "customers" for fk in orders_fks)


def test_relational_generation_produces_100_percent_fk_validity(temp_sqlite_db: Path, tmp_path: Path):
    manifest, contract, training_report = _run_up_to_training(temp_sqlite_db, tmp_path)
    adapters = _load_adapters(training_report)

    result = run_relational_generation(
        contract, adapters, {"customers": 10, "orders": 40}, manifest,
        contract_reference="dataset_contract.json", seed=42,
    )
    assert result.is_success()
    assert result.metrics["overall_fk_validity"] == 1.0

    report = load_relational_generation_report(
        manifest.output_path(RELATIONAL_REPORT_FILENAME)
    )
    assert report["generation_order"] == ["customers", "orders"]
    assert report["fk_validity"]["overall_fk_validity"] == 1.0
    assert report["tables"]["customers"]["row_count"] == 10
    assert report["tables"]["orders"]["row_count"] == 40

    import pandas as pd

    customers_df = pd.read_csv(report["tables"]["customers"]["path"])
    orders_df = pd.read_csv(report["tables"]["orders"]["path"])
    assert set(orders_df["customer_id"]) <= set(customers_df["customer_id"])
