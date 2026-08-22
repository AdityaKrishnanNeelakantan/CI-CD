"""Integration: business rules (derived fields + hard constraints) from
the approved dataset_contract are actually applied during relational
generation - the "hybrid" combination of the relational generator and
the constraint engine, checked against the already-FK-resolved row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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


def _build_contract_and_train(temp_sqlite_db: Path, tmp_path: Path):
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

    # Inject business rules directly - there is no UI/approval workflow
    # for these yet, so tests exercise the engine the same way a future
    # approval step would populate this same contract field.
    contract["tables"]["orders"]["business_rules"] = [
        {"type": "derived", "target_column": "amount_with_tax", "formula": "amount * 1.1"},
        {"type": "constraint", "rule_id": "non_negative_amount", "expression": "amount >= 0", "on_violation": "reject_row"},
    ]

    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100,
        num_rows_to_generate=5, seed=1, model_type="safe_gaussian_copula",
    )
    training_report = load_training_report(Path(training_result.output_references[0]))
    return manifest, contract, training_report


def _load_adapters(training_report):
    adapters = {}
    for table_name, table_report in training_report["tables"].items():
        adapter_cls = get_synthesizer_adapter_class(table_report["model_type"])
        adapters[table_name] = adapter_cls.load(table_report["model_path"])
    return adapters


def test_derived_field_is_recalculated_on_every_generated_row(temp_sqlite_db: Path, tmp_path: Path):
    manifest, contract, training_report = _build_contract_and_train(temp_sqlite_db, tmp_path)
    adapters = _load_adapters(training_report)

    result = run_relational_generation(
        contract, adapters, {"customers": 10, "orders": 40}, manifest,
        contract_reference="dataset_contract.json", seed=1,
    )
    assert result.is_success()

    report = load_relational_generation_report(manifest.output_path(RELATIONAL_REPORT_FILENAME))
    orders_df = pd.read_csv(report["tables"]["orders"]["path"])
    assert "amount_with_tax" in orders_df.columns
    assert (orders_df["amount_with_tax"] - orders_df["amount"] * 1.1).abs().max() < 1e-9


def test_constraint_report_is_recorded_and_reflects_full_compliance(temp_sqlite_db: Path, tmp_path: Path):
    manifest, contract, training_report = _build_contract_and_train(temp_sqlite_db, tmp_path)
    adapters = _load_adapters(training_report)

    run_relational_generation(
        contract, adapters, {"customers": 10, "orders": 40}, manifest,
        contract_reference="dataset_contract.json", seed=1,
    )
    report = load_relational_generation_report(manifest.output_path(RELATIONAL_REPORT_FILENAME))

    assert "orders" in report["constraint_reports"]
    constraint_report = report["constraint_reports"]["orders"]
    # amount is a real numeric column from the source schema (never
    # negative in the fixture), and the generator never produces negative
    # amounts either (numeric copula sampling around a positive mean) -
    # this constraint should hold with full compliance.
    orders_df = pd.read_csv(report["tables"]["orders"]["path"])
    assert (orders_df["amount"] >= 0).all()
    assert constraint_report["total_violations"] == 0
