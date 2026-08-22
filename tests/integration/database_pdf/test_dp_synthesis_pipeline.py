"""Integration: the full chain up through training with model_type=
"dp_gaussian_copula" - discovery -> profiling -> inference -> approval ->
training, using the project's real two-table fixture (customers ->
orders), proving the DP adapter trains through the real pipeline (not
just in isolation) and the persisted training report/model carry a real
privacy_summary with the configured budget actually spent.
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
from synth_platform.engine.profiling.database.service import PROFILE_FILENAME, load_profile, run_profiling
from synth_platform.engine.training.database.registry import get_synthesizer_adapter_class
from synth_platform.engine.training.database.service import load_training_report, run_training_and_sampling

pytestmark = pytest.mark.integration

# Generous, domain-plausible public bounds - not derived from the real
# data - covering every column that might plausibly be inferred as
# numerical/datetime in this fixture.
_COLUMN_BOUNDS = {
    "amount": (0.0, 1000.0),
    "signup_date": ("2000-01-01", "2030-01-01"),
    "order_id": (0, 10_000),
}


def _run_up_to_contract(temp_sqlite_db: Path, tmp_path: Path):
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    metadata_dir = tmp_path / "metadata"

    run_discovery(adapter, manifest, config_path="config/project.yaml")
    discovery_data = json.loads((manifest.run_dir / "discovery.json").read_text())
    run_profiling(adapter, discovery_data, manifest, "discovery.json", sample_limit=100)
    profile_data = load_profile(manifest.output_path(PROFILE_FILENAME))
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
    return adapter, manifest, contract


def test_dp_adapter_trains_through_the_real_pipeline_and_records_privacy_spend(
    temp_sqlite_db: Path, tmp_path: Path
):
    adapter, manifest, contract = _run_up_to_contract(temp_sqlite_db, tmp_path)

    training_result = run_training_and_sampling(
        adapter, contract, manifest, "dataset_contract.json", sample_limit=100,
        num_rows_to_generate=5, seed=1, model_type="dp_gaussian_copula",
        model_kwargs={"epsilon_budget": 3.0, "delta_budget": 1e-5, "column_bounds": _COLUMN_BOUNDS},
    )
    assert training_result.is_success()

    training_report = load_training_report(Path(training_result.output_references[0]))
    for table_name, table_report in training_report["tables"].items():
        assert table_report["model_type"] == "dp_gaussian_copula"
        privacy_summary = table_report["fit_evidence"]["privacy_summary"]
        assert privacy_summary["total_epsilon_spent"] <= 3.0 + 1e-9
        assert privacy_summary["expenditures"]  # at least one real DP query was spent

    # Round-trip: load the saved model back and confirm it can still
    # generate, and that the persisted privacy_summary survives the
    # save/load cycle (an auditor reading the artifact later must see the
    # same evidence the training run itself recorded).
    for table_name, table_report in training_report["tables"].items():
        adapter_cls = get_synthesizer_adapter_class(table_report["model_type"])
        loaded = adapter_cls.load(table_report["model_path"])
        generated = loaded.sample(5, seed=1)
        assert len(generated) == 5
        assert loaded._privacy_summary["total_epsilon_spent"] == pytest.approx(
            table_report["fit_evidence"]["privacy_summary"]["total_epsilon_spent"]
        )
