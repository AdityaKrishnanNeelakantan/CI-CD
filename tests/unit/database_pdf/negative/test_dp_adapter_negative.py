from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.discovery.database.adapters.sqlite_adapter import SQLiteSourceAdapter
from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.training.database.adapters.dp_copula_adapter import DPCopulaSynthesizerAdapter, MissingColumnBoundsError
from synth_platform.engine.training.database.service import run_training_and_sampling

pytestmark = pytest.mark.negative


def test_missing_column_bounds_fails_closed_through_the_real_training_pipeline(
    temp_sqlite_db: Path, tmp_path: Path
):
    """The pipeline-level stage must fail closed (recorded StageResult,
    not an unhandled crash) when the caller forgets to supply bounds for
    a numeric column - the same guarantee every other stage in this
    project already provides for its own failure modes.
    """
    adapter = SQLiteSourceAdapter({"path": str(temp_sqlite_db)})
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")

    contract = {
        "tables": {
            "orders": {
                "primary_key": ["order_id"],
                "foreign_keys": [],
                "columns": {
                    "order_id": {"semantic_type": "identifier", "inference_status": "approved", "physical_type": "INTEGER"},
                    "amount": {"semantic_type": "numerical", "inference_status": "approved", "physical_type": "REAL"},
                },
                "business_rules": [],
            }
        }
    }

    with pytest.raises(MissingColumnBoundsError):
        run_training_and_sampling(
            adapter, contract, manifest, "dataset_contract.json", sample_limit=100,
            num_rows_to_generate=5, seed=1, model_type="dp_gaussian_copula",
            model_kwargs={"epsilon_budget": 1.0, "column_bounds": {}},
        )


def test_sample_before_fit_raises():
    adapter = DPCopulaSynthesizerAdapter(epsilon_budget=1.0)
    with pytest.raises(Exception):
        adapter.sample(5)


def test_save_before_fit_raises(tmp_path: Path):
    adapter = DPCopulaSynthesizerAdapter(epsilon_budget=1.0)
    with pytest.raises(Exception):
        adapter.save(tmp_path / "model.json")


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(Exception):
        DPCopulaSynthesizerAdapter.load(tmp_path / "does_not_exist.json")


def test_fit_raises_on_negative_or_zero_epsilon_budget():
    df = pd.DataFrame({"age": [20, 30, 40]})
    contract = {
        "primary_key": [],
        "columns": {"age": {"semantic_type": "numerical", "inference_status": "approved", "physical_type": "INTEGER"}},
    }
    adapter = DPCopulaSynthesizerAdapter(epsilon_budget=0.0, column_bounds={"age": (0, 100)})
    with pytest.raises(Exception):
        adapter.fit(df, "t", contract, seed=1)
