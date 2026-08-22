from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from synth_platform.engine.common.database.core.run_manifest import RunManifest
from synth_platform.engine.generation.database.relational_service import (
    RelationalReportLoadError,
    load_relational_generation_report,
    run_relational_generation,
)
from synth_platform.engine.training.database.base import SynthesizerAdapter

pytestmark = pytest.mark.negative


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


def _acyclic_contract():
    return {
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


def _cyclic_contract():
    return {
        "tables": {
            "a": {"primary_key": ["id"], "foreign_keys": [{"column": "b_id", "references_table": "b", "references_column": "id"}], "columns": {}, "business_rules": []},
            "b": {"primary_key": ["id"], "foreign_keys": [{"column": "a_id", "references_table": "a", "references_column": "id"}], "columns": {}, "business_rules": []},
        }
    }


def test_relational_generation_never_overwrites_existing_output(tmp_path):
    adapters = {"customers": _FakeAdapter("customer_id"), "orders": _FakeAdapter("order_id")}
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    run_relational_generation(
        _acyclic_contract(), adapters, {"customers": 3, "orders": 5}, manifest, contract_reference="c.json", seed=1
    )
    with pytest.raises(RuntimeError):
        run_relational_generation(
            _acyclic_contract(), adapters, {"customers": 3, "orders": 5}, manifest, contract_reference="c.json", seed=1
        )


def test_relational_generation_fails_closed_on_write_error(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A disk-full/permission-denied failure while writing
    relational_generation_report.json must surface as a clean failed
    StageResult, not crash run_relational_generation() with a raw OSError.
    """
    import synth_platform.engine.generation.database.relational_service as relational_service_module

    real_dump = relational_service_module.json.dump

    def _raising_dump(obj, fp, *args, **kwargs):
        if Path(fp.name).name == "relational_generation_report.json":
            raise OSError("disk full")
        return real_dump(obj, fp, *args, **kwargs)

    adapters = {"customers": _FakeAdapter("customer_id"), "orders": _FakeAdapter("order_id")}
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    monkeypatch.setattr(relational_service_module.json, "dump", _raising_dump)

    result = run_relational_generation(
        _acyclic_contract(), adapters, {"customers": 3, "orders": 5}, manifest, contract_reference="c.json", seed=1
    )

    assert result.status == "failed"
    assert result.errors
    assert not manifest.output_path("relational_generation_report.json").exists()


def test_cyclic_schema_succeeds_via_scc_two_pass_resolution(tmp_path):
    """Cross-table FK cycles are no longer rejected: src/relational/schema_graph.py
    condenses them into a single SCC and src/relational/relational_generator.py
    resolves the internal FKs in a second pass (see both modules' docstrings).
    This replaces an older assertion that cyclic schemas failed closed, which
    predates that SCC-condensation support."""
    adapters = {"a": _FakeAdapter("id"), "b": _FakeAdapter("id")}
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_relational_generation(
        _cyclic_contract(), adapters, {"a": 3, "b": 3}, manifest, contract_reference="c.json", seed=1
    )
    assert result.status == "success"
    report = json.loads(manifest.output_path("relational_generation_report.json").read_text())
    table_a = pd.read_csv(report["tables"]["a"]["path"])
    table_b = pd.read_csv(report["tables"]["b"]["path"])
    assert set(table_a["b_id"]).issubset(set(table_b["id"]))
    assert set(table_b["a_id"]).issubset(set(table_a["id"]))



def test_missing_adapter_fails_closed(tmp_path):
    manifest = RunManifest.create(runs_dir=tmp_path / "runs")
    result = run_relational_generation(
        _acyclic_contract(), {"customers": _FakeAdapter("customer_id")}, {"customers": 3, "orders": 5},
        manifest, contract_reference="c.json", seed=1,
    )
    assert result.status == "failed"
    assert result.errors


def test_load_relational_generation_report_rejects_missing_file(tmp_path):
    with pytest.raises(RelationalReportLoadError):
        load_relational_generation_report(tmp_path / "does_not_exist.json")


def test_load_relational_generation_report_rejects_invalid_json(tmp_path):
    path = tmp_path / "relational_generation_report.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RelationalReportLoadError):
        load_relational_generation_report(path)


def test_load_relational_generation_report_rejects_missing_top_level_keys(tmp_path):
    path = tmp_path / "relational_generation_report.json"
    path.write_text(json.dumps({"generated_at": "x"}), encoding="utf-8")
    with pytest.raises(RelationalReportLoadError):
        load_relational_generation_report(path)
