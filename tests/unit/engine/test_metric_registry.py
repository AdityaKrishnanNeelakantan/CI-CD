"""Metric registry is the single validation driver (W3 / A-05).

Correlation preservation is a real gate: PASS when structure is preserved, FAIL
when it is destroyed. This is the standing regression guard against a silent
return to marginal sampling.
"""
import numpy as np
import pandas as pd

from synth_platform.domain.profiling.models import (
    ColumnProfile, DataProfile, SensitivityClass, TableProfile,
)
from synth_platform.domain.relational.models import RelationalPlan
from synth_platform.domain.schema.models import DatabaseSchema, TableSchema
from synth_platform.domain.semantics.dependencies import DependencyEdge, DependencyGraph
from synth_platform.domain.validation.metric import ValidationContext
from synth_platform.domain.validation.models import Status
from synth_platform.engine.validation.metric_registry import default_metrics, run_metrics
from synth_platform.engine.validation.metrics import CorrelationPreservationMetric


def _artifact_with_edge():
    # minimal artifact carrying a learned region->currency edge of strength 1.0
    tprof = TableProfile(
        table_name="t", row_count=100,
        columns={
            "region": ColumnProfile(table_name="t", column_name="region",
                                    physical_type="categorical", semantic_type="category",
                                    nullable=False, cardinality=3, missing_rate=0.0,
                                    sensitivity=SensitivityClass.PUBLIC),
            "currency": ColumnProfile(table_name="t", column_name="currency",
                                      physical_type="categorical", semantic_type="category",
                                      nullable=False, cardinality=3, missing_rate=0.0,
                                      sensitivity=SensitivityClass.PUBLIC),
        },
        dependencies=DependencyGraph(edges=[
            DependencyEdge(parent="region", child="currency", strength=1.0)]))

    class _Art:  # duck-typed stand-in for SynthArtifact fields the metric reads
        schema_ = DatabaseSchema(source_kind="x",
                                 tables={"t": TableSchema(name="t", primary_key="id")})
        data_profile = DataProfile(tables={"t": tprof})
        relational_plan = RelationalPlan()
    return _Art()


def _preserved_frame():
    cur = {"us": "USD", "eu": "EUR", "ap": "SGD"}
    rng = np.random.default_rng(0)
    regions = [list(cur)[rng.integers(0, 3)] for _ in range(300)]
    return {"t": pd.DataFrame({"id": range(1, 301), "region": regions,
                               "currency": [cur[r] for r in regions]})}


def test_default_registry_has_core_metrics():
    names = {m.name for m in default_metrics()}
    assert {"structural", "marginal_fidelity", "correlation_preservation"} <= names


def test_correlation_metric_passes_when_preserved():
    art = _artifact_with_edge()
    checks = CorrelationPreservationMetric().evaluate(
        ValidationContext(artifact=art, tables=_preserved_frame()))
    assert checks and all(c.status == Status.PASS for c in checks)
    assert all(c.is_critical for c in checks)   # it gates the release


def test_correlation_metric_fails_when_destroyed():
    art = _artifact_with_edge()
    tables = _preserved_frame()
    # shuffle currency independently of region -> correlation destroyed
    tables["t"]["currency"] = np.random.default_rng(1).permutation(tables["t"]["currency"].values)
    checks = CorrelationPreservationMetric().evaluate(
        ValidationContext(artifact=art, tables=tables))
    assert checks and checks[0].status == Status.FAIL


def test_run_metrics_aggregates():
    art = _artifact_with_edge()
    checks = run_metrics(ValidationContext(artifact=art, tables=_preserved_frame()))
    dims = {c.dimension for c in checks}
    assert "structural" in dims and "fidelity" in dims
