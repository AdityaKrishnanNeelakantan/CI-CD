"""Compile-time privacy gate: transform before serialization (A-07).

Proves the invariant the mandate demands — no raw value from a non-public column
survives into the compiled artifact — AND that the verifier is a real gate that
fires when redaction is incomplete. The quasi-identifier-preserved bug found
while building this is pinned here as a regression.
"""
import sqlite3

import pytest

from synth_platform import SyntheticDataPlatform, TrainingRequest
from synth_platform.domain.privacy.classification import classify
from synth_platform.domain.privacy.models import (
    ColumnPolicy, PrivacyAction, PrivacyPolicy,
)
from synth_platform.domain.privacy.verifier import verify_source_free
from synth_platform.domain.profiling.models import (
    CategoricalStatistics, ColumnProfile, DataProfile, SensitivityClass, TableProfile,
)
from synth_platform.errors import PrivacyLeakError


def _people_db(tmp_path):
    db = tmp_path / "p.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE people(id INTEGER PRIMARY KEY, zipcode TEXT, city TEXT)")
    import random
    random.seed(1)
    zips = [f"9{n:04d}" for n in range(25)]          # 25 distinct -> quasi-identifier
    rows = [(i, random.choice(zips), random.choice(["Alpha", "Beta", "Gamma"]))
            for i in range(1, 600)]
    c.executemany("INSERT INTO people VALUES(?,?,?)", rows)
    c.commit(); c.close()
    return str(db), zips


def test_quasi_identifier_values_never_reach_artifact(tmp_path):
    db, zips = _people_db(tmp_path)
    art = SyntheticDataPlatform.from_settings().train(
        TrainingRequest(source=f"sqlite:///{db}", sample_size=5000, seed=1))
    blob = art.model_dump_json()
    # REGRESSION: previously all 25 raw zipcodes survived (quasi-id preserved bug)
    assert not any(z in blob for z in zips), "raw quasi-identifier leaked into artifact"
    # public business enum is still preserved (not over-redacted)
    assert all(x in blob for x in ["Alpha", "Beta", "Gamma"])


def test_quasi_identifier_is_bucketed_not_preserved():
    # the inverted-logic bug: high-cardinality quasi-id must be BUCKET, not PRESERVE
    prof = ColumnProfile(table_name="t", column_name="zip", physical_type="categorical",
                         semantic_type="category", nullable=False, cardinality=25,
                         missing_rate=0.0, sensitivity=SensitivityClass.QUASI_IDENTIFIER)
    assert classify(prof).action == PrivacyAction.BUCKET
    prof.cardinality = 3   # very low -> suppressed outright
    assert classify(prof).action == PrivacyAction.SUPPRESS


def test_verifier_fires_when_redaction_incomplete():
    cp = ColumnProfile(table_name="t", column_name="zip", physical_type="categorical",
                       semantic_type="category", nullable=False, cardinality=2,
                       missing_rate=0.0, sensitivity=SensitivityClass.QUASI_IDENTIFIER,
                       categorical=CategoricalStatistics(values=["90001", "90002"],
                                                         probabilities=[0.5, 0.5]))
    prof = DataProfile(tables={"t": TableProfile(table_name="t", row_count=10,
                                                 columns={"zip": cp})})
    policy = PrivacyPolicy(columns=[ColumnPolicy(table="t", column="zip",
                                                 action=PrivacyAction.BUCKET)])
    with pytest.raises(PrivacyLeakError):
        verify_source_free(prof, policy, {("t", "zip"): {"90001", "90002"}})


def test_direct_identifier_marginal_cleared(tmp_path):
    db = tmp_path / "e.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE u(id INTEGER PRIMARY KEY, email TEXT)")
    c.executemany("INSERT INTO u VALUES(?,?)",
                  [(i, f"real.person{i}@corp.example") for i in range(1, 200)])
    c.commit(); c.close()
    art = SyntheticDataPlatform.from_settings().train(
        TrainingRequest(source=f"sqlite:///{db}", sample_size=5000, seed=1))
    blob = art.model_dump_json()
    assert "real.person" not in blob   # no raw email survives
