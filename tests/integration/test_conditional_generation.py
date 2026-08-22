"""Conditional generation preserves cross-column correlation (W2 / RC-1, A-01).

The keystone: independent marginal sampling is forbidden. A deterministic
source relationship must be reproduced in the synthetic data, and a conditional
mean ordering must be preserved — neither is possible with per-column marginals.
"""
import os
import sqlite3

import numpy as np
import pytest

from synth_platform import GenerationRequest, SyntheticDataPlatform, TrainingRequest


@pytest.fixture()
def correlated_db(tmp_path):
    db = tmp_path / "corr.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, tier TEXT, region TEXT, "
              "balance REAL, currency TEXT)")
    rng = np.random.default_rng(0)
    tiers = ["bronze", "silver", "gold"]
    regions = ["us", "eu", "apac"]
    cur = {"us": "USD", "eu": "EUR", "apac": "SGD"}
    band = {"bronze": (50, 200), "silver": (500, 1500), "gold": (5000, 20000)}
    rows = []
    for i in range(1, 1501):
        t = tiers[rng.integers(0, 3)]
        r = regions[rng.integers(0, 3)]
        lo, hi = band[t]
        rows.append((i, t, r, float(rng.uniform(lo, hi)), cur[r]))
    c.executemany("INSERT INTO t VALUES(?,?,?,?,?)", rows)
    c.commit(); c.close()
    return str(db), cur


def test_deterministic_relationship_preserved(correlated_db):
    db, cur = correlated_db
    plat = SyntheticDataPlatform.from_settings()
    art = plat.train(TrainingRequest(source=f"sqlite:///{db}", sample_size=5000, seed=1))
    syn = plat.generate(art, GenerationRequest(root_table_rows={"t": 1500}, seed=1))["t"]
    # region -> currency is deterministic in the source; must hold in synthetic.
    consistent = sum(1 for _, r in syn.iterrows() if cur.get(r["region"]) == r["currency"])
    assert consistent / len(syn) >= 0.98   # marginal sampling would give ~0.33


def test_conditional_mean_ordering_preserved(correlated_db):
    db, _ = correlated_db
    plat = SyntheticDataPlatform.from_settings()
    art = plat.train(TrainingRequest(source=f"sqlite:///{db}", sample_size=5000, seed=2))
    syn = plat.generate(art, GenerationRequest(root_table_rows={"t": 1500}, seed=2))["t"]
    m = syn.groupby("tier")["balance"].mean()
    assert m["bronze"] < m["silver"] < m["gold"]   # tier->balance correlation kept


def test_dependency_inference_finds_edges(correlated_db):
    db, _ = correlated_db
    plat = SyntheticDataPlatform.from_settings()
    art = plat.train(TrainingRequest(source=f"sqlite:///{db}", sample_size=5000, seed=3))
    edges = {(e.parent, e.child) for e in art.data_profile.tables["t"].dependencies.edges}
    # inference is automatic + domain-free; it must discover both relationships
    assert ("region", "currency") in edges
    assert ("tier", "balance") in edges
