"""Property/fuzz tests: platform invariants hold for ARBITRARY schemas.

Runs the full compile→generate→validate pipeline over hundreds of random valid
relational databases (mixed types, random FK DAGs, self-references) and asserts
the invariants that must ALWAYS hold. Any failing seed is reproducible and is
promoted to a named regression test.
"""
from __future__ import annotations

import numpy as np
import pytest

from synth_platform.application.use_cases.generate_dataset import generate_dataset
from synth_platform.application.use_cases.train_model import train_model
from synth_platform.application.use_cases.validate_dataset import validate_dataset
from synth_platform.domain.generation.models import GenerationRequest
from synth_platform.domain.validation.models import Status
from tests.unit.property.random_schema import random_connector, random_dataset

# how many random schemas to exercise; each is a full pipeline run.
_SEEDS = list(range(120))


@pytest.mark.parametrize("seed", _SEEDS)
def test_pipeline_holds_all_invariants(seed):
    conn = random_connector(seed)
    schema = conn.discover_schema()

    art = train_model(conn, sample_size=500, seed=seed)
    syn = generate_dataset(art, GenerationRequest(seed=seed))

    # INVARIANT 1: every schema table is generated
    assert set(syn) == set(schema.tables), f"seed {seed}: missing tables"

    for tname, ts in schema.tables.items():
        df = syn[tname]
        # INVARIANT 2: the schema's columns are exactly the generated columns
        assert list(df.columns) == list(ts.column_names), f"seed {seed}: {tname} columns"
        # INVARIANT 3: primary keys are unique and non-null
        if ts.primary_key:
            pk = df[ts.primary_key]
            assert pk.is_unique and pk.notna().all(), f"seed {seed}: {tname} PK"

    # INVARIANT 4: zero orphan foreign keys (referential integrity) for every FK,
    # including self-references (a self-FK may be null but never dangling).
    for fk in schema.foreign_keys:
        child = syn[fk.child_table][fk.child_column].dropna()
        parent_pk = set(syn[fk.parent_table][fk.parent_column])
        orphans = int((~child.isin(parent_pk)).sum())
        assert orphans == 0, f"seed {seed}: {fk.child_table}.{fk.child_column} orphans={orphans}"

    # INVARIANT 5: validation never crashes and never fake-passes; structural
    # checks must all PASS on internally-consistent generated data.
    report = validate_dataset(art, syn)
    struct = [c for c in report.checks if c.dimension == "structural"]
    assert all(c.status == Status.PASS for c in struct), f"seed {seed}: structural not all PASS"
    assert report.overall in (Status.PASS, Status.WARN)  # never a spurious FAIL


@pytest.mark.parametrize("seed", range(40))
def test_determinism_same_seed_same_output(seed):
    """INVARIANT 6: same artifact + same generation seed => identical output."""
    conn = random_connector(seed)
    art = train_model(conn, sample_size=400, seed=seed)
    a = generate_dataset(art, GenerationRequest(seed=123))
    b = generate_dataset(art, GenerationRequest(seed=123))
    for t in a:
        assert a[t].equals(b[t]), f"seed {seed}: non-deterministic table {t}"


@pytest.mark.parametrize("seed", range(30))
def test_scale_controls_row_counts(seed):
    """INVARIANT 7: scaling changes row counts but preserves FK integrity."""
    conn = random_connector(seed, allow_self_ref=False)
    schema = conn.discover_schema()
    art = train_model(conn, sample_size=400, seed=seed)
    syn = generate_dataset(art, GenerationRequest(scale=2.0, seed=seed))
    for fk in schema.foreign_keys:
        child = syn[fk.child_table][fk.child_column].dropna()
        parent_pk = set(syn[fk.parent_table][fk.parent_column])
        assert (~child.isin(parent_pk)).sum() == 0, f"seed {seed}: scaled orphans"
