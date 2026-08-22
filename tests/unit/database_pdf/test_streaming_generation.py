"""Unit tests for streaming synthesizer batches and ParentKeyStore."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synth_platform.engine.generation.database.key_store import ParentKeyStore
from synth_platform.engine.training.database.adapters.context_generators import generate_identifier_values
from synth_platform.engine.training.database.base import SynthesizerAdapter

pytestmark = pytest.mark.unit


class _OffsetAwareAdapter(SynthesizerAdapter):
    model_type = "fake_offset"
    file_extension = ".json"
    serialization_format = "json"

    def __init__(self, pk_column: str = "id") -> None:
        self._pk_column = pk_column

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
        return pd.DataFrame({self._pk_column: [f"{self._pk_column}-{start + i}" for i in range(num_rows)]})

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()


def test_iter_sample_totals_and_seeds():
    adapter = _OffsetAwareAdapter("customer_id")
    batches = list(adapter.iter_sample(25, batch_size=10, seed=7))
    assert [len(b) for b in batches] == [10, 10, 5]
    all_ids = pd.concat(batches, ignore_index=True)["customer_id"].tolist()
    assert len(all_ids) == 25
    assert len(set(all_ids)) == 25  # sequential IDs unique across batches

    again = list(adapter.iter_sample(25, batch_size=10, seed=7))
    assert [b["customer_id"].tolist() for b in batches] == [b["customer_id"].tolist() for b in again]


def test_generate_identifier_values_respects_row_offset():
    rng = np.random.default_rng(1)
    first = generate_identifier_values(
        3,
        column_name="cust_id",
        format_spec={"kind": "segmented", "separator": "-", "segments": [
            {"type": "literal", "value": "cust"},
            {"type": "numeric", "width": 3},
        ], "start_at": 1},
        sequential=True,
        rng=rng,
        row_offset=0,
    )
    second = generate_identifier_values(
        3,
        column_name="cust_id",
        format_spec={"kind": "segmented", "separator": "-", "segments": [
            {"type": "literal", "value": "cust"},
            {"type": "numeric", "width": 3},
        ], "start_at": 1},
        sequential=True,
        rng=rng,
        row_offset=3,
    )
    assert first == ["cust-001", "cust-002", "cust-003"]
    assert second == ["cust-004", "cust-005", "cust-006"]
    assert not set(first) & set(second)


def test_parent_key_store_roundtrip(tmp_path: Path):
    import random

    store_path = tmp_path / "keys.sqlite"
    with ParentKeyStore(store_path) as store:
        inserted = store.register_keys("customers", "customer_id", [1, 2, 3, 2])
        assert inserted == 3
        assert store.key_set_count("customers", "customer_id") == 3
        rng = random.Random(0)
        sampled = store.sample_keys("customers", "customer_id", 20, rng)
        assert len(sampled) == 20
        assert set(sampled) <= {1, 2, 3}
        assert store.contains("customers", "customer_id", 2)
        assert not store.contains("customers", "customer_id", 99)
        valid, total = store.contains_all("customers", "customer_id", [1, 99, 2])
        assert (valid, total) == (2, 3)
