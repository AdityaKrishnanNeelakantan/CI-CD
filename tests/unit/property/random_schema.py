"""Generator of random VALID relational databases for property/fuzz testing.

Produces arbitrary acyclic schemas (random tables, mixed-type columns, a random
FK DAG, optional self-references) materialized as a RelationalDataset behind a
FrameConnector — i.e. exactly what the platform claims to support. Used to assert
invariants hold for ALL schemas, not just hand-built fixtures.

Deterministic given a seed, so any discovered failure is reproducible and can
become a regression test (mandate: every discovered bug becomes a regression).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from synth_platform.infrastructure.sources.frame_connector import FrameConnector
from synth_platform.application.dto.dataset import RelationalDataset
from synth_platform.domain.schema.models import (
    ColumnSchema, DatabaseSchema, ForeignKey, TableSchema,
)

_CATS = [list("abcde"[:k]) for k in (2, 3, 4, 5)]


def _random_column(rng, name):
    kind = rng.choice(["numeric", "categorical", "text"])
    if kind == "numeric":
        return ColumnSchema(name=name, physical_type="real", nullable=bool(rng.random() < 0.3)), kind
    if kind == "categorical":
        return ColumnSchema(name=name, physical_type="text", nullable=bool(rng.random() < 0.3)), kind
    return ColumnSchema(name=name, physical_type="text", nullable=True), kind


def random_dataset(seed: int, max_tables: int = 5, max_cols: int = 5,
                   min_rows: int = 20, max_rows: int = 120,
                   allow_self_ref: bool = True) -> RelationalDataset:
    rng = np.random.default_rng(seed)
    n_tables = int(rng.integers(1, max_tables + 1))
    table_names = [f"t{i}" for i in range(n_tables)]

    schemas: dict[str, TableSchema] = {}
    kinds: dict[str, dict[str, str]] = {}
    for i, tname in enumerate(table_names):
        pk = "id"
        cols = [ColumnSchema(name=pk, physical_type="integer", nullable=False)]
        ck = {}
        for j in range(int(rng.integers(1, max_cols + 1))):
            col, kind = _random_column(rng, f"c{j}")
            cols.append(col); ck[col.name] = kind
        schemas[tname] = TableSchema(name=tname, primary_key=pk,
                                     primary_key_confirmed=True, columns=cols, row_count=0)
        kinds[tname] = ck

    # random FK DAG: table i may reference an EARLIER table (acyclic by index),
    # plus optional self-reference on a nullable fk column.
    fks: list[ForeignKey] = []
    for i, tname in enumerate(table_names):
        if i > 0 and rng.random() < 0.6:
            parent = table_names[int(rng.integers(0, i))]
            fkcol = f"{parent}_fk"
            schemas[tname].columns.append(
                ColumnSchema(name=fkcol, physical_type="integer", nullable=False))
            fks.append(ForeignKey(parent_table=parent, parent_column="id",
                                  child_table=tname, child_column=fkcol,
                                  confirmed=True, evidence="random"))
        if allow_self_ref and rng.random() < 0.25:
            selfcol = "parent_id"
            schemas[tname].columns.append(
                ColumnSchema(name=selfcol, physical_type="integer", nullable=True))
            fks.append(ForeignKey(parent_table=tname, parent_column="id",
                                  child_table=tname, child_column=selfcol,
                                  confirmed=True, evidence="self_ref"))

    schema = DatabaseSchema(source_kind="random", tables=schemas, foreign_keys=fks)

    # materialize rows respecting FK integrity (parents first, by table index)
    frames: dict[str, pd.DataFrame] = {}
    pk_pool: dict[str, np.ndarray] = {}
    fk_by_child: dict[str, list[ForeignKey]] = {}
    for fk in fks:
        fk_by_child.setdefault(fk.child_table, []).append(fk)
    for tname in table_names:
        n = int(rng.integers(min_rows, max_rows + 1))
        data = {"id": np.arange(1, n + 1)}
        for cname, kind in kinds[tname].items():
            if kind == "numeric":
                data[cname] = rng.normal(rng.uniform(-5, 50), rng.uniform(1, 10), n)
            elif kind == "categorical":
                data[cname] = rng.choice(_CATS[int(rng.integers(0, len(_CATS)))], size=n)
            else:
                data[cname] = [f"x{int(v)}" for v in rng.integers(0, 1000, n)]
        for fk in fk_by_child.get(tname, []):
            if fk.parent_table == tname:      # self-ref: point at earlier rows or null
                vals = np.array([rng.integers(1, k) if k > 1 and rng.random() < 0.7 else None
                                 for k in range(1, n + 1)], dtype=object)
                data[fk.child_column] = vals
            else:
                pool = pk_pool[fk.parent_table]
                data[fk.child_column] = rng.choice(pool, size=n) if len(pool) else np.ones(n, int)
        df = pd.DataFrame(data)
        frames[tname] = df
        pk_pool[tname] = df["id"].to_numpy()

    return RelationalDataset(schema=schema, tables=frames).finalize_counts()


def random_connector(seed: int, **kw) -> FrameConnector:
    return FrameConnector(random_dataset(seed, **kw))
