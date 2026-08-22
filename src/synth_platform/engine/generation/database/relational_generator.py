"""Generates a full relational dataset across multiple tables, in the
schema graph's topological order, assigning real foreign keys by
sampling from already-generated parent primary keys.

src/synthesis/service.py's run_training_and_sampling already trains and
samples one table at a time (Checkpoint 4); this module is what turns
those independent per-table samples into a referentially-linked dataset,
matching this project's reference design's relational-generation
algorithm: generate parents first, then assign each child row to an
existing generated parent - never the reverse.

Self-referencing foreign keys (see src/relational/schema_graph.py) are
left exactly as each table's own SynthesizerAdapter generated them - this
generator only resolves *cross-table* foreign keys, since a genuine
self-reference would need incremental, row-by-row generation within a
single table, which this batch-oriented approach does not attempt.

For tables within the same strongly connected component (cyclic FK
schemas), a two-pass strategy is used: pass 1 samples all tables in the
SCC without cross-table FK assignment; pass 2 resolves internal FKs by
sampling from the generated parent keys within the same SCC.

Child row counts are caller-specified per table, not learned from the
source's real per-parent cardinality - a data-driven "N children per
parent" model is a real refinement this version does not implement (see
this module's own test suite for the exact, disclosed behaviour: FK
values are assigned by uniform random choice among generated parent
keys, not weighted by any learned distribution).
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from synth_platform.engine.training.database.base import SynthesizerAdapter


class RelationalGenerationError(Exception):
    """Raised when a relational dataset cannot be generated as requested."""


def generate_relational_dataset(
    schema_graph: dict[str, Any],
    adapters_by_table: dict[str, SynthesizerAdapter],
    row_counts_by_table: dict[str, int],
    seed: int | None = None,
    category_overrides_by_table: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, pd.DataFrame]:
    """category_overrides_by_table (table_name -> column_name ->
    {category: weight}) is forwarded to each table's own
    adapter.sample() - see src/synthesis/adapters/category_rebalancing.py.
    An adapter that doesn't support it (e.g. the SDV-wrapped one) raises
    its own TypeError rather than silently ignoring the request; this
    function does not swallow that.
    """
    missing_adapters = set(schema_graph["nodes"]) - set(adapters_by_table)
    if missing_adapters:
        raise RelationalGenerationError(f"no adapter provided for tables: {sorted(missing_adapters)}")
    missing_counts = set(schema_graph["nodes"]) - set(row_counts_by_table)
    if missing_counts:
        raise RelationalGenerationError(f"no row count requested for tables: {sorted(missing_counts)}")

    category_overrides_by_table = category_overrides_by_table or {}
    rng = random.Random(seed)
    tables: dict[str, pd.DataFrame] = {}

    edges_by_child: dict[str, list[dict[str, Any]]] = {}
    for edge in schema_graph["edges"]:
        edges_by_child.setdefault(edge["child"], []).append(edge)

    condensed_order = schema_graph.get("condensed_order")
    if condensed_order is None:
        condensed_order = [[t] for t in schema_graph["generation_order"]]

    scc_table_set: dict[frozenset[str], frozenset[str]] = {}
    for scc in condensed_order:
        scc_table_set[frozenset(scc)] = frozenset(scc)

    for scc in condensed_order:
        scc_set = frozenset(scc)

        for table_name in scc:
            adapter = adapters_by_table[table_name]
            num_rows = row_counts_by_table[table_name]
            table_overrides = category_overrides_by_table.get(table_name)
            if table_overrides:
                generated = adapter.sample(num_rows, seed=seed, category_overrides=table_overrides).reset_index(drop=True)
            else:
                generated = adapter.sample(num_rows, seed=seed).reset_index(drop=True)
            tables[table_name] = generated

        _assign_cross_scc_foreign_keys(tables, edges_by_child, scc_set, rng)
        _assign_intra_scc_foreign_keys(tables, edges_by_child, scc_set, rng)

    return tables


def _assign_cross_scc_foreign_keys(
    tables: dict[str, pd.DataFrame],
    edges_by_child: dict[str, list[dict[str, Any]]],
    current_scc: frozenset[str],
    rng: random.Random,
) -> None:
    """Assign FKs from parent tables outside the current SCC."""
    for child_table in current_scc:
        for edge in edges_by_child.get(child_table, []):
            if edge["parent"] in current_scc:
                continue
            parent_df = tables[edge["parent"]]
            parent_key_values = parent_df[edge["parent_key"]].tolist()
            if not parent_key_values:
                raise RelationalGenerationError(
                    f"cannot assign {child_table}.{edge['child_key']}: parent table "
                    f"{edge['parent']!r} generated zero rows"
                )
            num_rows = len(tables[child_table])
            tables[child_table][edge["child_key"]] = [
                rng.choice(parent_key_values) for _ in range(num_rows)
            ]


def _assign_intra_scc_foreign_keys(
    tables: dict[str, pd.DataFrame],
    edges_by_child: dict[str, list[dict[str, Any]]],
    current_scc: frozenset[str],
    rng: random.Random,
) -> None:
    """Pass 2: resolve FKs between tables within the same SCC."""
    for child_table in current_scc:
        for edge in edges_by_child.get(child_table, []):
            if edge["parent"] not in current_scc:
                continue
            parent_df = tables[edge["parent"]]
            parent_key_values = parent_df[edge["parent_key"]].tolist()
            if not parent_key_values:
                raise RelationalGenerationError(
                    f"cannot assign {child_table}.{edge['child_key']}: parent table "
                    f"{edge['parent']!r} generated zero rows"
                )
            num_rows = len(tables[child_table])
            tables[child_table][edge["child_key"]] = [
                rng.choice(parent_key_values) for _ in range(num_rows)
            ]


def compute_fk_validity(tables: dict[str, pd.DataFrame], schema_graph: dict[str, Any]) -> dict[str, Any]:
    """FKValidity = (#child rows whose FK exists in the parent's PK) / (#child rows),
    required to equal 1.0 - see this project's reference design's relational
    generation section. Computed and reported explicitly rather than only
    assumed, so a future refactor that breaks the guarantee is caught here,
    not discovered downstream.
    """
    edge_results: dict[str, Any] = {}
    for edge in schema_graph["edges"]:
        child_df = tables[edge["child"]]
        parent_df = tables[edge["parent"]]
        parent_keys = set(parent_df[edge["parent_key"]])
        child_values = child_df[edge["child_key"]]
        valid_count = int(child_values.isin(parent_keys).sum())
        total_count = len(child_values)
        edge_key = f"{edge['child']}.{edge['child_key']}->{edge['parent']}.{edge['parent_key']}"
        edge_results[edge_key] = {
            "valid_count": valid_count,
            "total_count": total_count,
            "fk_validity": round(valid_count / total_count, 6) if total_count else 1.0,
        }

    overall_fk_validity = min((r["fk_validity"] for r in edge_results.values()), default=1.0)
    return {"edges": edge_results, "overall_fk_validity": overall_fk_validity}
