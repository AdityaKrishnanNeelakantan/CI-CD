"""Streaming relational generation: O(batch_size) working memory.

Walks the same SCC / condensed order as :func:`generate_relational_dataset`,
but samples via ``adapter.iter_sample``, appends CSV sinks, and resolves FKs
from a disk-backed :class:`ParentKeyStore` instead of retaining full frames.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from synth_platform.engine.common.database.core.scaling import DEFAULT_CHUNK_SIZE
from synth_platform.engine.generation.database.constraint_engine import (
    apply_derived_fields,
    check_constraints,
    repair_violations,
)
from synth_platform.engine.generation.database.key_store import ParentKeyStore
from synth_platform.engine.generation.database.relational_generator import RelationalGenerationError
from synth_platform.engine.training.database.base import SynthesizerAdapter


class CsvAppendSink:
    """Append DataFrame batches to a CSV, writing the header once."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self._header_written = False
        self.row_count = 0

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        frame.to_csv(
            self.path,
            mode="a",
            header=not self._header_written,
            index=False,
        )
        self._header_written = True
        self.row_count += len(frame)


def _primary_key_columns(dataset_contract: dict[str, Any], table_name: str) -> list[str]:
    pk = dataset_contract.get("tables", {}).get(table_name, {}).get("primary_key") or []
    return list(pk)


def _merge_constraint_reports(accumulated: dict[str, Any] | None, report: dict[str, Any]) -> dict[str, Any]:
    if accumulated is None:
        return {
            "total_violations": int(report.get("total_violations", 0)),
            "constraints": list(report.get("constraints", [])),
        }
    accumulated["total_violations"] = int(accumulated.get("total_violations", 0)) + int(
        report.get("total_violations", 0)
    )
    # Keep the latest per-constraint detail list if present.
    if report.get("constraints"):
        accumulated["constraints"] = list(report["constraints"])
    return accumulated


def _assign_fk_batch(
    batch: pd.DataFrame,
    edge: dict[str, Any],
    key_store: ParentKeyStore,
    rng: random.Random,
) -> None:
    parent = edge["parent"]
    parent_key = edge["parent_key"]
    child_key = edge["child_key"]
    if key_store.key_set_count(parent, parent_key) == 0:
        raise RelationalGenerationError(
            f"cannot assign {edge['child']}.{child_key}: parent table "
            f"{parent!r} generated zero rows"
        )
    batch[child_key] = key_store.sample_keys(parent, parent_key, len(batch), rng)


def _rewrite_csv_assign_fk(
    csv_path: Path,
    edge: dict[str, Any],
    key_store: ParentKeyStore,
    rng: random.Random,
    *,
    chunk_size: int,
) -> None:
    """Stream-rewrite a child CSV assigning one FK column from the key store."""
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    sink = CsvAppendSink(tmp_path)
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        _assign_fk_batch(chunk, edge, key_store, rng)
        sink.append(chunk)
    tmp_path.replace(csv_path)


def compute_fk_validity_streaming(
    table_paths: dict[str, Path],
    schema_graph: dict[str, Any],
    key_store: ParentKeyStore,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Exact FK validity via key store + chunked child CSV reads."""
    edge_results: dict[str, Any] = {}
    for edge in schema_graph["edges"]:
        child_path = table_paths[edge["child"]]
        parent = edge["parent"]
        parent_key = edge["parent_key"]
        child_key = edge["child_key"]
        valid_count = 0
        total_count = 0
        for chunk in pd.read_csv(child_path, chunksize=chunk_size, usecols=[child_key]):
            values = chunk[child_key].tolist()
            v, t = key_store.contains_all(parent, parent_key, values)
            valid_count += v
            total_count += t
        edge_key = f"{edge['child']}.{child_key}->{parent}.{parent_key}"
        edge_results[edge_key] = {
            "valid_count": valid_count,
            "total_count": total_count,
            "fk_validity": round(valid_count / total_count, 6) if total_count else 1.0,
        }
    overall_fk_validity = min((r["fk_validity"] for r in edge_results.values()), default=1.0)
    return {"edges": edge_results, "overall_fk_validity": overall_fk_validity}


def generate_relational_dataset_streaming(
    schema_graph: dict[str, Any],
    adapters_by_table: dict[str, SynthesizerAdapter],
    row_counts_by_table: dict[str, int],
    dataset_contract: dict[str, Any],
    samples_dir: Path,
    key_store_path: Path,
    *,
    batch_size: int,
    seed: int | None = None,
    category_overrides_by_table: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    """Generate linked tables to CSV with O(batch_size) working memory.

    Returns report metadata compatible with ``relational_generation_report.json``
    (``fk_validity``, ``constraint_reports``, ``tables`` paths/row counts).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    missing_adapters = set(schema_graph["nodes"]) - set(adapters_by_table)
    if missing_adapters:
        raise RelationalGenerationError(f"no adapter provided for tables: {sorted(missing_adapters)}")
    missing_counts = set(schema_graph["nodes"]) - set(row_counts_by_table)
    if missing_counts:
        raise RelationalGenerationError(f"no row count requested for tables: {sorted(missing_counts)}")

    category_overrides_by_table = category_overrides_by_table or {}
    rng = random.Random(seed)
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    edges_by_child: dict[str, list[dict[str, Any]]] = {}
    for edge in schema_graph["edges"]:
        edges_by_child.setdefault(edge["child"], []).append(edge)

    condensed_order = schema_graph.get("condensed_order")
    if condensed_order is None:
        condensed_order = [[t] for t in schema_graph["generation_order"]]

    table_paths: dict[str, Path] = {}
    row_counts_out: dict[str, int] = {}
    constraint_reports: dict[str, Any] = {}
    warnings: list[str] = []

    with ParentKeyStore(key_store_path) as key_store:
        for scc in condensed_order:
            scc_set = frozenset(scc)

            # Pass 1: stream each table; assign only cross-SCC FKs.
            for table_name in scc:
                adapter = adapters_by_table[table_name]
                num_rows = int(row_counts_by_table[table_name])
                table_overrides = category_overrides_by_table.get(table_name)
                csv_path = samples_dir / f"{table_name}.csv"
                sink = CsvAppendSink(csv_path)
                pk_cols = _primary_key_columns(dataset_contract, table_name)

                table_contract = dataset_contract.get("tables", {}).get(table_name, {})
                business_rules = table_contract.get("business_rules", [])
                derived_fields = [r for r in business_rules if r.get("type") == "derived"]
                constraints = [r for r in business_rules if r.get("type") == "constraint"]
                merged_report: dict[str, Any] | None = None

                for batch in adapter.iter_sample(
                    num_rows,
                    batch_size=batch_size,
                    seed=seed,
                    category_overrides=table_overrides,
                ):
                    batch = batch.reset_index(drop=True)
                    for edge in edges_by_child.get(table_name, []):
                        if edge["parent"] in scc_set:
                            continue  # intra-SCC: deferred to pass 2
                        _assign_fk_batch(batch, edge, key_store, rng)

                    if derived_fields:
                        batch = apply_derived_fields(batch, derived_fields)
                    if constraints:
                        report = check_constraints(batch, constraints)
                        batch = repair_violations(batch, constraints, report)
                        merged_report = _merge_constraint_reports(merged_report, report)

                    sink.append(batch)
                    for pk_col in pk_cols:
                        if pk_col in batch.columns:
                            key_store.register_keys(table_name, pk_col, batch[pk_col].tolist())

                if merged_report is not None:
                    constraint_reports[table_name] = merged_report
                    if merged_report.get("total_violations"):
                        warnings.append(
                            f"{table_name}: {merged_report['total_violations']} constraint violation(s) repaired"
                        )

                table_paths[table_name] = csv_path
                row_counts_out[table_name] = sink.row_count

            # Pass 2: resolve intra-SCC FKs by streaming rewrite.
            for child_table in scc:
                for edge in edges_by_child.get(child_table, []):
                    if edge["parent"] not in scc_set:
                        continue
                    _rewrite_csv_assign_fk(
                        table_paths[child_table],
                        edge,
                        key_store,
                        rng,
                        chunk_size=batch_size,
                    )

        fk_validity = compute_fk_validity_streaming(
            table_paths, schema_graph, key_store, chunk_size=batch_size
        )

    return {
        "generation_order": schema_graph["generation_order"],
        "fk_validity": fk_validity,
        "constraint_reports": constraint_reports,
        "tables": {
            name: {"row_count": row_counts_out[name], "path": str(table_paths[name])}
            for name in schema_graph["generation_order"]
            if name in table_paths
        },
        "warnings": warnings,
        "batch_size": batch_size,
        "streaming": True,
    }
