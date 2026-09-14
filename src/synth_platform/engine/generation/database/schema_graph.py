"""Builds a schema dependency graph from the approved dataset_contract's
foreign_keys, and a topological generation order from it - the relational
track's analogue of src/documents/layout_engine.py's reading order: this
decides WHAT ORDER tables must be generated in (every parent fully
generated before any child that references it), not how to fill them.

Self-referencing foreign keys (a table referencing its own primary key,
e.g. employee.manager_id -> employee.employee_id) are deliberately
excluded from the dependency graph rather than treated as a cycle:
resolving them properly requires incremental, row-by-row generation
within a single table (each row can only reference an *earlier* row),
which this batch-oriented generator does not attempt - see
src/relational/relational_generator.py's own docstring for how that
column is handled instead.

Cross-table cycles (A depends on B depends on A) are handled via SCC
condensation: Tarjan's algorithm finds strongly connected components,
which are condensed into a DAG.  Generation uses a two-pass strategy
within each SCC (sample non-FK values first, then resolve internal FKs).
The Tarjan implementation is iterative (explicit stack) to avoid Python's
recursion limit on deep linear FK chains.

This module supports cyclic schemas end-to-end via two-pass SCC
generation and uses iterative Tarjan traversal to avoid recursion-depth
failures on deep chains.
"""

from __future__ import annotations

from typing import Any


class CyclicSchemaError(Exception):
    """Raised when the schema's foreign keys form a cycle that cannot be
    resolved even with SCC condensation (should not occur for valid schemas).
    """


def build_schema_graph(dataset_contract: dict[str, Any]) -> dict[str, Any]:
    tables = dataset_contract["tables"]
    nodes = sorted(tables.keys())

    edges: list[dict[str, Any]] = []
    self_referencing_edges: list[dict[str, Any]] = []
    for child_table, table_entry in tables.items():
        for fk in table_entry.get("foreign_keys", []):
            parent_table = fk["references_table"]
            if parent_table not in tables:
                continue
            edge = {
                "parent": parent_table,
                "child": child_table,
                "parent_key": fk["references_column"],
                "child_key": fk["column"],
            }
            if parent_table == child_table:
                self_referencing_edges.append(edge)
            else:
                edges.append(edge)

    sccs = find_strongly_connected_components(nodes, edges)
    condensed_order = _condensed_topological_order(nodes, edges, sccs)
    generation_order = _flatten_scc_order(condensed_order)

    return {
        "nodes": nodes,
        "edges": edges,
        "self_referencing_edges": self_referencing_edges,
        "sccs": sccs,
        "condensed_order": condensed_order,
        "generation_order": generation_order,
    }


def find_strongly_connected_components(
    nodes: list[str], edges: list[dict[str, Any]]
) -> list[list[str]]:
    """Iterative Tarjan's algorithm - returns SCCs in reverse topological order."""
    adjacency: dict[str, list[str]] = {n: [] for n in nodes}
    for edge in edges:
        adjacency[edge["parent"]].append(edge["child"])

    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    sccs: list[list[str]] = []

    for start_node in nodes:
        if start_node in indices:
            continue
        tarjan_stack: list[tuple[str, list[str], int]] = [(start_node, adjacency[start_node], 0)]
        while tarjan_stack:
            node, neighbors, neighbor_idx = tarjan_stack[-1]
            if node not in indices:
                indices[node] = index_counter[0]
                lowlinks[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack.add(node)

            if neighbor_idx < len(neighbors):
                neighbor = neighbors[neighbor_idx]
                tarjan_stack[-1] = (node, neighbors, neighbor_idx + 1)
                if neighbor not in indices:
                    tarjan_stack.append((neighbor, adjacency[neighbor], 0))
                elif neighbor in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])
            else:
                tarjan_stack.pop()
                if lowlinks[node] == indices[node]:
                    scc: list[str] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.append(w)
                        if w == node:
                            break
                    sccs.append(sorted(scc))

    return sccs


def _condensed_topological_order(
    nodes: list[str], edges: list[dict[str, Any]], sccs: list[list[str]]
) -> list[list[str]]:
    """Return SCC groups in topological order (parents before children)."""
    node_to_scc_idx: dict[str, int] = {}
    for idx, scc in enumerate(sccs):
        for node in scc:
            node_to_scc_idx[node] = idx

    scc_count = len(sccs)
    in_degree = [0] * scc_count
    children_of: dict[int, list[int]] = {i: [] for i in range(scc_count)}
    seen_pairs: set[tuple[int, int]] = set()

    for edge in edges:
        parent_idx = node_to_scc_idx[edge["parent"]]
        child_idx = node_to_scc_idx[edge["child"]]
        if parent_idx == child_idx:
            continue
        pair = (parent_idx, child_idx)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        children_of[parent_idx].append(child_idx)
        in_degree[child_idx] += 1

    ready = sorted(i for i in range(scc_count) if in_degree[i] == 0)
    order: list[list[str]] = []
    while ready:
        current = ready.pop(0)
        order.append(sccs[current])
        for child_idx in sorted(children_of[current]):
            in_degree[child_idx] -= 1
            if in_degree[child_idx] == 0:
                ready.append(child_idx)
        ready.sort()

    if len(order) != scc_count:
        remaining = [sccs[i] for i in range(scc_count) if sccs[i] not in order]
        raise CyclicSchemaError(
            f"schema contains an unresolvable foreign-key cycle across {remaining}"
        )

    return order


def _flatten_scc_order(condensed_order: list[list[str]]) -> list[str]:
    """Flatten SCC groups into a single table generation order."""
    return [table for scc in condensed_order for table in scc]


def _topological_order(nodes: list[str], edges: list[dict[str, Any]]) -> list[str]:
    """Legacy acyclic topological sort - kept for backward compatibility in tests."""
    children_of: dict[str, list[str]] = {n: [] for n in nodes}
    in_degree: dict[str, int] = {n: 0 for n in nodes}

    seen_dependency_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        pair = (edge["parent"], edge["child"])
        if pair in seen_dependency_pairs:
            continue
        seen_dependency_pairs.add(pair)
        children_of[edge["parent"]].append(edge["child"])
        in_degree[edge["child"]] += 1

    ready = sorted(n for n in nodes if in_degree[n] == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for child in sorted(children_of[current]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
        ready.sort()

    if len(order) != len(nodes):
        remaining = sorted(set(nodes) - set(order))
        raise CyclicSchemaError(
            f"schema contains a foreign-key cycle across tables {remaining} - "
            "cyclic schemas are not supported by this generator"
        )

    return order
