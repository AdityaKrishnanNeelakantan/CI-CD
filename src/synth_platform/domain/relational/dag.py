"""Relational DAG: Tarjan SCC detection + level-based ordering.

Correctly distinguishes acyclic schemas (safe to order), self-references
(two-pass), and genuine multi-table cycles (rejected unless a deferred-key
strategy is configured). This replaces the legacy 'silently append cyclic
tables' behavior the audit flagged.

Phase 7 reconciliation note (master_repo consolidation): a second,
independently-written Tarjan-SCC implementation exists at
the Database Twin relational schema graph. That one
uses an iterative Tarjan (avoids Python's recursion limit on deep linear
FK chains, a real robustness edge this module doesn't have) and actually
SUPPORTS multi-table cycles end-to-end via two-pass SCC generation,
rather than rejecting them as "unsupported" the way this module's
`unsupported_cycles` does. THIS module's advantage is its typed, frozen
GraphPlan with explicit topological levels (useful for visualizing/
parallelizing generation waves) and its use of a real `ForeignKey` domain
type instead of loose dicts. Unifying the two (borrowing this module's
typed level-output shape while keeping the other's cyclic-schema support
and iterative-Tarjan safety) is deliberately left for Phase 8's
architecture refactor, not done here - Phase 7 is relocation only.
"""
from __future__ import annotations

from dataclasses import dataclass

from synth_platform.domain.schema.models import ForeignKey


@dataclass(frozen=True)
class GraphPlan:
    levels: list[list[str]]           # topological levels, parents first
    self_references: list[str]        # tables with a FK to themselves
    unsupported_cycles: list[list[str]]  # multi-table SCCs (size > 1)

    @property
    def order(self) -> list[str]:
        return [t for level in self.levels for t in level]


def _tarjan_scc(nodes: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    index = {}
    low = {}
    on_stack = set()
    stack: list[str] = []
    counter = [0]
    result: list[list[str]] = []

    def strong(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in edges.get(v, ()):  # v depends on w
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    for v in nodes:
        if v not in index:
            strong(v)
    return result


def build_graph_plan(tables: list[str], fks: list[ForeignKey]) -> GraphPlan:
    # child depends on parent
    deps: dict[str, set[str]] = {t: set() for t in tables}
    self_refs: list[str] = []
    for fk in fks:
        if fk.child_table not in deps or fk.parent_table not in deps:
            continue
        if fk.child_table == fk.parent_table:
            self_refs.append(fk.child_table)
        else:
            deps[fk.child_table].add(fk.parent_table)

    sccs = _tarjan_scc(tables, deps)
    unsupported = [c for c in sccs if len(c) > 1]

    # level assignment via Kahn on the condensation (ignoring self-loops)
    levels: list[list[str]] = []
    placed: set[str] = set()
    remaining = set(tables)
    # only meaningful when there are no unsupported cycles
    while remaining and not unsupported:
        ready = sorted(t for t in remaining if deps[t] <= placed)
        if not ready:
            break
        levels.append(ready)
        placed |= set(ready)
        remaining -= set(ready)
    return GraphPlan(levels=levels, self_references=sorted(set(self_refs)),
                     unsupported_cycles=unsupported)
