"""Architecture contract (ARCHITECTURE_AUDIT A-06 / RC-10).

Enforces the layer law with a pure-AST scan (no external linter dependency), so
the boundaries are guarded by a standing test before the W1/W2 refactors land.

Canonical dependency direction (from the mandate):
    interfaces -> application -> engine -> domain
Infrastructure implements application ports, so infrastructure may import application/domain,
but inner layers may never import infrastructure or interfaces. Domain must import
no I/O or dataframe libraries at all.
"""
from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "synth_platform"

# lower index = more inward. A module may import its own layer and any inner one.
INWARD_ORDER = ["domain", "engine", "application", "infrastructure", "interfaces"]
RANK = {name: i for i, name in enumerate(INWARD_ORDER)}

# domain must be pure: no dataframe/IO/UI libraries (pydantic + stdlib only).
FORBIDDEN_IN_DOMAIN = {"pandas", "numpy", "sqlalchemy", "streamlit", "fastapi",
                       "sqlite3", "pdfplumber", "safetensors", "celery"}


def _iter_modules():
    for f in SRC.rglob("*.py"):
        parts = f.relative_to(SRC).parts
        layer = parts[0] if parts and parts[0] in RANK else None
        if layer is not None:
            yield f, layer


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_domain_is_pure():
    offenders = []
    for f, layer in _iter_modules():
        if layer != "domain":
            continue
        for mod in _imports(f):
            if mod.split(".")[0] in FORBIDDEN_IN_DOMAIN:
                offenders.append(f"{f.relative_to(SRC)} imports {mod}")
    assert not offenders, "domain purity violated:\n" + "\n".join(offenders)


def test_dependencies_point_inward():
    offenders = []
    for f, layer in _iter_modules():
        for mod in _imports(f):
            if not mod.startswith("synth_platform."):
                continue
            parts = mod.split(".")
            target = parts[1] if len(parts) > 1 and parts[1] in RANK else None
            if target is None:
                continue
            if RANK[target] > RANK[layer]:  # importing a MORE OUTWARD layer
                offenders.append(
                    f"{f.relative_to(SRC)} [{layer}] -> {mod} [{target}]")
    assert not offenders, (
        "inward-dependency rule violated (a layer imported a more outward one):\n"
        + "\n".join(offenders))


def test_inner_layers_never_import_infrastructure_or_interfaces():
    offenders = []
    for f, layer in _iter_modules():
        if layer not in ("application", "engine", "domain"):
            continue
        for mod in _imports(f):
            top = mod.split(".")
            if len(top) > 1 and top[0] == "synth_platform" and top[1] in ("infrastructure", "interfaces"):
                offenders.append(f"{f.relative_to(SRC)} [{layer}] -> {mod}")
    assert not offenders, (
        "inner layer imported infrastructure/interface (must depend on a port):\n"
        + "\n".join(offenders))
