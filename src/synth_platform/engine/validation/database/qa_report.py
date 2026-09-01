"""Combines integrity, fidelity, and business-rule evidence into one
unified QA report for a relationally-generated synthetic dataset - never
one combined score. This project's reference design explicitly warns
against treating utility and privacy as one combined metric, and the
same reasoning applies to integrity vs fidelity here: a dataset with
perfect FK validity but poor fidelity, or vice versa, needs to be
visible as two separate numbers, not averaged into something that hides
which one is wrong.

Fidelity is checked only against the already-privacy-sanitized companion
fields src/profiling/profiler.py computes (generation_lower_bound/
upper_bound, safe_category_frequencies) - never the exact minimum/
maximum/category_frequencies, even though this stage runs inside the
same protected training environment where those exact fields are
technically still available on the same profile object. One rule,
applied everywhere a profile is read, is easier to trust than "safe
except when it's convenient not to be."
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from synth_platform.engine.common.database.core.scaling import DEFAULT_CHUNK_SIZE


def _check_primary_key_uniqueness(df: pd.DataFrame, primary_key: list[str]) -> dict[str, Any]:
    if not primary_key or not all(col in df.columns for col in primary_key):
        return {"checked": False, "reason": "primary_key column(s) not present in generated table"}
    duplicate_count = int(df.duplicated(subset=primary_key).sum())
    return {
        "checked": True,
        "row_count": len(df),
        "duplicate_count": duplicate_count,
        "is_unique": duplicate_count == 0,
    }


def check_primary_key_uniqueness_from_path(
    path: str | Path,
    primary_key: list[str],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Exact PK uniqueness via chunked CSV reads into a temp SQLite UNIQUE table.

    Bounded RAM: O(chunk_size) for the DataFrame slice plus the SQLite index of
    distinct PK tuples (disk-backed). Hard PK checks remain exact — not approximate.
    """
    csv_path = Path(path)
    if not primary_key:
        return {"checked": False, "reason": "no primary_key configured"}
    if not csv_path.is_file():
        return {"checked": False, "reason": f"generated table CSV not found: {csv_path}"}

    # Peek header for column presence without loading the body.
    header = pd.read_csv(csv_path, nrows=0)
    if not all(col in header.columns for col in primary_key):
        return {"checked": False, "reason": "primary_key column(s) not present in generated table"}

    row_count = 0
    duplicate_count = 0
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "pk_check.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            cols_sql = ", ".join(f"c{i} TEXT" for i in range(len(primary_key)))
            conn.execute(
                f"CREATE TABLE pk_tuples ({cols_sql}, PRIMARY KEY ({', '.join(f'c{i}' for i in range(len(primary_key)))}))"
            )
            placeholders = ", ".join("?" for _ in primary_key)
            insert_sql = f"INSERT OR IGNORE INTO pk_tuples VALUES ({placeholders})"
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=list(primary_key)):
                rows = []
                for record in chunk.itertuples(index=False, name=None):
                    rows.append(
                        tuple(
                            "" if v is None or (isinstance(v, float) and v != v) else str(v)
                            for v in record
                        )
                    )
                before = conn.execute("SELECT COUNT(*) FROM pk_tuples").fetchone()[0]
                conn.executemany(insert_sql, rows)
                after = conn.execute("SELECT COUNT(*) FROM pk_tuples").fetchone()[0]
                inserted = after - before
                duplicate_count += len(rows) - inserted
                row_count += len(rows)
            conn.commit()
        finally:
            conn.close()

    return {
        "checked": True,
        "row_count": row_count,
        "duplicate_count": int(duplicate_count),
        "is_unique": duplicate_count == 0,
        "method": "streaming_sqlite",
    }


def _check_numeric_fidelity(series: pd.Series, column_profile: dict[str, Any]) -> dict[str, Any] | None:
    lower = column_profile.get("generation_lower_bound")
    upper = column_profile.get("generation_upper_bound")
    if lower is None or upper is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    within_bounds = int(((numeric >= lower) & (numeric <= upper)).sum())
    total = len(numeric)
    return {
        "check": "bounded_range",
        "generation_lower_bound": lower,
        "generation_upper_bound": upper,
        "within_bounds_count": within_bounds,
        "total_count": total,
        "within_bounds_rate": round(within_bounds / total, 6) if total else 1.0,
    }


def _check_category_fidelity(series: pd.Series, column_profile: dict[str, Any]) -> dict[str, Any] | None:
    safe_frequencies = column_profile.get("safe_category_frequencies")
    if not safe_frequencies:
        return None
    known_categories = set(safe_frequencies.keys())
    observed = series.astype(str)
    matched = int(observed.isin(known_categories).sum())
    total = len(observed)
    return {
        "check": "known_vocabulary",
        "known_category_count": len(known_categories),
        "matched_count": matched,
        "total_count": total,
        "matched_rate": round(matched / total, 6) if total else 1.0,
    }


def _accumulate_fidelity(existing: dict[str, Any] | None, piece: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return dict(piece)
    if piece["check"] == "bounded_range":
        within = int(existing.get("within_bounds_count", 0)) + int(piece.get("within_bounds_count", 0))
        total = int(existing.get("total_count", 0)) + int(piece.get("total_count", 0))
        existing["within_bounds_count"] = within
        existing["total_count"] = total
        existing["within_bounds_rate"] = round(within / total, 6) if total else 1.0
        return existing
    matched = int(existing.get("matched_count", 0)) + int(piece.get("matched_count", 0))
    total = int(existing.get("total_count", 0)) + int(piece.get("total_count", 0))
    existing["matched_count"] = matched
    existing["total_count"] = total
    existing["matched_rate"] = round(matched / total, 6) if total else 1.0
    return existing


def build_qa_report(
    tables: dict[str, pd.DataFrame],
    dataset_contract: dict[str, Any],
    fk_validity: dict[str, Any],
    constraint_reports: dict[str, Any],
    reference_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_key_checks = {
        table_name: _check_primary_key_uniqueness(
            df, dataset_contract["tables"].get(table_name, {}).get("primary_key", [])
        )
        for table_name, df in tables.items()
    }
    integrity = {"fk_validity": fk_validity, "primary_key_checks": primary_key_checks}

    fidelity: dict[str, Any] = {}
    if reference_profile is not None:
        for table_name, df in tables.items():
            profile_columns = reference_profile.get("tables", {}).get(table_name, {}).get("columns", {})
            table_fidelity: dict[str, Any] = {}
            for column_name, column_profile in profile_columns.items():
                if column_name not in df.columns:
                    continue
                check = _check_numeric_fidelity(df[column_name], column_profile)
                if check is None:
                    check = _check_category_fidelity(df[column_name], column_profile)
                if check is not None:
                    table_fidelity[column_name] = check
            if table_fidelity:
                fidelity[table_name] = table_fidelity

    pk_checks_passed = all(
        check.get("is_unique", True) for check in primary_key_checks.values() if check.get("checked")
    )
    hard_checks_passed = fk_validity["overall_fk_validity"] == 1.0 and pk_checks_passed

    return {
        "integrity": integrity,
        "fidelity": fidelity,
        "business_rules": constraint_reports,
        "hard_checks_passed": hard_checks_passed,
    }


def build_qa_report_from_paths(
    table_entries: dict[str, dict[str, Any]],
    dataset_contract: dict[str, Any],
    fk_validity: dict[str, Any],
    constraint_reports: dict[str, Any],
    reference_profile: dict[str, Any] | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Path-based QA: exact PK uniqueness + reused FK validity; chunked fidelity.

    Does not load full tables into memory. FK validity is taken from the
    relational generation report (already exact when computed via the key store).
    """
    primary_key_checks = {
        table_name: check_primary_key_uniqueness_from_path(
            entry["path"],
            dataset_contract["tables"].get(table_name, {}).get("primary_key", []),
            chunk_size=chunk_size,
        )
        for table_name, entry in table_entries.items()
    }
    integrity = {"fk_validity": fk_validity, "primary_key_checks": primary_key_checks}

    fidelity: dict[str, Any] = {}
    if reference_profile is not None:
        for table_name, entry in table_entries.items():
            profile_columns = reference_profile.get("tables", {}).get(table_name, {}).get("columns", {})
            if not profile_columns:
                continue
            path = Path(entry["path"])
            header = pd.read_csv(path, nrows=0)
            usable = [c for c in profile_columns if c in header.columns]
            if not usable:
                continue
            table_fidelity: dict[str, Any] = {}
            for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=usable):
                for column_name in usable:
                    column_profile = profile_columns[column_name]
                    check = _check_numeric_fidelity(chunk[column_name], column_profile)
                    if check is None:
                        check = _check_category_fidelity(chunk[column_name], column_profile)
                    if check is None:
                        continue
                    table_fidelity[column_name] = _accumulate_fidelity(
                        table_fidelity.get(column_name), check
                    )
            if table_fidelity:
                fidelity[table_name] = table_fidelity

    pk_checks_passed = all(
        check.get("is_unique", True) for check in primary_key_checks.values() if check.get("checked")
    )
    hard_checks_passed = fk_validity["overall_fk_validity"] == 1.0 and pk_checks_passed

    return {
        "integrity": integrity,
        "fidelity": fidelity,
        "business_rules": constraint_reports,
        "hard_checks_passed": hard_checks_passed,
        "validation_mode": "streaming_paths",
    }
