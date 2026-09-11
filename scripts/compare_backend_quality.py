"""Compare current-platform and NeMo backend benchmark artifacts.

The script reads generated files from two benchmark directories and writes
plain JSON/Markdown evidence for manager review. It does not call an LLM.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}\b"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
BUSINESS_ID_RE = re.compile(r"\b(?:POL|CLM|ACC|CUST|TXN)[-_A-Z0-9]*\d+[A-Z0-9-]*\b", re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score current vs NeMo backend outputs.")
    parser.add_argument("--current-dir", default="build/verified_all_current")
    parser.add_argument("--nemo-dir", default="build/real_nemo_all_final")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="build/quality_comparison_final")
    args = parser.parse_args()

    current_dir = Path(args.current_dir)
    nemo_dir = Path(args.nemo_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "inputs": {
            "current_dir": str(current_dir),
            "nemo_dir": str(nemo_dir),
            "data_dir": str(data_dir),
        },
        "workflow_scores": {
            "schema": score_schema(current_dir, nemo_dir),
            "database": score_database(current_dir, nemo_dir),
            "pdf": score_pdf(current_dir, nemo_dir),
            "transcript": score_transcript(current_dir, nemo_dir, data_dir),
        },
        "setup": score_setup(nemo_dir),
    }
    results["status_tracker"] = {
        "Output quality scoring": "completed",
        "Privacy leakage scoring": "completed",
        "Guardrails setup": _setup_status(results["setup"]["guardrails"]),
        "Curator setup": _setup_status(results["setup"]["curator"]),
    }

    (out_dir / "quality_comparison_scored.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "quality_comparison_scored.md").write_text(render_markdown(results), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote: {out_dir / 'quality_comparison_scored.md'}")
    print(f"Wrote: {out_dir / 'quality_comparison_scored.json'}")


def read_report(base: Path) -> dict[str, Any]:
    return json.loads((base / "benchmark_report.json").read_text(encoding="utf-8"))


def latest_run(path: Path) -> Path:
    return max([entry for entry in path.iterdir() if entry.is_dir()], key=lambda entry: entry.stat().st_mtime)


def duplicate_rate(df: pd.DataFrame) -> float:
    return round(float(df.duplicated().mean()), 4) if len(df) else 0.0


def nonnull_rate(df: pd.DataFrame) -> float:
    return round(float(1 - df.isna().mean().mean()), 4) if len(df) else 0.0


def score_from_checks(*checks: float) -> int:
    bounded = [max(0.0, min(1.0, float(check))) for check in checks]
    return round(sum(bounded) / max(1, len(bounded)) * 100)


def score_schema(current_dir: Path, nemo_dir: Path) -> dict[str, Any]:
    current = schema_metrics(current_dir)
    nemo = schema_metrics(nemo_dir)
    return {"current": current, "nemo": nemo}


def schema_metrics(base: Path) -> dict[str, Any]:
    report = read_report(base)
    users = pd.read_csv(base / "schema" / "exports" / "users.csv")
    orders = pd.read_csv(base / "schema" / "exports" / "orders.csv")
    email_validity = users["email"].astype(str).map(lambda value: bool(EMAIL_RE.fullmatch(value))).mean()
    fk_validity = orders["user_id"].isin(set(users["user_id"])).mean()
    duplicate = (duplicate_rate(users) + duplicate_rate(orders)) / 2
    status_domain = orders["status"].astype(str).isin(
        {"pending", "shipped", "delivered", "canceled", "cancelled", "active", "review", "standard", "premium"}
    ).mean()
    quality_score = score_from_checks(
        bool(report["modes"]["schema"].get("hard_checks_passed")),
        email_validity,
        fk_validity,
        1 - duplicate,
        status_domain,
    )
    privacy_hits = int(EMAIL_RE.findall("\n".join(users["email"].astype(str)).lower()).__len__())
    return {
        "runtime_seconds": report["comparison"]["schema"]["seconds"],
        "quality_score": quality_score,
        "privacy_safety_score": 100 if privacy_hits == len(users) else 70,
        "hard_checks_passed": report["modes"]["schema"].get("hard_checks_passed"),
        "fk_validity": round(float(fk_validity), 4),
        "email_format_validity": round(float(email_validity), 4),
        "duplicate_rate": round(float(duplicate), 4),
        "status_domain_validity": round(float(status_domain), 4),
        "rows": int(len(users) + len(orders)),
    }


def score_database(current_dir: Path, nemo_dir: Path) -> dict[str, Any]:
    current = database_metrics(current_dir, "current")
    nemo = database_metrics(nemo_dir, "nemo")
    return {"current": current, "nemo": nemo}


def database_metrics(base: Path, backend: str) -> dict[str, Any]:
    report = read_report(base)
    run = latest_run(base / "database" / "runs")
    sample_dir = run / ("relational_samples" if backend == "current" else "generated_samples")
    tables = {path.stem: pd.read_csv(path) for path in sample_dir.glob("*.csv")}
    relational = json.loads((run / "relational_generation_report.json").read_text(encoding="utf-8"))
    fk_validity = float(relational.get("fk_validity", {}).get("overall_fk_validity", 0))
    duplicate = sum(duplicate_rate(df) for df in tables.values()) / max(1, len(tables))
    nonnull = sum(nonnull_rate(df) for df in tables.values()) / max(1, len(tables))
    overlaps = sensitive_value_overlap(source_database_tables(base), tables)
    overlap_count = sum(overlaps.values())
    quality_score = score_from_checks(fk_validity, 1 - duplicate, nonnull)
    privacy_safety_score = max(0, 100 - min(100, overlap_count * 4))
    return {
        "runtime_seconds": report["comparison"]["database"]["seconds"],
        "quality_score": quality_score,
        "privacy_safety_score": privacy_safety_score,
        "fk_validity": fk_validity,
        "duplicate_rate": round(float(duplicate), 4),
        "nonnull_rate": round(float(nonnull), 4),
        "sensitive_overlap_count": int(overlap_count),
        "sensitive_overlap_by_column": overlaps,
        "total_rows": int(sum(len(df) for df in tables.values())),
        "numeric_distribution": numeric_distribution(tables),
    }


def source_database_tables(base: Path) -> dict[str, pd.DataFrame]:
    source_db = base / "database" / "source.db"
    tables: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(source_db) as connection:
        names = pd.read_sql("select name from sqlite_master where type='table'", connection)["name"].tolist()
        for name in names:
            tables[name] = pd.read_sql(f'select * from "{name}"', connection)
    return tables


def sensitive_value_overlap(source: dict[str, pd.DataFrame], synthetic: dict[str, pd.DataFrame]) -> dict[str, int]:
    overlaps: dict[str, int] = {}
    for table_name, source_df in source.items():
        synthetic_df = synthetic.get(table_name)
        if synthetic_df is None:
            continue
        for column in source_df.columns:
            if column not in synthetic_df.columns:
                continue
            if not any(token in column.lower() for token in ("id", "email", "name", "phone", "account")):
                continue
            source_values = {str(value) for value in source_df[column].dropna() if str(value)}
            synthetic_values = {str(value) for value in synthetic_df[column].dropna() if str(value)}
            overlaps[f"{table_name}.{column}"] = len(source_values & synthetic_values)
    return overlaps


def numeric_distribution(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for table_name, df in tables.items():
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                metrics[f"{table_name}.{column}"] = {
                    "mean": round(float(df[column].mean()), 3),
                    "std": round(float(df[column].std(ddof=0)), 3),
                }
    return metrics


def score_pdf(current_dir: Path, nemo_dir: Path) -> dict[str, Any]:
    return {"current": pdf_metrics(current_dir), "nemo": pdf_metrics(nemo_dir)}


def pdf_metrics(base: Path) -> dict[str, Any]:
    report = read_report(base)
    doc_dir = latest_run(base / "pdf" / "runs") / "documents" / "manual_pdf"
    validation = json.loads((doc_dir / "document_validation_report.json").read_text(encoding="utf-8"))
    validation_report = validation.get("report", validation)
    values = json.loads((doc_dir / "document_synthetic_values.json").read_text(encoding="utf-8"))
    field_accuracy = float(validation_report.get("field_accuracy") or 0)
    overflow_failures = int(validation_report.get("overflow_failure_count") or 0)
    rendered_exists = (doc_dir / "rendered.pdf").exists()
    quality_score = score_from_checks(
        bool(validation_report.get("hard_checks_passed")),
        field_accuracy,
        1.0 if overflow_failures == 0 else 0.0,
        1.0 if rendered_exists else 0.0,
    )
    source_reference = validation.get("source_reference") or report["modes"]["pdf"].get("input_pdf")
    source_text = pdf_text(Path(source_reference)) if source_reference else ""
    rendered_text = pdf_text(doc_dir / "rendered.pdf")
    source_pii_values = regex_pii_values(source_text)
    generated_text = rendered_text + "\n" + json.dumps(values, default=str)
    exact_source_pii_replays = sum(1 for value in source_pii_values if value and value in generated_text)
    synthetic_pii_format_hits = count_regex_pii(generated_text)
    return {
        "runtime_seconds": report["comparison"]["pdf"]["seconds"],
        "quality_score": quality_score,
        "privacy_safety_score": 100 if exact_source_pii_replays == 0 else max(0, 100 - min(100, exact_source_pii_replays * 20)),
        "hard_checks_passed": validation_report.get("hard_checks_passed"),
        "field_accuracy": field_accuracy,
        "matched_fields": validation_report.get("matched_fields"),
        "total_fields": validation_report.get("total_fields"),
        "overflow_failure_count": overflow_failures,
        "rendered_pdf_exists": rendered_exists,
        "field_count": len(values.get("fields", {})),
        "table_count": len(values.get("tables", {})),
        "inline_span_count": sum(len(spans) for spans in values.get("inline_spans", {}).values()),
        "source_pii_value_count": len(source_pii_values),
        "exact_source_pii_replays": exact_source_pii_replays,
        "synthetic_pii_format_hits": synthetic_pii_format_hits,
    }


def score_transcript(current_dir: Path, nemo_dir: Path, data_dir: Path) -> dict[str, Any]:
    return {
        "current": transcript_metrics(current_dir, data_dir),
        "nemo": transcript_metrics(nemo_dir, data_dir),
    }


def transcript_metrics(base: Path, data_dir: Path) -> dict[str, Any]:
    report = read_report(base)
    rows = json.loads((base / "transcript" / "synthetic_transcript.json").read_text(encoding="utf-8"))
    text = "\n".join(str(row.get("text", "")) for row in rows)
    source_path = data_dir / "Transcript1-HP.txt"
    source = source_path.read_text(encoding="utf-8", errors="ignore") if source_path.exists() else ""
    source_lines = [line.strip() for line in source.splitlines() if len(line.strip()) >= 24]
    exact_replays = sum(1 for line in source_lines if line in text)
    repeated_turns = len(rows) - len({str(row.get("text", "")).strip().lower() for row in rows})
    pii_hits = count_regex_pii(text)
    speaker_validity = sum(
        1
        for row in rows
        if str(row.get("speaker", "")).lower() in {"customer", "agent", "speaker_1", "speaker_2", "assistant", "user"}
    ) / max(1, len(rows))
    quality_score = score_from_checks(
        bool(report["modes"]["transcript"].get("validation_passed")),
        1.0 if exact_replays == 0 else 0.0,
        1.0 if repeated_turns == 0 else max(0.0, 1 - repeated_turns / max(1, len(rows))),
        speaker_validity,
    )
    privacy_safety_score = max(0, 100 - min(100, pii_hits * 20 + exact_replays * 20))
    return {
        "runtime_seconds": report["comparison"]["transcript"]["seconds"],
        "quality_score": quality_score,
        "privacy_safety_score": privacy_safety_score,
        "validation_passed": report["modes"]["transcript"].get("validation_passed"),
        "turns": len(rows),
        "exact_source_replays": exact_replays,
        "repeated_turn_texts": repeated_turns,
        "speaker_label_validity": round(float(speaker_validity), 4),
        "regex_pii_hits": pii_hits,
        "business_id_like_mentions": len(BUSINESS_ID_RE.findall(text)),
        "avg_words_per_turn": round(
            sum(len(str(row.get("text", "")).split()) for row in rows) / max(1, len(rows)),
            2,
        ),
    }


def count_regex_pii(text: str) -> int:
    return len(EMAIL_RE.findall(text)) + len(PHONE_RE.findall(text)) + len(SSN_RE.findall(text))


def regex_pii_values(text: str) -> set[str]:
    return set(EMAIL_RE.findall(text)) | set(PHONE_RE.findall(text)) | set(SSN_RE.findall(text))


def pdf_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def score_setup(nemo_dir: Path) -> dict[str, Any]:
    report = read_report(nemo_dir)
    env = report.get("nvidia_nemo_environment", {})
    transcript = report.get("modes", {}).get("transcript", {})
    return {
        "curator": {
            "package": "nemo-curator",
            "installed": bool((env.get("transcript/audio curation") or {}).get("installed")),
            "benchmark_status": transcript.get("nvidia_nemo_curator"),
            "install_command": "uv pip install 'nemo-curator[text_cpu]'",
            "evidence": env.get("transcript/audio curation"),
        },
        "guardrails": {
            "package": "nemoguardrails",
            "installed": bool((env.get("SSOT grounding guardrails") or {}).get("installed")),
            "benchmark_status": transcript.get("nvidia_nemo_guardrails"),
            "install_command": "pip install nemoguardrails",
            "config_env": "SP_NVIDIA_GUARDRAILS_CONFIG_PATH",
            "evidence": env.get("SSOT grounding guardrails"),
        },
    }

def _setup_status(setup: dict[str, Any]) -> str:
    status = setup.get("benchmark_status")
    if isinstance(status, dict) and status.get("status") == "ran":
        return "completed"
    if setup.get("installed"):
        return "installed_not_configured_or_not_run"
    return "setup_instructions_ready_package_not_installed"


def render_markdown(results: dict[str, Any]) -> str:
    workflows = results["workflow_scores"]
    lines = [
        "# Backend Quality And Setup Completion",
        "",
        "## Status Tracker",
        "",
        "| Work Item | Status |",
        "| --- | --- |",
    ]
    for item, status in results["status_tracker"].items():
        lines.append(f"| {item} | {status} |")

    lines.extend(
        [
            "",
            "## Score Summary",
            "",
            "| Workflow | Metric | Current Platform | NeMo SDK |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for workflow, pair in workflows.items():
        for metric in ("quality_score", "privacy_safety_score", "runtime_seconds"):
            lines.append(
                f"| {workflow} | {metric} | `{pair['current'].get(metric)}` | `{pair['nemo'].get(metric)}` |"
            )

    lines.extend(["", "## Technical Metrics", ""])
    for workflow, pair in workflows.items():
        lines.extend(
            [
                f"### {workflow.title()}",
                "",
                "| Metric | Current Platform | NeMo SDK |",
                "| --- | ---: | ---: |",
            ]
        )
        keys = sorted(set(pair["current"]) | set(pair["nemo"]))
        for key in keys:
            if key in {"numeric_distribution", "sensitive_overlap_by_column"}:
                continue
            lines.append(f"| {key} | `{pair['current'].get(key)}` | `{pair['nemo'].get(key)}` |")

    lines.extend(
        [
            "",
            "## Setup Evidence",
            "",
            "| Capability | Package | Status | Command |",
            "| --- | --- | --- | --- |",
        ]
    )
    for capability, setup in results["setup"].items():
        lines.append(
            f"| {capability} | `{setup.get('package')}` | `{_setup_status(setup)}` | `{setup.get('install_command')}` |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
