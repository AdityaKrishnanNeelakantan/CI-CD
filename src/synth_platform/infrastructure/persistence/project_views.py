"""Shared project-history and product-settings view helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from synth_platform.domain.product_settings import DEFAULT_PRODUCT_SETTINGS, read_generation_defaults
from synth_platform.infrastructure.persistence.platform_db import PlatformDB, ProjectRecord, RunRecord

DEFAULT_SETTINGS: dict[str, Any] = dict(DEFAULT_PRODUCT_SETTINGS)

WORKFLOW_LABELS = {
    "schema": "Schema",
    "database": "Database",
    "pdf": "Document",
    "interaction": "Interaction",
}


@dataclass(frozen=True)
class ProjectSummary:
    id: str
    name: str
    workflow_type: str
    workflow_label: str
    created_at: str
    created_on: str
    status: str
    records: str
    validation: str
    transfer: str
    latest_run_id: str | None


def read_product_settings(db: PlatformDB) -> dict[str, Any]:
    return read_generation_defaults(db)


def write_product_settings(db: PlatformDB, values: dict[str, Any]) -> None:
    payload = {
        "generation_mode": str(values.get("generation_mode") or DEFAULT_SETTINGS["generation_mode"]),
        "default_record_count": int(values.get("default_record_count") or DEFAULT_SETTINGS["default_record_count"]),
        "privacy_level": str(values.get("privacy_level") or DEFAULT_SETTINGS["privacy_level"]),
        "default_output_format": str(values.get("default_output_format") or DEFAULT_SETTINGS["default_output_format"]),
    }
    db.write_settings(payload)


def load_project_summaries(db: PlatformDB, *, workflow_type: str | None = None) -> list[ProjectSummary]:
    projects = db.list_projects(workflow_type=workflow_type)
    runs_by_project = _latest_runs_by_project(db)
    return [_project_summary(project, runs_by_project.get(project.id)) for project in projects]


def project_run_rows(db: PlatformDB, project_id: str) -> list[dict[str, str]]:
    return [_run_row(run) for run in db.list_runs(project_id=project_id)]


def project_table_rows(summaries: list[ProjectSummary]) -> list[dict[str, str]]:
    return [
        {
            "Name": summary.name,
            "Type": summary.workflow_label,
            "Records": summary.records,
            "Created On": summary.created_on,
            "Status": _label(summary.status),
            "Validation": summary.validation,
            "Transfer": summary.transfer,
        }
        for summary in summaries
    ]


def _latest_runs_by_project(db: PlatformDB) -> dict[str, RunRecord]:
    latest: dict[str, RunRecord] = {}
    for run in db.list_runs():
        latest.setdefault(run.project_id, run)
    return latest


def _project_summary(project: ProjectRecord, latest_run: RunRecord | None) -> ProjectSummary:
    return ProjectSummary(
        id=project.id,
        name=project.name,
        workflow_type=project.workflow_type,
        workflow_label=WORKFLOW_LABELS.get(project.workflow_type, _label(project.workflow_type)),
        created_at=project.created_at,
        created_on=_format_timestamp(project.created_at),
        status=project.status,
        records=_records_label(latest_run),
        validation=_validation_label(latest_run),
        transfer=_transfer_label(latest_run),
        latest_run_id=latest_run.id if latest_run else None,
    )


def _run_row(run: RunRecord) -> dict[str, str]:
    return {
        "Run": run.id,
        "Created On": _format_timestamp(run.created_at),
        "Status": _label(run.status),
        "Validation": _validation_label(run),
        "Transfer": _transfer_label(run),
        "Output": run.output_id or "-",
        "Records": _records_label(run),
    }


def _records_label(run: RunRecord | None) -> str:
    if run is None:
        return "-"
    for key in ("row_counts", "row_counts_by_table"):
        counts = run.metadata.get(key)
        if isinstance(counts, dict):
            numeric = [int(value) for value in counts.values() if isinstance(value, int | float)]
            if numeric:
                return f"{sum(numeric):,}"
    turn_count = run.metadata.get("turn_count")
    if isinstance(turn_count, int | float):
        return f"{int(turn_count):,}"
    return "-"


def _validation_label(run: RunRecord | None) -> str:
    if run is None:
        return "-"
    if run.validation_passed is True:
        return "Passed"
    if run.validation_passed is False:
        return "Failed"
    return _label(run.validation_status) if run.validation_status else "-"


def _transfer_label(run: RunRecord | None) -> str:
    if run is None:
        return "-"
    if run.transfer_status == "success":
        return "Allowed"
    if run.transfer_status == "blocked":
        return "Blocked"
    return "Not attempted"


def _format_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.strftime("%b %d, %Y %I:%M %p")


def _label(value: str | None) -> str:
    if not value:
        return "-"
    return value.replace("_", " ").title()
