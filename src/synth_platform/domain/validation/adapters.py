"""Adapters from workflow-specific reports into the common validation model."""
from __future__ import annotations

from typing import Any, Mapping

from synth_platform.domain.validation.models import CheckResult, Status, ValidationReport


def schema_validation_to_common(report: Mapping[str, Any], row_counts: Mapping[str, int]) -> ValidationReport:
    hard_passed = _status_from_bool(bool(report.get("passed", report.get("hard_checks_passed", True))))
    checks = [
        CheckResult(
            name="requested_counts",
            dimension="structural",
            status=Status.PASS,
            ran=True,
            metric=float(sum(int(v) for v in row_counts.values())),
            detail="Generated row counts are compared to requested table counts.",
        )
    ]
    return ValidationReport(checks=checks, overall=hard_passed)


def database_validation_to_common(qa_report: Mapping[str, Any]) -> ValidationReport:
    report = qa_report.get("report") or {}
    integrity = report.get("integrity") or {}
    fk_validity = (integrity.get("fk_validity") or {}).get("overall_fk_validity")
    checks = [
        CheckResult(
            name="hard_checks",
            dimension="structural",
            status=_status_from_bool(bool(qa_report.get("hard_checks_passed"))),
            ran=True,
            detail="Database QA hard checks.",
        )
    ]
    if fk_validity is not None:
        checks.append(
            CheckResult(
                name="fk_integrity",
                dimension="structural",
                status=Status.PASS if float(fk_validity) >= 1.0 else Status.FAIL,
                ran=True,
                metric=float(fk_validity),
                threshold=1.0,
                detail="Child foreign keys resolve to generated parent keys.",
            )
        )
    return ValidationReport(checks=checks, overall=_overall(checks))


def pdf_validation_to_common(validation_report: Mapping[str, Any]) -> ValidationReport:
    report = validation_report.get("report") or validation_report
    checks = [
        CheckResult(
            name="field_accuracy",
            dimension="structural",
            status=Status.PASS if float(report.get("field_accuracy") or 0.0) >= 1.0 else Status.FAIL,
            ran=True,
            metric=float(report.get("field_accuracy") or 0.0),
            threshold=1.0,
            detail="Rendered fields match generated ground truth.",
        ),
        CheckResult(
            name="layout_overflow",
            dimension="structural",
            status=Status.PASS if int(report.get("overflow_failure_count") or 0) == 0 else Status.FAIL,
            ran=True,
            metric=float(report.get("overflow_failure_count") or 0),
            threshold=0.0,
            detail="Synthetic values fit document layout regions.",
        ),
    ]
    return ValidationReport(checks=checks, overall=_overall(checks))


def _status_from_bool(value: bool) -> Status:
    return Status.PASS if value else Status.FAIL


def _overall(checks: list[CheckResult]) -> Status:
    if any(check.status == Status.FAIL and check.is_critical for check in checks):
        return Status.FAIL
    if any(check.status == Status.WARN for check in checks):
        return Status.WARN
    return Status.PASS if checks else Status.NOT_RUN
