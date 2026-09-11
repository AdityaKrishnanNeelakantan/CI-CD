from __future__ import annotations

from synth_platform.domain.validation.adapters import (
    database_validation_to_common,
    pdf_validation_to_common,
    schema_validation_to_common,
)
from synth_platform.domain.validation.models import Status


def test_schema_validation_adapter_reports_requested_counts():
    report = schema_validation_to_common({"passed": True}, {"customers": 10, "orders": 20})

    assert report.overall == Status.PASS
    assert report.checks[0].name == "requested_counts"
    assert report.checks[0].metric == 30.0


def test_database_validation_adapter_reports_fk_integrity():
    report = database_validation_to_common(
        {
            "hard_checks_passed": True,
            "report": {"integrity": {"fk_validity": {"overall_fk_validity": 1.0}}},
        }
    )

    assert report.overall == Status.PASS
    assert any(check.name == "fk_integrity" for check in report.checks)


def test_pdf_validation_adapter_reports_layout_and_fields():
    report = pdf_validation_to_common(
        {"report": {"field_accuracy": 1.0, "overflow_failure_count": 0}}
    )

    assert report.overall == Status.PASS
    assert {check.name for check in report.checks} == {"field_accuracy", "layout_overflow"}
