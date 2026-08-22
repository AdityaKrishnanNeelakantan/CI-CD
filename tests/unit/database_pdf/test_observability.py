"""Unit tests for src.core.observability.log_stage_result."""

from __future__ import annotations

import logging

import pytest

from synth_platform.engine.common.database.core.observability import LOGGER_NAME, log_stage_result
from synth_platform.engine.common.database.core.stage_result import StageResult

pytestmark = pytest.mark.unit


def test_successful_stage_with_no_warnings_logs_at_info(caplog: pytest.LogCaptureFixture):
    result = StageResult(
        stage_name="discovery", status="success", input_references=[], output_references=["discovery.json"]
    )
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_stage_result("run-1", result)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert "discovery" in caplog.records[0].message


def test_successful_stage_with_warnings_logs_at_warning(caplog: pytest.LogCaptureFixture):
    result = StageResult(
        stage_name="profiling",
        status="success",
        input_references=[],
        output_references=["profile.json"],
        warnings=["correlations_skipped_in_chunked_mode"],
    )
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_stage_result("run-1", result)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


def test_failed_stage_logs_at_error(caplog: pytest.LogCaptureFixture):
    result = StageResult(
        stage_name="cleaning",
        status="failed",
        input_references=[],
        output_references=[],
        errors=["dataset_contract.json missing"],
    )
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_stage_result("run-1", result)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert "dataset_contract.json missing" in caplog.records[0].message


def test_log_message_includes_run_id_and_stage_name(caplog: pytest.LogCaptureFixture):
    result = StageResult(stage_name="inference", status="success", input_references=[], output_references=[])
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_stage_result("run-abc123", result)

    message = caplog.records[0].message
    assert "run-abc123" in message
    assert "inference" in message
