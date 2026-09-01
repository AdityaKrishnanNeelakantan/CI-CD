from __future__ import annotations

import pytest

from synth_platform.engine.common.database.core.stage_result import STATUS_FAILED, STATUS_SUCCESS, StageResult

pytestmark = pytest.mark.unit


def test_is_success_true_for_success_status():
    result = StageResult(
        stage_name="discovery",
        status=STATUS_SUCCESS,
        input_references=[],
        output_references=[],
    )
    assert result.is_success() is True


def test_is_success_false_for_failed_status():
    result = StageResult(
        stage_name="discovery",
        status=STATUS_FAILED,
        input_references=[],
        output_references=[],
    )
    assert result.is_success() is False


def test_to_dict_round_trips_all_fields():
    result = StageResult(
        stage_name="discovery",
        status=STATUS_SUCCESS,
        input_references=["a"],
        output_references=["b"],
        metrics={"m": 1},
        warnings=["w"],
        errors=[],
        evidence={"e": True},
    )
    data = result.to_dict()
    assert data == {
        "stage_name": "discovery",
        "status": "success",
        "input_references": ["a"],
        "output_references": ["b"],
        "metrics": {"m": 1},
        "warnings": ["w"],
        "errors": [],
        "evidence": {"e": True},
    }
