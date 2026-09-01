from __future__ import annotations

import pytest

from synth_platform.engine.validation.database.release_manager import (
    MODE_DP_SHAREABLE,
    MODE_LEARNED_RESTRICTED,
    MODE_MOCK_PRIVATE,
    RELEASE_BLOCKED,
    RELEASE_PASS,
    RELEASE_REVIEW,
    evaluate_release,
)

pytestmark = pytest.mark.unit


def test_learned_restricted_passes_when_hard_checks_pass():
    result = evaluate_release({"hard_checks_passed": True, "report": {}}, MODE_LEARNED_RESTRICTED)
    assert result["decision"] == RELEASE_PASS


def test_learned_restricted_blocks_when_hard_checks_fail():
    result = evaluate_release({"hard_checks_passed": False, "report": {}}, MODE_LEARNED_RESTRICTED)
    assert result["decision"] == RELEASE_BLOCKED


def test_mock_private_reviews_on_failure_not_blocks():
    result = evaluate_release({"hard_checks_passed": False, "report": {}}, MODE_MOCK_PRIVATE)
    assert result["decision"] == RELEASE_REVIEW


def test_mock_private_passes_when_hard_checks_pass():
    result = evaluate_release({"hard_checks_passed": True, "report": {}}, MODE_MOCK_PRIVATE)
    assert result["decision"] == RELEASE_PASS


def test_dp_shareable_requires_verified_dp_report():
    result = evaluate_release(
        {"hard_checks_passed": True, "report": {"fidelity": {"t": {}}}},
        MODE_DP_SHAREABLE,
        dp_report=None,
    )
    assert result["decision"] == RELEASE_BLOCKED


def test_dp_shareable_blocks_when_hard_checks_fail_despite_verified_dp():
    result = evaluate_release(
        {"hard_checks_passed": False, "report": {"fidelity": {"t": {"col": {}}}}},
        MODE_DP_SHAREABLE,
        dp_report={"verified": True, "epsilon": 3.0},
    )
    assert result["decision"] == RELEASE_BLOCKED


def test_dp_shareable_requires_review_when_no_fidelity_evidence():
    result = evaluate_release(
        {"hard_checks_passed": True, "report": {}},
        MODE_DP_SHAREABLE,
        dp_report={"verified": True, "epsilon": 3.0},
    )
    assert result["decision"] == RELEASE_REVIEW


def test_dp_shareable_passes_with_verified_dp_and_hard_checks():
    result = evaluate_release(
        {"hard_checks_passed": True, "report": {"fidelity": {"t": {"col": {}}}}},
        MODE_DP_SHAREABLE,
        dp_report={"verified": True, "epsilon": 3.0},
    )
    assert result["decision"] == RELEASE_PASS


def test_unknown_release_mode_raises():
    with pytest.raises(ValueError):
        evaluate_release({"hard_checks_passed": True}, "INVALID_MODE")
