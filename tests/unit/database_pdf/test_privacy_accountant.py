from __future__ import annotations

import pytest

from synth_platform.engine.common.database.privacy.privacy_accountant import PrivacyAccountant, PrivacyBudgetExceededError

pytestmark = pytest.mark.unit


def test_spend_within_budget_succeeds():
    accountant = PrivacyAccountant(epsilon_budget=1.0)
    accountant.spend(0.3, purpose="mean")
    accountant.spend(0.3, purpose="variance")
    assert accountant.total_epsilon() == pytest.approx(0.6)
    assert accountant.remaining_epsilon() == pytest.approx(0.4)


def test_spend_exceeding_budget_raises_and_does_not_partially_apply():
    accountant = PrivacyAccountant(epsilon_budget=0.5)
    accountant.spend(0.4, purpose="mean")
    with pytest.raises(PrivacyBudgetExceededError):
        accountant.spend(0.2, purpose="variance")
    # The rejected spend must not have been recorded.
    assert accountant.total_epsilon() == pytest.approx(0.4)


def test_composition_is_additive_across_many_queries():
    accountant = PrivacyAccountant(epsilon_budget=10.0)
    for i in range(5):
        accountant.spend(0.5, purpose=f"query_{i}")
    assert accountant.total_epsilon() == pytest.approx(2.5)


def test_delta_budget_is_tracked_and_enforced_independently_of_epsilon():
    accountant = PrivacyAccountant(epsilon_budget=10.0, delta_budget=1e-5)
    accountant.spend(1.0, delta=6e-6, purpose="correlation")
    with pytest.raises(PrivacyBudgetExceededError):
        accountant.spend(1.0, delta=6e-6, purpose="correlation_again")


def test_spend_rejects_non_positive_epsilon():
    accountant = PrivacyAccountant(epsilon_budget=1.0)
    with pytest.raises(ValueError):
        accountant.spend(0.0, purpose="x")


def test_summary_reports_every_expenditure_and_totals():
    accountant = PrivacyAccountant(epsilon_budget=1.0, delta_budget=1e-5)
    accountant.spend(0.3, delta=1e-6, purpose="mean")
    accountant.spend(0.2, delta=2e-6, purpose="variance")
    summary = accountant.summary()
    assert summary["total_epsilon_spent"] == pytest.approx(0.5)
    assert summary["total_delta_spent"] == pytest.approx(3e-6)
    assert len(summary["expenditures"]) == 2
    assert summary["expenditures"][0]["purpose"] == "mean"


def test_fresh_accountant_has_zero_spend():
    accountant = PrivacyAccountant(epsilon_budget=1.0)
    assert accountant.total_epsilon() == 0.0
    assert accountant.remaining_epsilon() == 1.0
