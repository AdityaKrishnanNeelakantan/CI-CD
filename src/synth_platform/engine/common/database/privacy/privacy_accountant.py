"""Tracks every differentially-private query spent against a table's
training data, and composes them into a total (epsilon, delta) - the
piece this project's earlier privacy review explicitly said was missing:
"the mechanism, clipping, noise addition and accountant must all be
implemented and verified."

Composition here is the classical *basic* composition theorem (Dwork &
Roth): spending (eps_1, delta_1) then (eps_2, delta_2) on the same data
costs (eps_1+eps_2, delta_1+delta_2) in total. This is provably correct
and simple to verify, at the cost of being looser (more conservative)
than advanced/Renyi-DP composition - a deliberate choice: a slightly
pessimistic but definitely-correct budget is safer than an optimistic one
with a subtle implementation bug in a more complex accountant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PrivacyBudgetExceededError(Exception):
    """Raised when a requested expenditure would exceed the configured total budget."""


@dataclass
class PrivacyExpenditure:
    purpose: str
    epsilon: float
    delta: float


@dataclass
class PrivacyAccountant:
    """One accountant per table (or per dataset, if the caller wants a
    single shared budget across tables) - construct fresh per training
    run, since a budget is meaningless without knowing exactly which
    queries against which data it has already paid for.
    """

    epsilon_budget: float
    delta_budget: float = 0.0
    _expenditures: list[PrivacyExpenditure] = field(default_factory=list)

    def spend(self, epsilon: float, delta: float = 0.0, purpose: str = "") -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")

        projected_epsilon = self.total_epsilon() + epsilon
        projected_delta = self.total_delta() + delta
        if projected_epsilon > self.epsilon_budget + 1e-12:
            raise PrivacyBudgetExceededError(
                f"spending epsilon={epsilon} for {purpose!r} would bring total epsilon to "
                f"{projected_epsilon}, exceeding the configured budget of {self.epsilon_budget}"
            )
        if projected_delta > self.delta_budget + 1e-15:
            raise PrivacyBudgetExceededError(
                f"spending delta={delta} for {purpose!r} would bring total delta to "
                f"{projected_delta}, exceeding the configured budget of {self.delta_budget}"
            )
        self._expenditures.append(PrivacyExpenditure(purpose=purpose, epsilon=epsilon, delta=delta))

    def total_epsilon(self) -> float:
        return sum(e.epsilon for e in self._expenditures)

    def total_delta(self) -> float:
        return sum(e.delta for e in self._expenditures)

    def remaining_epsilon(self) -> float:
        return self.epsilon_budget - self.total_epsilon()

    def summary(self) -> dict[str, Any]:
        return {
            "epsilon_budget": self.epsilon_budget,
            "delta_budget": self.delta_budget,
            "total_epsilon_spent": round(self.total_epsilon(), 12),
            "total_delta_spent": round(self.total_delta(), 15),
            "expenditures": [
                {"purpose": e.purpose, "epsilon": e.epsilon, "delta": e.delta} for e in self._expenditures
            ],
        }
