from __future__ import annotations

from typing import Any

import pytest

from synth_platform.engine.documents.pdf.archetypes import registry as archetype_registry
from synth_platform.engine.documents.pdf.archetypes.bank_statement import BankStatementArchetype
from synth_platform.engine.documents.pdf.archetypes.base import DocumentArchetype

pytestmark = pytest.mark.unit


class _FakeArchetype(DocumentArchetype):
    archetype_id = "fake_archetype"
    document_types = frozenset({"invoice"})
    display_name = "Fake Archetype"

    def expected_field_labels(self) -> list[str]:
        return ["Fake Label"]

    def generator_priors(self) -> dict[str, Any]:
        return {"fake_field": {"pattern": "X"}}

    def validation_rules(self) -> list[dict[str, Any]]:
        return [{"rule_id": "fake_rule", "expression": "true"}]


@pytest.fixture(autouse=True)
def _isolated_registry():
    """The archetype registry is a module-level dict populated lazily on
    first use; snapshot/restore it so tests that register extra archetypes
    never leak state into other tests or the real service.py callers.
    """
    original = dict(archetype_registry._REGISTRY)
    yield
    archetype_registry._REGISTRY.clear()
    archetype_registry._REGISTRY.update(original)


def test_bank_statement_archetype_declares_expected_metadata():
    archetype = BankStatementArchetype()
    assert archetype.archetype_id == "bank_statement"
    assert archetype.document_types == frozenset({"bank_statement", "report"})
    assert "Account Number" in archetype.expected_field_labels()
    assert "currency_amount" in archetype.generator_priors()
    rule_ids = {rule["rule_id"] for rule in archetype.validation_rules()}
    assert rule_ids == {"balance_consistency", "date_order"}


def test_bank_statement_to_config_round_trips_all_metadata():
    archetype = BankStatementArchetype()
    config = archetype.to_config()
    assert config["archetype_id"] == "bank_statement"
    assert config["document_types"] == sorted({"bank_statement", "report"})
    assert config["display_name"] == "Bank Statement"
    assert config["expected_field_labels"] == archetype.expected_field_labels()
    assert config["generator_priors"] == archetype.generator_priors()
    assert config["validation_rules"] == archetype.validation_rules()


def test_bank_statement_match_score_boosted_by_keywords_even_without_matching_document_type():
    archetype = BankStatementArchetype()
    text = "Account Number: 12345 Opening Balance: 100 Closing Balance: 50 Available Balance: 50"
    classification = {"document_type": "letter", "confidence": 0.9}
    score = archetype.match_score(text, classification)
    # document_type doesn't match, so keyword heuristic (not confidence) must drive the score.
    assert score == pytest.approx(0.8, abs=1e-6)


def test_bank_statement_match_score_uses_classification_confidence_when_document_type_matches():
    archetype = BankStatementArchetype()
    classification = {"document_type": "report", "confidence": 0.9}
    score = archetype.match_score("no bank keywords here", classification)
    assert score == pytest.approx(0.9, abs=1e-6)


def test_base_match_score_returns_zero_when_document_type_not_in_archetype_types():
    archetype = _FakeArchetype()
    classification = {"document_type": "resume", "confidence": 0.95}
    assert archetype.match_score("irrelevant text", classification) == 0.0


def test_register_and_get_archetype_returns_registered_instance():
    fake = _FakeArchetype()
    archetype_registry.register_archetype(fake)
    assert archetype_registry.get_archetype("fake_archetype") is fake


def test_get_archetype_returns_none_for_unknown_id():
    assert archetype_registry.get_archetype("does_not_exist") is None


def test_list_archetypes_includes_default_bank_statement():
    archetypes = archetype_registry.list_archetypes()
    assert any(isinstance(a, BankStatementArchetype) for a in archetypes)


def test_resolve_archetype_falls_back_to_generic_shape_when_nothing_matches():
    result = archetype_registry.resolve_archetype("irrelevant text", {"document_type": "resume", "confidence": 0.9})
    assert result["archetype_id"] == "generic_shape_fallback"
    assert result["display_name"] == "Generic Shape Fallback"
    assert result["config"] is None


def test_resolve_archetype_selects_highest_scoring_registered_archetype():
    archetype_registry.register_archetype(_FakeArchetype())
    classification = {"document_type": "invoice", "confidence": 0.6}
    result = archetype_registry.resolve_archetype("irrelevant text", classification)
    assert result["archetype_id"] == "fake_archetype"
    assert result["match_score"] == pytest.approx(0.6, abs=1e-6)
    assert result["config"]["archetype_id"] == "fake_archetype"
