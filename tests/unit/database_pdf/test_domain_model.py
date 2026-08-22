"""Unit tests for the canonical cross-track domain model."""

from __future__ import annotations

from synth_platform.engine.common.database.core.domain_model import (
    CanonicalDomainModel,
    CanonicalEntity,
    EntityField,
    build_banking_domain_model,
    map_contract_to_domain,
)


def test_build_banking_domain_model_covers_expected_entities_and_relationships():
    model = build_banking_domain_model()

    assert set(model.entities) == {"Customer", "Account", "Statement", "Transaction"}
    assert model.entity_for_table("customers").entity_type == "Customer"
    assert model.entity_for_table("accounts").entity_type == "Account"
    assert model.entity_for_table("no_such_table") is None
    assert len(model.relationships) == 2


def test_entity_for_document_field_resolves_by_field_id():
    model = build_banking_domain_model()

    assert model.entity_for_document_field("field_account_number").entity_type == "Account"
    assert model.entity_for_document_field("nonexistent_field") is None


def test_to_dict_round_trips_entities_and_relationships():
    model = CanonicalDomainModel()
    model.add_entity(
        CanonicalEntity(
            entity_type="Widget",
            fields=[EntityField("widget_id", "identifier", "INTEGER")],
            table_mapping="widgets",
            document_field_ids=["field_widget_id"],
        )
    )

    payload = model.to_dict()

    assert payload["entities"]["Widget"]["table_mapping"] == "widgets"
    assert payload["entities"]["Widget"]["fields"] == [
        {"name": "widget_id", "semantic_role": "identifier", "physical_type": "INTEGER"}
    ]
    assert payload["relationships"] == []


def test_map_contract_to_domain_only_maps_known_tables():
    model = build_banking_domain_model()
    contract = {"tables": {"customers": {}, "accounts": {}, "unmapped_table": {}}}

    mapping = map_contract_to_domain(contract, model)

    assert mapping == {"customers": "Customer", "accounts": "Account"}
