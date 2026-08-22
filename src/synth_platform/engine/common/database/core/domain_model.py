"""Canonical domain model: shared entity abstraction bridging the
Database Twin and Document Twin tracks.

Both tracks currently maintain independent data models (relational
schema graph vs document template spec).  This module defines the
cross-track entity vocabulary so a Customer, Account, Statement, or
Transaction can be generated consistently across both a relational
database representation and a document representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntityField:
    name: str
    semantic_role: str
    physical_type: str = "TEXT"


@dataclass(frozen=True)
class EntityRelationship:
    parent_entity: str
    child_entity: str
    parent_key: str
    child_key: str


@dataclass
class CanonicalEntity:
    entity_type: str
    fields: list[EntityField] = field(default_factory=list)
    table_mapping: str | None = None
    document_field_ids: list[str] = field(default_factory=list)


@dataclass
class CanonicalDomainModel:
    """Shared entity graph consumed by both DB and PDF twin tracks."""

    entities: dict[str, CanonicalEntity] = field(default_factory=dict)
    relationships: list[EntityRelationship] = field(default_factory=list)

    def add_entity(self, entity: CanonicalEntity) -> None:
        self.entities[entity.entity_type] = entity

    def add_relationship(self, relationship: EntityRelationship) -> None:
        self.relationships.append(relationship)

    def entity_for_table(self, table_name: str) -> CanonicalEntity | None:
        for entity in self.entities.values():
            if entity.table_mapping == table_name:
                return entity
        return None

    def entity_for_document_field(self, field_id: str) -> CanonicalEntity | None:
        for entity in self.entities.values():
            if field_id in entity.document_field_ids:
                return entity
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": {
                name: {
                    "entity_type": e.entity_type,
                    "fields": [
                        {"name": f.name, "semantic_role": f.semantic_role, "physical_type": f.physical_type}
                        for f in e.fields
                    ],
                    "table_mapping": e.table_mapping,
                    "document_field_ids": e.document_field_ids,
                }
                for name, e in self.entities.items()
            },
            "relationships": [
                {
                    "parent_entity": r.parent_entity,
                    "child_entity": r.child_entity,
                    "parent_key": r.parent_key,
                    "child_key": r.child_key,
                }
                for r in self.relationships
            ],
        }


def build_banking_domain_model() -> CanonicalDomainModel:
    """Reference domain model: Customer -> Account -> Statement -> Transaction."""
    model = CanonicalDomainModel()

    model.add_entity(CanonicalEntity(
        entity_type="Customer",
        fields=[
            EntityField("customer_id", "identifier"),
            EntityField("first_name", "person_name"),
            EntityField("last_name", "person_name"),
            EntityField("email", "email_address"),
        ],
        table_mapping="customers",
        document_field_ids=["field_account_holder", "field_customer_name"],
    ))

    model.add_entity(CanonicalEntity(
        entity_type="Account",
        fields=[
            EntityField("account_id", "identifier"),
            EntityField("account_number", "account_number"),
            EntityField("balance", "currency_amount"),
            EntityField("account_type", "category"),
        ],
        table_mapping="accounts",
        document_field_ids=["field_account_number", "field_balance"],
    ))

    model.add_entity(CanonicalEntity(
        entity_type="Statement",
        fields=[
            EntityField("statement_id", "identifier"),
            EntityField("statement_date", "date"),
            EntityField("opening_balance", "currency_amount"),
            EntityField("closing_balance", "currency_amount"),
        ],
        table_mapping=None,
        document_field_ids=["field_statement_date", "field_opening_balance", "field_closing_balance"],
    ))

    model.add_entity(CanonicalEntity(
        entity_type="Transaction",
        fields=[
            EntityField("transaction_id", "identifier"),
            EntityField("amount", "currency_amount"),
            EntityField("transaction_date", "date"),
            EntityField("description", "generic_text"),
        ],
        table_mapping="transactions",
        document_field_ids=["field_transaction_amount", "field_transaction_date"],
    ))

    model.add_relationship(EntityRelationship("Customer", "Account", "customer_id", "customer_id"))
    model.add_relationship(EntityRelationship("Account", "Transaction", "account_id", "account_id"))

    return model


def map_contract_to_domain(
    dataset_contract: dict[str, Any], domain_model: CanonicalDomainModel
) -> dict[str, str]:
    """Map dataset_contract table names to canonical entity types."""
    mapping: dict[str, str] = {}
    for table_name in dataset_contract.get("tables", {}):
        entity = domain_model.entity_for_table(table_name)
        if entity:
            mapping[table_name] = entity.entity_type
    return mapping
