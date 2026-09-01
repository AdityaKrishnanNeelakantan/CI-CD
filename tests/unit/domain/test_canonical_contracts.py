from __future__ import annotations

import pytest

from synth_platform.domain.contracts import (
    CANONICAL_CONTRACT_VERSION,
    canonical_fingerprint,
    contract_from_json,
    contract_to_json,
)
from synth_platform.domain.contracts.schema_adapter import database_schema_to_contract
from synth_platform.domain.contracts.dataset_contract_adapter import dataset_contract_to_canonical_contract
from synth_platform.domain.contracts.versioning import UnsupportedContractVersion
from synth_platform.domain.schema.models import ColumnSchema, DatabaseSchema, ForeignKey, TableSchema
from synth_platform.engine.inference.schema.canonical_adapter import schema_config_to_canonical_contract
from synth_platform.engine.inference.schema.schema import Column, Relationship, SchemaConfig, Table

pytestmark = pytest.mark.unit


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        source_kind="sqlite",
        tables={
            "accounts": TableSchema(
                name="accounts",
                primary_key="account_id",
                primary_key_columns=["account_id"],
                columns=[
                    ColumnSchema(name="account_id", physical_type="TEXT", nullable=False),
                    ColumnSchema(name="customer_id", physical_type="TEXT", nullable=False),
                    ColumnSchema(name="balance", physical_type="REAL", nullable=True),
                ],
            ),
            "customers": TableSchema(
                name="customers",
                primary_key="customer_id",
                primary_key_columns=["customer_id"],
                columns=[
                    ColumnSchema(name="customer_id", physical_type="TEXT", nullable=False),
                    ColumnSchema(name="email", physical_type="TEXT", logical_type="email", nullable=True),
                ],
            ),
        },
        foreign_keys=[
            ForeignKey(
                parent_table="customers",
                parent_column="customer_id",
                child_table="accounts",
                child_column="customer_id",
            )
        ],
    )


def test_database_schema_to_canonical_contract_maps_tables_fields_and_fks():
    contract = database_schema_to_contract(_schema(), source_fingerprint="sha256:abc")

    assert contract.contract_version == CANONICAL_CONTRACT_VERSION
    assert contract.source_type == "database"
    assert {entity.entity_id for entity in contract.entities} == {"table:accounts", "table:customers"}
    account_id = next(field for field in contract.fields if field.field_id == "field:accounts.account_id")
    assert account_id.primary_key is True
    assert account_id.nullable is False
    assert contract.relationships[0].relationship_type == "foreign_key"
    assert contract.relationships[0].from_field_ids == ["field:accounts.customer_id"]
    assert contract.relationships[0].to_field_ids == ["field:customers.customer_id"]


def test_contract_json_round_trip_and_fingerprint_are_stable():
    contract = database_schema_to_contract(_schema(), source_fingerprint="sha256:abc")

    text = contract_to_json(contract)
    loaded = contract_from_json(text)

    assert loaded == contract
    assert canonical_fingerprint(loaded) == canonical_fingerprint(contract)
    assert canonical_fingerprint(contract).startswith("sha256:")


def test_unsupported_contract_version_is_rejected():
    contract = database_schema_to_contract(_schema()).model_copy(update={"contract_version": "0.0.1"})

    with pytest.raises(UnsupportedContractVersion):
        contract_to_json(contract)


def test_legacy_dataset_contract_to_canonical_preserves_approved_semantics():
    legacy = {
        "contract_version": "1.0",
        "revision": 4,
        "dataset_id": "banking",
        "source_fingerprint": "sha256:abc",
        "updated_at": "2026-08-24T00:00:00+00:00",
        "tables": {
            "customers": {
                "primary_key": ["customer_id"],
                "foreign_keys": [],
                "business_rules": [],
                "columns": {
                    "customer_id": {
                        "physical_type": "TEXT",
                        "semantic_type": "identifier",
                        "nullable": False,
                        "sensitive": True,
                        "inference_status": "approved",
                        "confidence": 0.99,
                        "evidence": ["human_accepted_proposal"],
                        "alternatives": [],
                    },
                    "email": {
                        "physical_type": "TEXT",
                        "semantic_type": "email",
                        "nullable": True,
                        "sensitive": True,
                        "inference_status": "approved",
                        "confidence": 0.95,
                        "evidence": ["human_override"],
                        "alternatives": [],
                    },
                },
            }
        },
    }

    contract = dataset_contract_to_canonical_contract(legacy)
    email = next(field for field in contract.fields if field.field_id == "field:customers.email")

    assert contract.contract_id == "banking"
    assert email.semantic_type == "email"
    assert email.constraints["inference_status"] == "approved"
    assert email.provenance is not None
    assert email.provenance.evidence_refs == ["human_override"]


def test_schema_config_to_canonical_contract_maps_schema_mode_relationships():
    schema = SchemaConfig(
        name="schema-mode",
        tables=[Table(name="customers"), Table(name="orders")],
        columns={
            "customers": [Column(name="customer_id", type="uuid", unique=True)],
            "orders": [
                Column(name="order_id", type="uuid", unique=True),
                Column(name="customer_id", type="foreign_key"),
            ],
        },
        relationships=[
            Relationship(
                parent_table="customers",
                parent_key="customer_id",
                child_table="orders",
                child_key="customer_id",
            )
        ],
        seed=7,
    )

    contract = schema_config_to_canonical_contract(schema)

    assert contract.source_type == "schema"
    assert contract.generation_policy["seed"] == 7
    assert contract.relationships[0].from_field_ids == ["field:orders.customer_id"]
    assert next(field for field in contract.fields if field.name == "order_id").semantic_type == "identifier"
