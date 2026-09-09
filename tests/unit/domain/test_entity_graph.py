from __future__ import annotations

import pytest

from synth_platform.domain.contracts.models import (
    CanonicalContract,
    CanonicalEntity,
    CanonicalRelationship,
)
from synth_platform.domain.entities.assembler import entity_graph_from_contract

pytestmark = pytest.mark.unit


def test_entity_graph_assembles_nodes_and_relationships_from_contract():
    contract = CanonicalContract(
        source_type="database",
        entities=[
            CanonicalEntity(entity_id="table:customers", name="customers", fields=["field:customers.id"]),
            CanonicalEntity(entity_id="table:accounts", name="accounts", fields=["field:accounts.customer_id"]),
        ],
        relationships=[
            CanonicalRelationship(
                relationship_id="fk:accounts->customers",
                relationship_type="foreign_key",
                from_entity_id="table:accounts",
                to_entity_id="table:customers",
                from_field_ids=["field:accounts.customer_id"],
                to_field_ids=["field:customers.id"],
            )
        ],
    )

    graph = entity_graph_from_contract(contract)

    assert graph.node("table:customers") is not None
    assert graph.outgoing("table:accounts")[0].to_entity_id == "table:customers"
    assert graph.incoming("table:customers")[0].from_field_ids == ["field:accounts.customer_id"]
