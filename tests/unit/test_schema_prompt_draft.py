from synth_platform.application.workflows import schema_prompt_draft


def test_schema_prompt_table_hints_keep_compound_tables_and_drop_field_words():
    prompt = (
        "Build an ecommerce dataset with customers, orders, order items, products, payments, "
        "and shipments. Include relationships and realistic operational fields."
    )

    hints = schema_prompt_draft._schema_prompt_table_hints(prompt)

    assert hints == ["customers", "orders", "order_items", "products", "payments", "shipments"]


def test_schema_prompt_repairs_infer_generic_operational_relationships():
    prompt = (
        "Build an ecommerce dataset with customers, orders, order items, products, payments, "
        "and shipments. Include relationships and realistic operational fields."
    )
    table_hints = schema_prompt_draft._schema_prompt_table_hints(prompt)
    payload = {"name": "draft", "tables": [], "relationships": []}

    repaired, repairs = schema_prompt_draft._repair_schema_draft_payload(
        payload,
        table_hints,
        schema_prompt_draft._relationship_hints_from_prompt(prompt, table_hints),
    )

    assert [table["name"] for table in repaired["tables"]] == table_hints
    assert "fields" not in {table["name"] for table in repaired["tables"]}
    relationships = {
        (rel["parent_table"], rel["parent_key"], rel["child_table"], rel["child_key"])
        for rel in repaired["relationships"]
    }
    assert ("customers", "customer_id", "orders", "customer_id") in relationships
    assert ("orders", "order_id", "order_items", "order_id") in relationships
    assert ("products", "product_id", "order_items", "product_id") in relationships
    assert ("orders", "order_id", "payments", "order_id") in relationships
    assert ("orders", "order_id", "shipments", "order_id") in relationships
    assert any(repair["action"] == "added_inferred_relationship" for repair in repairs)


def test_schema_prompt_hints_stop_before_follow_on_instructions_and_keep_row_count():
    prompt = (
        "Build a retail returns dataset with customers, orders, return requests, refund transactions, "
        "products, and support agents. Keep primary keys and foreign keys valid, include return reasons "
        "and refund status distributions, and generate 30 rows per table."
    )

    hints = schema_prompt_draft._schema_prompt_table_hints(prompt)
    repaired, _repairs = schema_prompt_draft._scaffold_schema_from_prompt_hints(prompt, hints)

    assert hints == [
        "customers",
        "orders",
        "return_requests",
        "refund_transactions",
        "products",
        "support_agents",
    ]
    assert {table["row_count"] for table in repaired["tables"]} == {30}
    relationships = {
        (rel["parent_table"], rel["parent_key"], rel["child_table"], rel["child_key"])
        for rel in repaired["relationships"]
    }
    assert ("customers", "customer_id", "orders", "customer_id") in relationships
    assert ("orders", "order_id", "return_requests", "order_id") in relationships
    assert ("products", "product_id", "return_requests", "product_id") in relationships
    assert ("support_agents", "support_agent_id", "return_requests", "support_agent_id") in relationships
    assert ("return_requests", "return_request_id", "refund_transactions", "return_request_id") in relationships
