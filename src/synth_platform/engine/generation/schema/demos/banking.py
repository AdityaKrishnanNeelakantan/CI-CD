"""Banking/KYC example schema — optional demo, not the core product default."""

from __future__ import annotations

from typing import Optional

from synth_platform.engine.inference.schema.schema import BankingConfig, Column, Relationship, SchemaConfig, Table


def build_banking_demo_schema(
    *,
    transactions: int,
    seed: int,
    country: str = "US",
    llm_notes: bool = True,
    llm_model: Optional[str] = None,
) -> SchemaConfig:
    """Build the multi-table banking example schema used by CLI and Streamlit demos."""
    transactions = max(1, int(transactions))
    country = str(country or "US").upper()
    if country not in {"US", "IN", "GLOBAL"}:
        raise ValueError("country must be one of: US, IN, GLOBAL")

    banking = BankingConfig(
        country=country,
        currency="INR" if country == "IN" else "USD",
        routing_number_format="ifsc" if country == "IN" else "aba",
        account_number_digits=12,
    )

    customer_rows = max(10, transactions // 5)
    account_rows = max(10, transactions // 3)

    return SchemaConfig(
        name="Banking Example",
        domain="banking",
        seed=seed,
        banking=banking,
        tables=[
            Table(name="customers", row_count=customer_rows),
            Table(name="accounts", row_count=account_rows),
            Table(name="transactions", row_count=transactions),
        ],
        columns={
            "customers": [
                Column(
                    name="customer_id",
                    type="int",
                    unique=True,
                    distribution_params={"min": 1, "max": customer_rows + 1000},
                ),
                Column(name="full_name", type="text", distribution_params={"text_type": "name"}),
                Column(name="ssn", type="ssn", unique=True),
                Column(name="aadhaar", type="aadhaar", unique=True),
                Column(
                    name="risk_tier",
                    type="categorical",
                    distribution_params={
                        "choices": ["low", "medium", "high"],
                        "probabilities": [0.70, 0.25, 0.05],
                    },
                ),
            ],
            "accounts": [
                Column(
                    name="account_id",
                    type="int",
                    unique=True,
                    distribution_params={"min": 1, "max": account_rows + 1000},
                ),
                Column(name="customer_id", type="foreign_key"),
                Column(name="account_number", type="account_number", unique=True),
                Column(name="routing_number", type="routing_number" if country != "IN" else "ifsc_code"),
                Column(
                    name="account_type",
                    type="categorical",
                    distribution_params={
                        "choices": ["checking", "savings", "loan"],
                        "probabilities": [0.55, 0.35, 0.10],
                    },
                ),
                Column(
                    name="balance",
                    type="money",
                    distribution_params={"min": 0, "max": 250000, "decimals": 2},
                ),
                Column(name="currency", type="currency"),
            ],
            "transactions": [
                Column(
                    name="transaction_id",
                    type="int",
                    unique=True,
                    distribution_params={"min": 1, "max": transactions + 1000},
                ),
                Column(name="account_id", type="foreign_key"),
                Column(
                    name="amount",
                    type="money",
                    distribution_params={"min": 1, "max": 10000, "decimals": 2},
                ),
                Column(name="merchant", type="text", distribution_params={"text_type": "company"}),
                Column(
                    name="case_note",
                    type="text",
                    description="Privacy-safe service narrative for support review.",
                    distribution_params={
                        "text_type": "service_case_note",
                        "llm_text": True,
                        "llm_enabled": llm_notes,
                        "llm_model": llm_model,
                        "country": country,
                        "context": "transaction support note for card, transfer, ATM, mobile, or merchant settlement review",
                        "max_words": 42,
                    },
                ),
                Column(
                    name="channel",
                    type="categorical",
                    distribution_params={
                        "choices": ["ach", "wire", "card", "atm", "mobile"],
                        "probabilities": [0.25, 0.10, 0.35, 0.10, 0.20],
                    },
                ),
                Column(
                    name="posted_at",
                    type="datetime",
                    distribution_params={"start": "2024-01-01", "end": "2024-12-31"},
                ),
                Column(name="is_flagged", type="boolean", distribution_params={"probability": 0.20}),
            ],
        },
        relationships=[
            Relationship(
                parent_table="customers",
                parent_key="customer_id",
                child_table="accounts",
                child_key="customer_id",
            ),
            Relationship(
                parent_table="accounts",
                parent_key="account_id",
                child_table="transactions",
                child_key="account_id",
            ),
        ],
    )
