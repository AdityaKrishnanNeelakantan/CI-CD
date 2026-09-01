"""Generates a realistic, self-contained sample SQLite database (Database
A for the demo UI) - customer -> account -> transaction, the same
three-table chain this project's own reference design uses as its
running example. The customer table's gender column is constructed at
an exact 4:3 male:female ratio so the demo directly shows the
distribution-replication and post-generation-rebalancing story with a
concrete, verifiable number instead of an abstract example.

This is demo-only scaffolding (fictional data, for exercising the
pipeline in the UI) - never used by the actual pipeline code itself.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from faker import Faker


def build_sample_database(path: str | Path, seed: int = 42, customer_count: int = 400) -> Path:
    path = Path(path)
    if path.exists():
        path.unlink()

    rng = np.random.default_rng(seed)
    faker = Faker()
    faker.seed_instance(seed)

    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE customer (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            gender TEXT NOT NULL,
            signup_date TEXT NOT NULL,
            age INTEGER NOT NULL
        );

        CREATE TABLE account (
            account_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            account_type TEXT NOT NULL,
            balance REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customer(customer_id)
        );

        CREATE TABLE "transaction" (
            transaction_id INTEGER PRIMARY KEY,
            account_id TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            transaction_date TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES account(account_id)
        );
        """
    )

    # Gender at an exact 4:3 ratio - not an approximation of some other split.
    male_count = round(customer_count * 4 / 7)
    genders = ["male"] * male_count + ["female"] * (customer_count - male_count)
    rng.shuffle(genders)

    customer_rows = []
    account_rows = []
    transaction_rows = []
    account_types = ["checking", "savings", "credit"]
    categories = ["groceries", "utilities", "entertainment", "transfer", "salary"]
    transaction_id = 1

    for i in range(customer_count):
        customer_id = f"CUST-{i:05d}"
        customer_rows.append(
            (
                customer_id,
                faker.name(),
                faker.unique.email(),
                genders[i],
                faker.date_between(start_date="-4y", end_date="today").isoformat(),
                int(rng.integers(18, 85)),
            )
        )

        for a in range(int(rng.integers(1, 3))):
            account_id = f"ACC-{i:05d}-{a}"
            balance = float(rng.uniform(-500, 25000))
            account_rows.append((account_id, customer_id, str(rng.choice(account_types)), round(balance, 2)))

            for _t in range(int(rng.integers(1, 8))):
                amount = float(rng.uniform(-2000, 3000))
                transaction_rows.append(
                    (
                        transaction_id,
                        account_id,
                        round(amount, 2),
                        str(rng.choice(categories)),
                        faker.date_between(start_date="-2y", end_date="today").isoformat(),
                    )
                )
                transaction_id += 1

    conn.executemany("INSERT INTO customer VALUES (?, ?, ?, ?, ?, ?)", customer_rows)
    conn.executemany("INSERT INTO account VALUES (?, ?, ?, ?)", account_rows)
    conn.executemany('INSERT INTO "transaction" VALUES (?, ?, ?, ?, ?)', transaction_rows)
    conn.commit()
    conn.close()

    return path
