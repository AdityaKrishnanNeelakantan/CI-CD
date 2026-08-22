"""Shared fixtures: a banking SQLite database + the platform."""
from __future__ import annotations

import random
import sqlite3

import pytest

from synth_platform import SyntheticDataPlatform


@pytest.fixture()
def banking_db(tmp_path):
    path = tmp_path / "banking.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER, email TEXT, segment TEXT);
        CREATE TABLE accounts (id INTEGER PRIMARY KEY, user_id INTEGER, balance REAL,
            status TEXT, FOREIGN KEY(user_id) REFERENCES users(id));
        """
    )
    random.seed(7)
    for i in range(1, 201):
        conn.execute("INSERT INTO users VALUES (?,?,?,?)",
                     (i, random.randint(18, 80), f"u{i}@corp.example",
                      random.choice(["retail", "smb", "corp"])))
    for j in range(1, 401):
        conn.execute("INSERT INTO accounts VALUES (?,?,?,?)",
                     (j, random.randint(1, 200), round(random.gauss(5000, 1500), 2),
                      random.choice(["active", "active", "closed", "premium"])))
    conn.commit(); conn.close()
    return str(path)


@pytest.fixture()
def platform():
    return SyntheticDataPlatform.from_settings()

# Shared database fixtures used by Database Twin unit/integration suites.
# Importing fixture functions into conftest registers them with pytest while
# keeping fixture builders under tests/fixtures/.
from tests.fixtures.pdf_factory import empty_sqlite_db, temp_sqlite_db  # noqa: E402,F401
