-- schema.sql
-- Banking demo source-of-truth schema for schema mode / SQLite-compatible execution.
-- This file defines the relational structure used by the synthetic data platform.

PRAGMA foreign_keys = ON;

-- Drop child tables first so reruns work cleanly.
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS loans;
DROP TABLE IF EXISTS cards;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS merchants;
DROP TABLE IF EXISTS branches;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    city TEXT,
    credit_score INTEGER,
    created_at TEXT
);

CREATE TABLE branches (
    branch_id TEXT PRIMARY KEY,
    branch_name TEXT,
    manager_name TEXT,
    city TEXT,
    country TEXT
);

CREATE TABLE merchants (
    merchant_id TEXT PRIMARY KEY,
    merchant_name TEXT,
    city TEXT
);

CREATE TABLE accounts (
    account_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    account_type TEXT,
    balance_usd REAL,
    open_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
);

CREATE TABLE cards (
    card_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    card_type TEXT,
    expiration_date TEXT,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE loans (
    loan_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    loan_amount REAL,
    interest_rate REAL,
    start_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    amount_usd REAL,
    transaction_date TEXT,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id),
    FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
);
