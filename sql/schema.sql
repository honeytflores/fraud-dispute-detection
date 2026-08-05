-- schema.sql
-- Defines the disputed_transactions table used throughout this project.
-- Designed for SQLite, but portable to Postgres/MySQL with minor type tweaks.

DROP TABLE IF EXISTS disputed_transactions;

CREATE TABLE disputed_transactions (
    transaction_id         TEXT PRIMARY KEY,
    customer_id             INTEGER NOT NULL,
    transaction_datetime    TEXT NOT NULL,      -- ISO 8601 timestamp
    amount                   REAL NOT NULL,
    merchant_category        TEXT NOT NULL,
    channel                  TEXT NOT NULL,      -- Online / In-Store / Phone/Call Center
    country                  TEXT NOT NULL,      -- ISO-ish country code
    account_age_days         INTEGER NOT NULL,
    prior_disputes_count     INTEGER NOT NULL,
    is_fraud                 INTEGER NOT NULL CHECK (is_fraud IN (0, 1)),
    dispute_status           TEXT NOT NULL       -- 'Fraudulent' / 'Legitimate'
);

CREATE INDEX idx_customer_id ON disputed_transactions(customer_id);
CREATE INDEX idx_merchant_category ON disputed_transactions(merchant_category);
CREATE INDEX idx_dispute_status ON disputed_transactions(dispute_status);
