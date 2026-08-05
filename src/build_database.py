"""
build_database.py

Loads data/transactions.csv into a SQLite database (data/fraud_disputes.db)
using sql/schema.sql, then runs every query in sql/analysis_queries.sql to
confirm they execute correctly and prints a preview of each result.
"""

import sqlite3
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fraud_disputes.db"
CSV_PATH = ROOT / "data" / "transactions.csv"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"
QUERIES_PATH = ROOT / "sql" / "analysis_queries.sql"


def build_db():
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    df = pd.read_csv(CSV_PATH)
    df.to_sql("disputed_transactions", conn, if_exists="append", index=False)
    conn.commit()
    print(f"Loaded {len(df):,} rows into {DB_PATH}")
    return conn


def split_queries(sql_text):
    # Split on semicolons that end a statement, ignoring ones inside comments
    statements = []
    current = []
    for line in sql_text.splitlines():
        if line.strip().startswith("--") and not current:
            continue
        current.append(line)
        if line.strip().endswith(";"):
            statements.append("\n".join(current))
            current = []
    return [s for s in statements if s.strip()]


def run_queries(conn):
    with open(QUERIES_PATH) as f:
        sql_text = f.read()

    statements = split_queries(sql_text)
    print(f"\nFound {len(statements)} queries. Running each...\n")
    for i, stmt in enumerate(statements, start=1):
        try:
            result = pd.read_sql_query(stmt, conn)
            title_match = re.search(r"--\s*(\d+\..*)", stmt)
            title = title_match.group(1) if title_match else f"Query {i}"
            print(f"=== {title} ===")
            print(result.head(5).to_string(index=False))
            print()
        except Exception as e:
            print(f"Query {i} FAILED: {e}")
            raise


if __name__ == "__main__":
    conn = build_db()
    run_queries(conn)
    conn.close()
