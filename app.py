"""
app.py

Streamlit dashboard for the Fraud & Dispute Detection project.
Loads data/transactions.csv into an in-memory SQLite database (using
sql/schema.sql), runs the queries in sql/analysis_queries.sql, and serves a
Random Forest model that scores new disputes for fraud risk.

Run locally:  streamlit run app.py
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "transactions.csv"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"
QUERIES_PATH = ROOT / "sql" / "analysis_queries.sql"

FEATURES_NUM = ["amount", "account_age_days", "prior_disputes_count", "hour"]
FEATURES_CAT = ["merchant_category", "channel", "country"]
HIGH_RISK, MEDIUM_RISK = 0.6, 0.4

st.set_page_config(page_title="Fraud & Dispute Detection", page_icon="🛡️", layout="wide")


# ---------- Data & model (cached so they build once per server) ----------

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=["transaction_datetime"])
    df["hour"] = df["transaction_datetime"].dt.hour
    df["month"] = df["transaction_datetime"].dt.to_period("M").astype(str)
    return df


@st.cache_resource
def get_connection():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(SCHEMA_PATH.read_text())
    pd.read_csv(CSV_PATH).to_sql("disputed_transactions", conn, if_exists="append", index=False)
    conn.commit()
    return conn


@st.cache_data
def load_queries():
    # Each query is preceded by a "-- N. Title" comment line
    sql_text = QUERIES_PATH.read_text()
    queries = {}
    for block in re.split(r"\n(?=--\s*\d+\.)", sql_text):
        m = re.match(r"--\s*(\d+\..*)\n", block)
        if m:
            queries[m.group(1).strip()] = block[m.end():].strip()
    return queries


@st.cache_resource
def train_model():
    df = load_data()
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline(steps=[
        ("preprocess", ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ], remainder="passthrough")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "auc": roc_auc_score(y_test, proba),
        "recall": recall_score(y_test, model.predict(X_test)),
    }

    # Held-out disputes ranked by risk: the order a review team would work them
    queue = df.loc[X_test.index].assign(risk_score=proba)
    queue = queue.sort_values("risk_score", ascending=False).reset_index(drop=True)
    queue["risk_level"] = queue["risk_score"].map(risk_level)
    return model, metrics, queue


def risk_level(score):
    if score >= HIGH_RISK:
        return "High"
    if score >= MEDIUM_RISK:
        return "Medium"
    return "Low"


def run_sql(sql):
    return pd.read_sql_query(sql, get_connection())


def fraud_rate_by(df, column):
    return (df.groupby(column)["is_fraud"].mean() * 100).round(2).sort_values(ascending=False)


df = load_data()
model, metrics, queue = train_model()

# ---------- Header ----------

st.title("🛡️ Fraud & Dispute Detection")
st.caption(
    "Analysis of 8,000 disputed transactions, each reviewed as Fraudulent or Legitimate. "
    "The dataset is synthetic; no real customer data is used."
)

tab_overview, tab_queue, tab_score, tab_sql = st.tabs(
    ["📊 Overview", "📋 Review Queue", "🔍 Risk Scorer", "🗄️ SQL Explorer"]
)

# ---------- Overview ----------

with tab_overview:
    new_acct = df[df["account_age_days"] < 60]["is_fraud"].mean() * 100
    foreign = df[df["country"] != "PH"]["is_fraud"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Disputes reviewed", f"{len(df):,}")
    c2.metric("Overall fraud rate", f"{df['is_fraud'].mean() * 100:.1f}%")
    c3.metric("New accounts (<60 days)", f"{new_acct:.1f}%")
    c4.metric("Foreign transactions", f"{foreign:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Fraud rate by merchant category (%)")
        st.bar_chart(fraud_rate_by(df, "merchant_category"), horizontal=True, color="#C44E52")
    with right:
        st.subheader("Monthly fraud rate (%)")
        monthly = (df.groupby("month")["is_fraud"].mean() * 100).round(2)
        st.line_chart(monthly, color="#C44E52")

    segments = df.assign(
        account=lambda d: d["account_age_days"].lt(60).map({True: "New (<60d)", False: "Established"}),
        geography=lambda d: d["country"].eq("PH").map({True: "Domestic", False: "Foreign"}),
        time_of_day=lambda d: d["hour"].between(0, 4).map({True: "Late night (12-4AM)", False: "Rest of day"}),
    )
    st.subheader("Risk factors (fraud rate %)")
    cols = st.columns(4)
    for col, (label, field) in zip(cols, [
        ("Account age", "account"), ("Channel", "channel"),
        ("Geography", "geography"), ("Time of day", "time_of_day"),
    ]):
        with col:
            st.markdown(f"**{label}**")
            st.bar_chart(fraud_rate_by(segments, field), color="#4C72B0", height=250)

# ---------- Review Queue ----------

with tab_queue:
    st.markdown(
        f"The model scored **{len(queue):,} disputes it never saw during training** and sorted them "
        "riskiest first. This is the order a review team would work them in, instead of first-come, "
        "first-served."
    )
    capacity = st.slider("Review capacity: % of the queue your team can get through", 5, 100, 20, step=5)

    total_fraud = queue["is_fraud"].sum()
    n_review = round(len(queue) * capacity / 100)
    caught = queue["is_fraud"].iloc[:n_review].sum()
    caught_pct = caught / total_fraud * 100
    base_rate = total_fraud / len(queue) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Disputes reviewed", f"{n_review:,} of {len(queue):,}")
    c2.metric(
        "Share of all fraud caught", f"{caught_pct:.0f}%",
        delta=f"{caught_pct - capacity:+.0f} pts vs. arrival order",
    )
    c3.metric(
        "Fraud rate in reviewed cases", f"{caught / n_review * 100:.0f}%",
        delta=f"{caught / n_review * 100 - base_rate:+.0f} pts vs. {base_rate:.0f}% overall",
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Fraud caught vs. disputes reviewed")
        reviewed = np.arange(1, len(queue) + 1) / len(queue) * 100
        gains = pd.DataFrame({
            "% of queue reviewed": reviewed,
            "Ranked by model": queue["is_fraud"].cumsum() / total_fraud * 100,
            "Arrival order (no model)": reviewed,
        }).iloc[::20]
        st.line_chart(
            gains, x="% of queue reviewed", y=["Ranked by model", "Arrival order (no model)"],
            y_label="% of all fraud caught", color=["#C44E52", "#999999"], height=320,
        )
    with right:
        st.subheader("Risk levels")
        levels = (
            queue.groupby("risk_level")
            .agg(disputes=("is_fraud", "size"), actual_fraud_rate=("is_fraud", "mean"))
            .reindex(["High", "Medium", "Low"])
            .reset_index()
        )
        levels["actual_fraud_rate"] *= 100
        st.dataframe(
            levels, hide_index=True, width="stretch",
            column_config={
                "risk_level": "Risk level",
                "disputes": "Disputes",
                "actual_fraud_rate": st.column_config.NumberColumn("Actual fraud rate", format="%.0f%%"),
            },
        )
        st.caption(
            f"High = score ≥ {HIGH_RISK:.0%}, Medium = {MEDIUM_RISK:.0%}–{HIGH_RISK:.0%}, "
            f"Low = below {MEDIUM_RISK:.0%}."
        )

    st.subheader(f"Your team's queue: top {n_review:,} disputes")
    show = queue.head(n_review).assign(
        rank=lambda d: d.index + 1, risk_score=lambda d: d["risk_score"] * 100
    )
    st.dataframe(
        show[[
            "rank", "transaction_id", "risk_score", "risk_level", "amount", "merchant_category",
            "channel", "country", "account_age_days", "prior_disputes_count", "hour", "dispute_status",
        ]],
        hide_index=True, width="stretch", height=400,
        column_config={
            "rank": "Rank",
            "transaction_id": "Transaction",
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", format="%.0f%%", min_value=0, max_value=100
            ),
            "risk_level": "Risk level",
            "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
            "merchant_category": "Merchant category",
            "channel": "Channel",
            "country": "Country",
            "account_age_days": "Account age (days)",
            "prior_disputes_count": "Prior disputes",
            "hour": "Hour",
            "dispute_status": "Actual outcome",
        },
    )
    st.caption(
        "\"Actual outcome\" is the real review decision, shown only to check the ranking. "
        "In live use it wouldn't be known until the dispute is reviewed."
    )

# ---------- SQL Explorer ----------

with tab_sql:
    st.markdown(
        "The data is loaded into SQLite using `sql/schema.sql`. "
        "Pick one of the analytical queries from `sql/analysis_queries.sql`, or write your own."
    )
    queries = load_queries()
    choice = st.selectbox("Query", list(queries))
    st.code(queries[choice], language="sql")
    st.dataframe(run_sql(queries[choice]), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Write your own query")
    custom = st.text_area(
        "Table: disputed_transactions",
        "SELECT channel, merchant_category, COUNT(*) AS n, SUM(is_fraud) AS fraud\n"
        "FROM disputed_transactions\nGROUP BY 1, 2\nORDER BY fraud DESC\nLIMIT 10;",
        height=150,
    )
    if st.button("Run query"):
        if not re.match(r"^\s*(SELECT|WITH)\b", custom, re.IGNORECASE):
            st.error("Only SELECT queries are allowed.")
        else:
            try:
                st.dataframe(run_sql(custom), width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"Query failed: {e}")

# ---------- Risk Scorer ----------

with tab_score:
    st.markdown(
        "Score a new dispute with the Random Forest model from the notebook "
        f"(test ROC-AUC **{metrics['auc']:.2f}**, fraud recall **{metrics['recall']:.0%}**)."
    )

    with st.form("score"):
        a, b = st.columns(2)
        with a:
            amount = st.number_input("Amount", min_value=0.0, value=250.0, step=10.0)
            category = st.selectbox("Merchant category", sorted(df["merchant_category"].unique()))
            channel = st.selectbox("Channel", sorted(df["channel"].unique()))
            countries = sorted(df["country"].unique())
            country = st.selectbox("Country", countries, index=countries.index("PH"))
        with b:
            account_age = st.number_input("Account age (days)", min_value=0, value=365, step=1)
            prior = st.number_input("Prior disputes", min_value=0, value=0, step=1)
            hour = st.slider("Hour of transaction (0-23)", 0, 23, 14)
        submitted = st.form_submit_button("Score this dispute", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "amount": amount, "account_age_days": account_age,
            "prior_disputes_count": prior, "hour": hour,
            "merchant_category": category, "channel": channel, "country": country,
        }])
        proba = model.predict_proba(row[FEATURES_NUM + FEATURES_CAT])[0, 1]
        level = risk_level(proba)
        rank = (queue["risk_score"] > proba).sum() + 1
        message = (
            f"**{level} risk:** {proba:.0%} risk score. It would rank #{rank:,} of "
            f"{len(queue):,} in the review queue."
        )
        {"High": st.error, "Medium": st.warning, "Low": st.success}[level](message)
        st.progress(float(proba))
        st.caption(
            "The model uses class_weight='balanced', so scores are relative risk "
            "rankings rather than calibrated probabilities."
        )
