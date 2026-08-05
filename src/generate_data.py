"""
generate_data.py

Generates a synthetic financial transactions dataset that mimics the kind of
dispute/fraud review work described in the project's README (e.g., monthly
transaction disputes categorized as fraudulent or legitimate).

The data is entirely synthetic (no real customer or company data), but the
feature relationships are designed to reflect realistic fraud patterns:
- Larger transaction amounts have a higher chance of being fraudulent.
- Newer accounts are riskier than long-tenured accounts.
- Certain merchant categories (electronics, gift cards, crypto) see more fraud.
- Late-night transactions and foreign transactions carry more risk.
- Customers with prior disputes are more likely to have another one.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
N_TRANSACTIONS = 8000

rng = np.random.default_rng(RANDOM_SEED)

MERCHANT_CATEGORIES = [
    "Grocery", "Electronics", "Gift Cards", "Travel", "Dining",
    "Utilities", "Subscription", "Crypto/Investment", "Apparel", "Healthcare"
]
# Base fraud risk weight per category (higher = riskier)
CATEGORY_RISK = {
    "Grocery": 0.01, "Electronics": 0.06, "Gift Cards": 0.09, "Travel": 0.04,
    "Dining": 0.01, "Utilities": 0.005, "Subscription": 0.02,
    "Crypto/Investment": 0.10, "Apparel": 0.02, "Healthcare": 0.01,
}

CHANNELS = ["Online", "In-Store", "Phone/Call Center"]
CHANNEL_RISK = {"Online": 0.05, "In-Store": 0.01, "Phone/Call Center": 0.04}

COUNTRIES = ["PH", "US", "SG", "UK", "AE", "NG", "BR"]
DOMESTIC = "PH"

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)


def random_dates(n):
    span = (END_DATE - START_DATE).days
    offsets = rng.integers(0, span, size=n)
    hours = rng.integers(0, 24, size=n)
    minutes = rng.integers(0, 60, size=n)
    return [START_DATE + timedelta(days=int(o), hours=int(h), minutes=int(m))
            for o, h, m in zip(offsets, hours, minutes)]


def generate():
    n = N_TRANSACTIONS
    customer_id = rng.integers(10000, 10000 + int(n * 0.4), size=n)  # repeat customers
    amount = np.round(np.exp(rng.normal(4.2, 1.1, size=n)), 2)  # right-skewed, like real spend
    amount = np.clip(amount, 3.0, 15000.0)

    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=n)
    channel = rng.choice(CHANNELS, size=n, p=[0.55, 0.35, 0.10])
    country = rng.choice(COUNTRIES, size=n, p=[0.80, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01])
    account_age_days = rng.integers(1, 2600, size=n)
    prior_disputes = rng.poisson(0.15, size=n)
    txn_datetime = random_dates(n)
    hour = np.array([d.hour for d in txn_datetime])

    # ---- Compute fraud probability from realistic risk factors ----
    # A logistic (sigmoid) link is used so the overall fraud rate (intercept)
    # and the strength of separation between risk factors (weights) can be
    # tuned independently. Weights are intentionally strong so the underlying
    # patterns are learnable by a downstream model — this is a teaching /
    # portfolio dataset, not a claim about real-world fraud base rates.
    base_risk = np.array([CATEGORY_RISK[c] for c in merchant_category])
    channel_risk = np.array([CHANNEL_RISK[c] for c in channel])
    amount_z = (amount - amount.mean()) / amount.std()
    new_account_flag = (account_age_days < 60).astype(float)
    foreign_flag = (country != DOMESTIC).astype(float)
    late_night_flag = ((hour >= 0) & (hour <= 4)).astype(float)

    INTERCEPT = -3.1  # calibrated below to land near a ~20% overall fraud rate
    score = (
        INTERCEPT
        + 14.0 * base_risk
        + 10.0 * channel_risk
        + 0.55 * amount_z
        + 1.7 * new_account_flag
        + 1.4 * foreign_flag
        + 1.1 * late_night_flag
        + 0.65 * prior_disputes
    )
    fraud_prob = 1 / (1 + np.exp(-score))

    is_fraud = rng.binomial(1, fraud_prob)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{100000 + i}" for i in range(n)],
        "customer_id": customer_id,
        "transaction_datetime": txn_datetime,
        "amount": amount,
        "merchant_category": merchant_category,
        "channel": channel,
        "country": country,
        "account_age_days": account_age_days,
        "prior_disputes_count": prior_disputes,
        "is_fraud": is_fraud,
    })

    df["dispute_status"] = np.where(df["is_fraud"] == 1, "Fraudulent", "Legitimate")
    df = df.sort_values("transaction_datetime").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/fraud-dispute-detection/data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} transactions -> {out_path}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(df.head())
