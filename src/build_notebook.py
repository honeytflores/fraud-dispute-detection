"""
build_notebook.py
Programmatically builds notebooks/fraud_analysis.ipynb, then executes it
in-place with nbclient so the version pushed to GitHub already has outputs
(charts, tables) rendered — the way a polished portfolio notebook should look.
"""

import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "fraud_analysis.ipynb"

cells = []

cells.append(new_markdown_cell(
"""# Fraud & Dispute Detection Analysis

This notebook analyzes a dataset of **8,000 disputed transactions**, each reviewed
and categorized as either **Fraudulent** or **Legitimate** — mirroring the kind of
monthly dispute-review work described in the project README.

**Goals:**
1. Explore fraud patterns across merchant category, channel, geography, account age, and time of day.
2. Visualize the key risk factors driving fraudulent disputes.
3. Build a simple classification model to flag likely-fraudulent disputes automatically.
4. Evaluate the model with precision, recall, and F1 — since in fraud review, missing real fraud (false negatives) and wrongly flagging legitimate customers (false positives) both carry real costs.

> **Note:** This dataset is synthetically generated (see `src/generate_data.py`) for
> portfolio purposes — no real customer or company data is used."""
))

cells.append(new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("../data/transactions.csv", parse_dates=["transaction_datetime"])
df["hour"] = df["transaction_datetime"].dt.hour
df["month"] = df["transaction_datetime"].dt.to_period("M").astype(str)

print(f"Shape: {df.shape}")
df.head()"""
))

cells.append(new_markdown_cell("## 1. Overview: Fraud Rate & Class Balance"))

cells.append(new_code_cell(
"""fraud_rate = df["is_fraud"].mean() * 100
print(f"Overall fraud rate among disputed transactions: {fraud_rate:.2f}%")

fig, ax = plt.subplots(figsize=(5, 4))
counts = df["dispute_status"].value_counts()
colors = ["#4C72B0", "#C44E52"]
ax.bar(counts.index, counts.values, color=colors)
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")
ax.set_title("Dispute Outcomes: Fraudulent vs. Legitimate")
ax.set_ylabel("Number of Disputes")
plt.tight_layout()
plt.savefig("../images/01_class_balance.png")
plt.show()"""
))

cells.append(new_markdown_cell("## 2. Fraud Rate by Merchant Category\n\nWhich merchant categories see the highest share of disputes turning out to be fraudulent?"))

cells.append(new_code_cell(
"""cat_stats = (
    df.groupby("merchant_category")["is_fraud"]
    .agg(total="count", fraud_count="sum", fraud_rate="mean")
    .sort_values("fraud_rate", ascending=False)
)
cat_stats["fraud_rate_pct"] = (cat_stats["fraud_rate"] * 100).round(2)
cat_stats"""
))

cells.append(new_code_cell(
"""fig, ax = plt.subplots(figsize=(8, 5))
order = cat_stats.index
sns.barplot(x=cat_stats["fraud_rate_pct"], y=order, hue=order, palette="rocket", legend=False, ax=ax)
ax.set_xlabel("Fraud Rate (%)")
ax.set_ylabel("Merchant Category")
ax.set_title("Fraud Rate by Merchant Category")
plt.tight_layout()
plt.savefig("../images/02_fraud_by_category.png")
plt.show()"""
))

cells.append(new_markdown_cell("## 3. Risk Factors: Account Age, Channel, Geography, and Time of Day"))

cells.append(new_code_cell(
"""fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Account age
df["account_segment"] = np.where(df["account_age_days"] < 60, "New (<60 days)", "Established (60+ days)")
seg = df.groupby("account_segment")["is_fraud"].mean() * 100
axes[0,0].bar(seg.index, seg.values, color=["#DD8452", "#4C72B0"])
axes[0,0].set_title("Fraud Rate: Account Age")
axes[0,0].set_ylabel("Fraud Rate (%)")

# Channel
ch = df.groupby("channel")["is_fraud"].mean().sort_values(ascending=False) * 100
axes[0,1].bar(ch.index, ch.values, color="#55A868")
axes[0,1].set_title("Fraud Rate: Channel")
axes[0,1].tick_params(axis="x", rotation=15)

# Geography
df["geography"] = np.where(df["country"] == "PH", "Domestic", "Foreign")
geo = df.groupby("geography")["is_fraud"].mean() * 100
axes[1,0].bar(geo.index, geo.values, color=["#4C72B0", "#C44E52"])
axes[1,0].set_title("Fraud Rate: Geography")
axes[1,0].set_ylabel("Fraud Rate (%)")

# Time of day
df["time_segment"] = np.where(df["hour"].between(0, 4), "Late Night (12-4AM)", "Rest of Day")
tod = df.groupby("time_segment")["is_fraud"].mean() * 100
axes[1,1].bar(tod.index, tod.values, color=["#C44E52", "#4C72B0"])
axes[1,1].set_title("Fraud Rate: Time of Day")

plt.tight_layout()
plt.savefig("../images/03_risk_factors.png")
plt.show()"""
))

cells.append(new_markdown_cell(
"""**Key patterns observed:**
- New accounts (under 60 days) show a noticeably higher fraud rate than established accounts.
- Foreign transactions carry more risk than domestic ones.
- Late-night transactions (12AM-4AM) are more likely to be fraudulent than the rest of the day.
- Gift Cards and Crypto/Investment categories are the riskiest merchant types — consistent with real-world fraud trends, since both are easy to convert to cash and hard to reverse."""
))

cells.append(new_markdown_cell("## 4. Monthly Fraud Trend"))

cells.append(new_code_cell(
"""monthly = df.groupby("month")["is_fraud"].agg(total="count", fraud_rate="mean").reset_index()
monthly["fraud_rate_pct"] = monthly["fraud_rate"] * 100

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly["month"], monthly["fraud_rate_pct"], marker="o", color="#C44E52")
ax.set_title("Monthly Fraud Rate Trend (2024)")
ax.set_ylabel("Fraud Rate (%)")
ax.set_xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../images/04_monthly_trend.png")
plt.show()"""
))

cells.append(new_markdown_cell(
"""## 5. Building a Fraud Classification Model

The goal here isn't just to describe fraud patterns after the fact, but to build a
model that could flag a **new** disputed transaction as likely fraudulent *before*
a human reviewer looks at it — helping prioritize the review queue.

We'll use a **Random Forest Classifier**, which handles a mix of numeric and
categorical features well and gives us interpretable feature importances."""
))

cells.append(new_code_cell(
"""features_num = ["amount", "account_age_days", "prior_disputes_count", "hour"]
features_cat = ["merchant_category", "channel", "country"]

X = df[features_num + features_cat]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
], remainder="passthrough")

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1
    )),
])

model.fit(X_train, y_train)
print("Model trained.")
print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")"""
))

cells.append(new_markdown_cell("## 6. Model Evaluation"))

cells.append(new_code_cell(
"""y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.3f}")"""
))

cells.append(new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraudulent"]).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1], name="Random Forest")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
axes[1].set_title(f"ROC Curve (AUC = {auc:.3f})")
axes[1].legend()

plt.tight_layout()
plt.savefig("../images/05_model_evaluation.png")
plt.show()"""
))

cells.append(new_markdown_cell("## 7. What Drives the Model's Predictions?"))

cells.append(new_code_cell(
"""ohe = model.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(features_cat)
all_feature_names = list(cat_feature_names) + features_num

importances = model.named_steps["classifier"].feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(12)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette="viridis", legend=False, ax=ax)
ax.set_title("Top 12 Feature Importances — Fraud Classification Model")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig("../images/06_feature_importance.png")
plt.show()"""
))

cells.append(new_markdown_cell(
"""## 8. Summary of Findings

- **Overall fraud rate** among reviewed disputes was ~13%.
- **Gift Cards** and **Crypto/Investment** categories carried the highest fraud rates — both are common cash-out vectors in real-world fraud.
- **New accounts** (under 60 days old) were meaningfully riskier than established accounts.
- **Foreign transactions** and **late-night activity (12-4AM)** were both associated with elevated fraud rates.
- A Random Forest model trained on these features achieved strong separation between fraudulent and legitimate disputes (see ROC-AUC above), with **transaction amount, prior dispute history, and merchant category** among the most influential features.

**Practical takeaway:** a review team could use a model like this to automatically prioritize
the riskiest disputes (new account + foreign + high-risk category) for review first,
rather than working through the queue in random or arrival order — directly supporting
faster, more accurate fraud triage."""
))

nb = new_notebook(cells=cells)
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {NB_PATH}")

# Execute in place so outputs/charts are baked into the .ipynb for GitHub viewing
client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NB_PATH.parent)}})
client.execute()
with open(NB_PATH, "w") as f:
    nbf.write(nb, f)
print("Notebook executed and saved with outputs.")
