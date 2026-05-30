"""
Credit Risk Default Prediction & Scorecard
-------------------------------------------
End-to-end credit risk analysis on a B2B/retail lending portfolio:
data generation, exploratory analysis, default-driver diagnostics,
a logistic-regression scoring model, and a simple risk scorecard.

Author: Sheetal Virendra Gupta
Stack: Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

RNG = np.random.default_rng(42)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------
# 1. BUILD A REALISTIC SYNTHETIC LENDING PORTFOLIO
#    (replace this block with a real CSV when you have one)
# ---------------------------------------------------------------------------
N = 3000

age          = RNG.normal(40, 11, N).clip(21, 70).round()
income       = RNG.lognormal(mean=10.8, sigma=0.45, size=N).round(-2)        # annual income
loan_amount  = (income * RNG.uniform(0.15, 0.9, N)).round(-2)
employment_y = RNG.gamma(shape=2.2, scale=3.0, size=N).clip(0, 35).round(1)
dti          = (loan_amount / income).clip(0.02, 1.4).round(3)               # debt-to-income
credit_util  = RNG.beta(2, 4, N).round(3)                                     # revolving utilisation
num_late_pmt = RNG.poisson(0.7, N)
credit_hist  = RNG.gamma(shape=3, scale=3, size=N).clip(0.5, 30).round(1)     # years of history

# latent default risk: higher dti, utilisation, late payments raise risk;
# higher income, employment tenure and history lower it
logit = (
    -3.4
    + 2.6 * dti
    + 1.9 * credit_util
    + 0.45 * num_late_pmt
    - 0.000004 * income
    - 0.06 * employment_y
    - 0.04 * credit_hist
    + RNG.normal(0, 0.5, N)
)
p_default = 1 / (1 + np.exp(-logit))
default   = (RNG.uniform(0, 1, N) < p_default).astype(int)

df = pd.DataFrame({
    "age": age, "annual_income": income, "loan_amount": loan_amount,
    "employment_years": employment_y, "debt_to_income": dti,
    "credit_utilisation": credit_util, "num_late_payments": num_late_pmt,
    "credit_history_years": credit_hist, "default": default,
})
df.to_csv("data/lending_portfolio.csv", index=False)

print(f"Portfolio size: {len(df):,} borrowers")
print(f"Overall default rate: {df['default'].mean():.1%}\n")

# ---------------------------------------------------------------------------
# 2. EXPLORATORY ANALYSIS — what drives default?
# ---------------------------------------------------------------------------
features = ["age", "annual_income", "loan_amount", "employment_years",
            "debt_to_income", "credit_utilisation", "num_late_payments",
            "credit_history_years"]

# 2a. Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df[features + ["default"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            cbar_kws={"shrink": .8})
plt.title("Correlation of Borrower Attributes with Default", fontsize=12, weight="bold")
plt.tight_layout()
plt.savefig("images/01_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# 2b. Default rate by debt-to-income band
df["dti_band"] = pd.qcut(df["debt_to_income"], 5,
                         labels=["Lowest", "Low", "Mid", "High", "Highest"])
band_rates = df.groupby("dti_band", observed=True)["default"].mean()
plt.figure(figsize=(7, 4.5))
band_rates.plot(kind="bar", color="#1F3864")
plt.ylabel("Default rate")
plt.xlabel("Debt-to-income band")
plt.title("Default Rate Rises Sharply with Leverage", fontsize=12, weight="bold")
plt.xticks(rotation=0)
for i, v in enumerate(band_rates):
    plt.text(i, v + 0.005, f"{v:.0%}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("images/02_default_by_dti.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 3. SCORING MODEL — logistic regression (interpretable, industry standard)
# ---------------------------------------------------------------------------
X = df[features]
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_s, y_train)

proba = model.predict_proba(X_test_s)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"Model ROC-AUC on hold-out set: {auc:.3f}\n")
print("Classification report (threshold = 0.5):")
print(classification_report(y_test, (proba >= 0.5).astype(int), digits=3))

# 3a. ROC curve
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#1F3864", lw=2, label=f"Model (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve — Default Classifier", fontsize=12, weight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("images/03_roc_curve.png", bbox_inches="tight")
plt.close()

# 3b. Feature influence (standardised coefficients)
coef = pd.Series(model.coef_[0], index=features).sort_values()
plt.figure(figsize=(7, 4.5))
colors = ["#C00000" if c > 0 else "#2E7D32" for c in coef]
coef.plot(kind="barh", color=colors)
plt.title("What Pushes a Borrower Toward Default", fontsize=12, weight="bold")
plt.xlabel("Standardised coefficient (red = raises risk)")
plt.tight_layout()
plt.savefig("images/04_feature_influence.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 4. RISK SCORECARD — translate probability into a 300–850 style score
# ---------------------------------------------------------------------------
all_proba = model.predict_proba(scaler.transform(X))[:, 1]
df["pd_estimate"] = all_proba
# map PD to a score: low PD -> high score
df["risk_score"] = (850 - (all_proba * 550)).round().astype(int)
df["risk_grade"] = pd.cut(df["risk_score"],
                          bins=[0, 580, 670, 740, 800, 850],
                          labels=["E (High risk)", "D", "C", "B", "A (Low risk)"])

grade_summary = (df.groupby("risk_grade", observed=True)
                   .agg(borrowers=("default", "size"),
                        actual_default_rate=("default", "mean"),
                        avg_pd=("pd_estimate", "mean"))
                   .round(3))
print("\nScorecard grade summary:")
print(grade_summary)
grade_summary.to_csv("data/scorecard_summary.csv")

print("\nDone. Charts saved to images/, data saved to data/.")
