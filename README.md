# Credit Risk Default Prediction & Scorecard

A complete, interpretable credit-risk workflow on a lending portfolio: data
diagnostics, default-driver analysis, a logistic-regression probability-of-default
(PD) model, and a 300–850 style risk scorecard that segments borrowers into grades A–E.

Built in Python as a practical demonstration of credit analytics — the same domain
as my M.Com thesis and final-year project on credit ratings.

---

## Business question

> Which borrowers are most likely to default, *why*, and how do we translate that
> into a lending decision a credit committee can act on?

## What the project does

1. **Builds a realistic lending portfolio** of 3,000 borrowers with income, leverage,
   utilisation, payment history, and tenure attributes. *(Swap in a real CSV to reuse
   the pipeline on actual data.)*
2. **Diagnoses default drivers** through correlation analysis and default-rate banding.
3. **Trains a PD model** (logistic regression with balanced class weights) and evaluates
   it on a hold-out set with ROC-AUC and a full classification report.
4. **Produces a risk scorecard** mapping each borrower's PD to a score and grade (A–E),
   then validates that grades line up with actual default rates.

## Key findings

| Risk grade | Borrowers | Actual default rate |
|------------|-----------|---------------------|
| A (Low risk)  | 994  | ~6%  |
| B             | 1,196 | ~15% |
| C             | 599  | ~23% |
| D             | 194  | ~42% |
| E (High risk) | 17   | ~47% |

- **Debt-to-income is the single strongest driver of default**, followed by credit
  utilisation and number of late payments.
- Longer employment tenure and credit history **reduce** default risk.
- The scorecard cleanly separates risk — Grade A borrowers default ~8x less often than
  Grade E, which is exactly the discrimination a lender needs to price and approve loans.

## Selected visuals

| Default driver influence | Default rate by leverage |
|--------------------------|--------------------------|
| ![](images/04_feature_influence.png) | ![](images/02_default_by_dti.png) |

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn`

## How to run

```bash
pip install -r requirements.txt
python credit_risk_analysis.py
```

Outputs are written to `images/` (charts) and `data/` (portfolio + scorecard summary).

## Repository structure

```
credit-risk-scorecard/
├── credit_risk_analysis.py     # end-to-end pipeline
├── requirements.txt
├── data/                       # generated portfolio + scorecard summary
└── images/                     # charts produced by the script
```

## Possible extensions

- Replace synthetic data with a public dataset (e.g. the German Credit dataset).
- Add a gradient-boosting model and compare AUC against the logistic baseline.
- Build a Power BI dashboard on top of the scorecard output.

---

*Author: Sheetal Virendra Gupta — M.Sc. Finance candidate, EBS Universität.*
