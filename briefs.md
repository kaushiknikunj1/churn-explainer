# Retention briefs

XGBoost churn model, AUC 0.792. SHAP drivers per customer. Briefs written by: none; every number verified against the evidence pack.

## C100320

96.5% churn risk, driven by support tickets in the last 90 days = 3 (median 0); failed payments in the last 90 days = 2 (median 0).
Next step: Have a senior agent own the open tickets and close the loop personally.

*source: template (no API key)*

<details><summary>evidence</summary>

```json
{
  "customer_id": "C100320",
  "churn_risk_pct": 96.5,
  "drivers": [
    {
      "feature": "support_tickets_90d",
      "label": "support tickets in the last 90 days",
      "value": 3.0,
      "population_median": 0.0,
      "pushes_risk": "up",
      "shap": 2.281
    },
    {
      "feature": "failed_payments_90d",
      "label": "failed payments in the last 90 days",
      "value": 2.0,
      "population_median": 0.0,
      "pushes_risk": "up",
      "shap": 0.999
    },
    {
      "feature": "days_since_login",
      "label": "days since last login",
      "value": 46.0,
      "population_median": 29.0,
      "pushes_risk": "up",
      "shap": 0.673
    }
  ]
}
```
</details>

## C103112

92.4% churn risk, driven by failed payments in the last 90 days = 3 (median 0); months as a customer = 5 (median 36).
Next step: Fix the payment method first - offer a one-click card update and waive the retry fee.

*source: template (no API key)*

<details><summary>evidence</summary>

```json
{
  "customer_id": "C103112",
  "churn_risk_pct": 92.4,
  "drivers": [
    {
      "feature": "failed_payments_90d",
      "label": "failed payments in the last 90 days",
      "value": 3.0,
      "population_median": 0.0,
      "pushes_risk": "up",
      "shap": 1.506
    },
    {
      "feature": "tenure_months",
      "label": "months as a customer",
      "value": 5.0,
      "population_median": 36.0,
      "pushes_risk": "up",
      "shap": 0.91
    },
    {
      "feature": "days_since_login",
      "label": "days since last login",
      "value": 43.0,
      "population_median": 29.0,
      "pushes_risk": "up",
      "shap": 0.767
    }
  ]
}
```
</details>
