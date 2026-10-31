#!/usr/bin/env python3
"""Churn Explainer - from "who will leave" to "why, and what to do about it".

1. Train an XGBoost churn model on synthetic fintech-style customer data.
2. Compute SHAP values per customer.
3. For the highest-risk customers, assemble an *evidence pack*: risk score,
   top drivers, each customer's value vs the population median.
4. Turn the evidence into a two-line retention brief an account manager can act on.
   With GEMINI_API_KEY or ANTHROPIC_API_KEY set, an LLM writes the sentence - but
   every number it uses is checked against the evidence pack. If it invents one,
   the brief falls back to the deterministic template. Without a key, you get the
   template. Either way the numbers come from the model, never from the LLM.

Run:  python churn_explainer.py [--top 8]
Out:  briefs.md, shap_summary.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import llm  # noqa: E402

RNG = np.random.default_rng(42)

FEATURES = {
    "tenure_months":        "months as a customer",
    "monthly_txns":         "transactions in the last 30 days",
    "avg_txn_value":        "average transaction value (INR)",
    "support_tickets_90d":  "support tickets in the last 90 days",
    "days_since_login":     "days since last login",
    "failed_payments_90d":  "failed payments in the last 90 days",
    "products_held":        "products held",
    "on_discount":          "on a promotional discount (1 = yes)",
}


def make_data(n: int = 5000) -> pd.DataFrame:
    d = pd.DataFrame({
        "tenure_months":       RNG.integers(1, 72, n),
        "monthly_txns":        RNG.poisson(14, n),
        "avg_txn_value":       np.round(RNG.lognormal(6.3, 0.6, n), 0),
        "support_tickets_90d": RNG.poisson(0.6, n),
        "days_since_login":    RNG.integers(0, 60, n),
        "failed_payments_90d": RNG.poisson(0.5, n),
        "products_held":       RNG.integers(1, 5, n),
        "on_discount":         RNG.integers(0, 2, n),
    })
    logit = (-2.2
             - 0.03 * d.tenure_months
             - 0.06 * d.monthly_txns
             + 0.55 * d.support_tickets_90d
             + 0.05 * d.days_since_login
             + 0.70 * d.failed_payments_90d
             - 0.35 * d.products_held
             + 0.60 * d.on_discount)
    p = 1 / (1 + np.exp(-logit))
    d["churned"] = (RNG.random(n) < p).astype(int)
    d.insert(0, "customer_id", [f"C{100000 + i}" for i in range(n)])
    return d


def train(d: pd.DataFrame):
    X, y = d[list(FEATURES)], d["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9, eval_metric="logloss")
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    return model, auc


def evidence_for(row: pd.Series, shap_row: np.ndarray, prob: float,
                 medians: pd.Series, k: int = 3) -> dict:
    order = np.argsort(-np.abs(shap_row))[:k]
    drivers = []
    for i in order:
        f = list(FEATURES)[i]
        drivers.append({
            "feature": f,
            "label": FEATURES[f],
            "value": float(row[f]),
            "population_median": float(medians[f]),
            "pushes_risk": "up" if shap_row[i] > 0 else "down",
            "shap": round(float(shap_row[i]), 3),
        })
    return {"customer_id": row["customer_id"], "churn_risk_pct": round(prob * 100, 1),
            "drivers": drivers}


def template_brief(ev: dict) -> str:
    ups = [d for d in ev["drivers"] if d["pushes_risk"] == "up"]
    downs = [d for d in ev["drivers"] if d["pushes_risk"] == "down"]

    def phrase(d):
        v, m = d["value"], d["population_median"]
        v_s = f"{v:g}"
        m_s = f"{m:g}"
        return f"{d['label']} = {v_s} (median {m_s})"

    line1 = f"{ev['churn_risk_pct']}% churn risk, driven by " + "; ".join(phrase(d) for d in ups[:2]) + "."
    if downs:
        line1 += f" Holding risk down: {phrase(downs[0])}."
    action = ACTIONS.get(ups[0]["feature"], "Schedule a check-in call this week.") if ups else \
        "Low-touch: add to the next-quarter nurture list."
    return f"{line1}\nNext step: {action}"


ACTIONS = {
    "failed_payments_90d": "Fix the payment method first - offer a one-click card update and waive the retry fee.",
    "days_since_login":    "Re-engagement nudge today, with a reason to log in (pending settlement, new report).",
    "support_tickets_90d": "Have a senior agent own the open tickets and close the loop personally.",
    "on_discount":         "The discount is ending soon - present the full-price value before it lapses.",
    "monthly_txns":        "Usage is thin - a 15-minute onboarding refresher usually lifts it.",
    "tenure_months":       "New account - prioritise first-90-day activation over upsell.",
    "products_held":       "Single-product customer - cross-sell one adjacent product, not a bundle.",
}


SYSTEM = """You write two-line retention briefs for account managers at a payments company.
Rules:
- Line 1: the risk and the two main drivers, plain English, no jargon, under 40 words.
- Line 2: starts with "Next step:" and gives ONE concrete action.
- Use ONLY the numbers in the evidence. Never compute, round differently, or add new numbers.
- Do not mention SHAP, models, or machine learning."""


def llm_brief(ev: dict) -> tuple[str, str]:
    """Returns (brief, source). Source is 'llm' or 'template' (+reason)."""
    text = llm.complete(SYSTEM, json.dumps(ev, indent=2), max_tokens=200)
    if text is None:
        return template_brief(ev), "template (no API key)"
    bad = llm.unpinned(text, ev)
    if bad:
        return template_brief(ev), f"template (LLM invented numbers: {sorted(bad)})"
    return text, f"llm ({llm.provider()})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8, help="how many at-risk customers to brief")
    args = ap.parse_args()

    d = make_data()
    model, auc = train(d)
    X = d[list(FEATURES)]
    probs = model.predict_proba(X)[:, 1]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    medians = X.median()

    print(f"trained XGBoost on {len(d):,} customers · churn rate {d.churned.mean():.1%} · AUC {auc:.3f}")
    print(f"LLM provider: {llm.provider()}\n")

    top = np.argsort(-probs)[: args.top]
    out = ["# Retention briefs", "",
           f"XGBoost churn model, AUC {auc:.3f}. SHAP drivers per customer. "
           f"Briefs written by: {llm.provider()}; every number verified against the evidence pack.", ""]
    for i in top:
        ev = evidence_for(d.iloc[i], sv[i], probs[i], medians)
        brief, src = llm_brief(ev)
        print(f"── {ev['customer_id']}  [{src}]")
        print(brief, "\n")
        out += [f"## {ev['customer_id']}", "", brief, "", f"*source: {src}*", "",
                "<details><summary>evidence</summary>", "", "```json",
                json.dumps(ev, indent=2), "```", "</details>", ""]
    Path("briefs.md").write_text("\n".join(out), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(sv, X, show=False, plot_size=(9, 5))
    plt.title("What drives churn risk (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=130)
    print("wrote briefs.md and shap_summary.png")


if __name__ == "__main__":
    main()
