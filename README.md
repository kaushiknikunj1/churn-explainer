# Churn Explainer

My churn model already tells me *who* is about to leave. For years that was where the hand-off happened: a ranked list lands with the account team, and the account team asks the only question that matters — *why this one, and what do I do?* — and gets a SHAP plot in reply.

This is the missing last step. It turns per-customer SHAP values into a two-line brief a human can act on, and it does it without letting the language model touch the numbers.

```
── C103112  [template (no API key)]
92.4% churn risk, driven by failed payments in the last 90 days = 3 (median 0); months as a customer = 5 (median 36).
Next step: Fix the payment method first - offer a one-click card update and waive the retry fee.
```

That is the deterministic template. With an API key the same evidence pack goes to an LLM, which rewrites it in plainer English — and is then checked.

## How it works

1. **Model.** XGBoost on synthetic fintech-style customer data (5,000 customers, 8 features). AUC ≈ 0.79. The data is synthetic — no employer data — but the feature shape is the one I've used in production: tenure, usage, support load, payment failures, product depth, discount status.
2. **Explain.** `shap.TreeExplainer` gives per-customer contributions. For each high-risk customer I assemble an *evidence pack*: risk %, the three strongest drivers, each driver's value and the population median.
3. **Write.** The evidence pack goes to an LLM with a strict brief: plain English, two lines, one action, **use only the numbers provided**.
4. **Verify.** Every number in the LLM's output is parsed and checked against the evidence pack. If it invented or re-rounded anything, the brief is discarded and a deterministic template is used instead. The output says which path produced it.

The point of step 4 is the whole project. An LLM that paraphrases a model is useful. An LLM that *reports* a model's numbers must be checked, because it will eventually say "94%" when the model said 92.4%, and nobody downstream will know.

## Run it

```bash
pip install -r requirements.txt
python churn_explainer.py --top 8
```

Works with no API key — you get the template briefs. For LLM-written briefs set one of:

```bash
export GEMINI_API_KEY=...        # preferred; uses gemini-2.5-flash via REST
export ANTHROPIC_API_KEY=...     # uses the official anthropic SDK, claude-opus-5-5
```

Outputs: `briefs.md` (briefs + collapsible evidence packs) and `shap_summary.png`.

## What I'd change for production

- Replace the template `ACTIONS` map with the retention team's actual playbook. The map is the part that encodes judgement; the model only ranks.
- Add a holdout: brief half the at-risk list, leave half alone, measure retention lift. A brief that doesn't change behaviour is a nicer-looking SHAP plot.
- Pin the evidence pack schema and log it next to the brief. When a brief is wrong you want to know whether the model or the writer was wrong.

Companion case study: [User Churn Prediction Model](https://nikunjkaushik.com/projects/churn-prediction.html). Part of the [Building with AI](https://nikunjkaushik.com/projects.html) set on my site.
