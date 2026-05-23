"""Top-25 Standard+ tier — reactive playground.

Mirrors what the Streamlit "Standard+ Application" page exposes, but as a
marimo notebook: every slider/input change re-runs all downstream cells
automatically so you can explore the model's behaviour live.

Run with:
    .venv/bin/marimo edit marimo/top25_squeeze.py

Sections:
  1. Load the production Top-25 model bundle (cached).
  2. Feature importance & gain ranking.
  3. Interactive "what-if" profile: tweak inputs, see risk score update.
  4. Insights (counter-factuals, cohort percentile, loan-affordability).
  5. Bundle metadata + reproducibility recipe.
"""
import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Top-25 Standard+ tier — reactive playground

        This notebook loads the production Standard+ model bundle and lets you
        explore predictions interactively. Slider/input changes propagate
        automatically through every downstream cell.

        **Production model:** tuned LightGBM on 25 self-reportable features +
        6 derived ratios. Holdout ROC-AUC = **0.7146** (Kaggle median ~0.75).
        Source bundle: `src/assets/top25_risk_model.pkl`.
        """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "marimo" else Path.cwd()
    sys.path.insert(0, str(PROJECT_ROOT))

    import pandas as pd
    import numpy as np
    import plotly.express as px

    from models.top25_predictor import Top25Predictor
    from models import insights

    return PROJECT_ROOT, Top25Predictor, insights, np, pd, px


@app.cell
def _(PROJECT_ROOT, Top25Predictor):
    predictor = Top25Predictor(PROJECT_ROOT / "src" / "assets" / "top25_risk_model.pkl")
    return (predictor,)


@app.cell
def _(mo, predictor):
    mo.md(
        f"""
        ### Bundle metadata

        - **Best model:** `{predictor.best_name}`
        - **Holdout ROC-AUC:** `{predictor.best_auc}`
        - **Total inputs:** {len(predictor.feature_set)} (25 selected + 6 derived ratios)
        - **Categorical features ({len(predictor.selected_categorical)}):** {", ".join(predictor.selected_categorical)}
        - **Numeric features ({len(predictor.selected_numeric)}):** {", ".join(predictor.selected_numeric)}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## 2. Feature importance (gain) — what drives the model?")
    return


@app.cell
def _(pd, predictor, px):
    importances = getattr(predictor.model, "feature_importances_", None)
    if importances is not None and len(importances) == len(predictor.feature_set):
        imp_df = (
            pd.DataFrame(
                {"feature": predictor.feature_set, "gain": importances}
            )
            .sort_values("gain", ascending=True)
            .tail(20)
        )
        fig = px.bar(
            imp_df,
            x="gain",
            y="feature",
            orientation="h",
            title="Top-20 LightGBM gain importance",
        )
        fig.update_layout(height=520, yaxis_title="", xaxis_title="Gain")
        out = fig
    else:
        out = "Feature importance not exposed by this model wrapper."
    out
    return (out,)


@app.cell
def _(mo):
    mo.md("## 3. Interactive what-if profile")
    return


@app.cell
def _(mo):
    # Personal & employment
    age = mo.ui.slider(start=18, stop=70, value=35, label="Age")
    income = mo.ui.slider(
        start=20_000, stop=500_000, value=150_000, step=5_000,
        label="Annual income (€)",
    )
    years_employed = mo.ui.slider(
        start=0.0, stop=40.0, value=5.0, step=0.5, label="Years at current employer",
    )

    # Loan parameters
    credit_amount = mo.ui.slider(
        start=50_000, stop=2_000_000, value=400_000, step=10_000,
        label="Loan amount (€)",
    )
    loan_annuity = mo.ui.slider(
        start=5_000, stop=200_000, value=22_000, step=500,
        label="Monthly payment (€)",
    )
    goods_price = mo.ui.slider(
        start=0, stop=2_000_000, value=400_000, step=10_000,
        label="Goods price (€, for purchase loans)",
    )

    # Household
    num_family = mo.ui.slider(start=1, stop=10, value=3, label="Family members")
    num_children = mo.ui.slider(start=0, stop=10, value=1, label="Children")

    mo.vstack([
        mo.md("**Personal & financial**"),
        mo.hstack([age, income, years_employed]),
        mo.md("**Loan**"),
        mo.hstack([credit_amount, loan_annuity, goods_price]),
        mo.md("**Household**"),
        mo.hstack([num_family, num_children]),
    ])
    return (
        age, credit_amount, goods_price, income, loan_annuity,
        num_children, num_family, years_employed,
    )


@app.cell
def _(
    age, credit_amount, goods_price, income, loan_annuity,
    num_children, num_family, predictor, years_employed,
):
    form = {
        "gender": "Male",
        "age_years": age.value,
        "num_children": num_children.value,
        "num_family_members": num_family.value,
        "family_status": "Married",
        "years_employed": years_employed.value,
        "organization_type": "Business Entity Type 3",
        "occupation_type": "Managers",
        "has_work_phone": True,
        "contract_type": "Cash loans",
        "credit_amount": credit_amount.value,
        "loan_annuity": loan_annuity.value,
        "goods_price": goods_price.value,
        "total_income": income.value,
        "owns_car": True,
        "car_age_years": 5,
        "owns_housing": True,
        "years_since_id_change": 4,
        "years_at_address": 7,
        "region_population_relative": 0.019,
        "city_rating": 2,
        "works_in_different_city": False,
        "has_landline": True,
    }
    result = predictor.predict(form)
    return form, result


@app.cell
def _(mo, result):
    tier_emoji = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}[result["risk_category"]]
    mo.md(
        f"""
        ## Live result

        | Metric | Value |
        |---|---|
        | Risk score | **{result['risk_score']} / 1000** |
        | Default probability | **{result['risk_probability']:.1%}** |
        | Tier | {tier_emoji} **{result['risk_category']}** |
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## 4. Insights for this profile")
    return


@app.cell
def _(form, insights, predictor):
    approval = insights.approval_with_confidence(predictor, form, n_bootstrap=20)
    rec_max = insights.recommended_max_loan(predictor, form, max_iter=15)
    counter_factuals = insights.counter_factual_recommendations(predictor, form)
    return approval, counter_factuals, rec_max


@app.cell
def _(approval, mo, rec_max):
    mo.md(
        f"""
        ### Approval probability + confidence

        - **Approval probability:** {approval['approval_probability']:.0%} (band: {approval['confidence_band']})
        - **CI:** {approval['ci_lower']:.1%} – {approval['ci_upper']:.1%}
        - **Approved at threshold 0.50:** {"✅" if approval['approved'] else "❌"}

        ### Recommended max loan
        - €{int(rec_max['amount']):,} for **{rec_max['projected_tier']}** tier
        - {rec_max.get('note', '')}
        """
    )
    return


@app.cell
def _(counter_factuals, mo):
    recs = counter_factuals.get("recommendations", [])
    if not recs:
        body = "_No clear single-change improvements found._"
    else:
        rows = "\n".join(
            f"- **{r['feature']}**: `{r['current_value']}` → `{r['suggested_value']}`"
            f" &nbsp;|&nbsp; score {r['current_score']} → {r['projected_score']}"
            f" &nbsp;(**Δ {r['delta']}**)"
            for r in recs
        )
        body = rows
    mo.md(f"### Counter-factual quick wins\n\n{body}")
    return


@app.cell
def _(mo):
    mo.md("## 5. Reproducibility recipe")
    return


@app.cell
def _(mo):
    mo.md(
        """
        From the project root:

        ```bash
        # 1. Feature selection (top 25 of 38 self-reportable candidates)
        .venv/bin/python scripts/select_top25_features.py

        # 2. Full squeeze pipeline (~5 min CPU)
        .venv/bin/python scripts/squeeze_top25_accuracy.py

        # 3. Precompute insight artefacts
        .venv/bin/python scripts/precompute_insights.py

        # The squeeze script writes scripts/results/best_top25_model.pkl,
        # which is then copied to src/assets/top25_risk_model.pkl by the build
        # step. This notebook loads the latter.
        ```

        Source of truth:

        - `scripts/results/squeeze_summary.json` — per-stage AUC + best params
        - `scripts/results/top25_features.json` — feature ranking + selection
        - `scripts/results/cohort_distributions.json` — for cohort percentile insight
        - `scripts/results/industry_region_benchmarks.json` — for industry/region context
        """
    )
    return


if __name__ == "__main__":
    app.run()
