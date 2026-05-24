"""Credit Default Risk Assessment — Streamlit entry point (Standard+ tier).

Single-flow UX:

  Landing  →  25-question form  →  Assessment result (tier badge + 6 tabs of insights)

The legacy 15-field flow has been removed. Behavioural traits is presented
as one of the result tabs, fed by a thin translator that adapts the Top-25
form keys to the behavioural predictor's expected schema.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import plotly.express as px
import streamlit as st

from models import insights
from models.top25_predictor import Top25Predictor
from src.components.behavioral_traits import BehavioralTraitsDisplay
from src.components.questionnaire_top25 import render_top25_questionnaire
from src.predictors.behavioral_predictor import (
    get_available_behavioral_models,
    predict_behavioral_traits,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "src/assets/top25_risk_model.pkl"
COHORT_PATH = "scripts/results/cohort_distributions.json"
BENCHMARK_PATH = "scripts/results/industry_region_benchmarks.json"

# ─────────────────────────────────────────────────────────────────────────────
# Styling — keep CSS lean; rely on Streamlit primitives where possible.
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* hero banner on the landing page */
.hero {
    padding: 2.5rem 2rem;
    background: linear-gradient(135deg, #4c6ef5 0%, #845ec2 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 6px 20px rgba(76,110,245,0.18);
}
.hero h1 { color: white; margin: 0; font-size: 2.2rem; font-weight: 700; }
.hero p  { color: rgba(255,255,255,0.92); margin: 0.6rem 0 0; font-size: 1.05rem; }
.hero ul { color: rgba(255,255,255,0.95); margin: 0.8rem 0 0; padding-left: 1.4rem; }
.hero li { margin: 0.15rem 0; }

/* tier badge displayed at the top of the result page */
.tier-card {
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.tier-card.low      { background: linear-gradient(135deg,#c8f4cd 0%,#a8e6b1 100%); color:#0a5d23; }
.tier-card.medium   { background: linear-gradient(135deg,#fff1c4 0%,#ffe39a 100%); color:#714d05; }
.tier-card.high     { background: linear-gradient(135deg,#f8d2d2 0%,#f0a8a8 100%); color:#7a1717; }
.tier-card .tier-emoji { font-size: 3rem; line-height: 1; }
.tier-card .tier-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
.tier-card .tier-sub   { font-size: 1rem; opacity: 0.85; margin: 0.25rem 0 0; }

/* small inline callout */
.callout {
    padding: 0.9rem 1.1rem;
    background: #f6f8fc;
    border-left: 4px solid #4c6ef5;
    border-radius: 6px;
    margin: 1rem 0;
    font-size: 0.95rem;
}
.callout strong { color: #2d3a7a; }

/* recommendation card */
.rec-card {
    padding: 0.9rem 1.1rem;
    background: #f3faf6;
    border-left: 4px solid #2ca02c;
    border-radius: 6px;
    margin: 0.6rem 0;
    font-size: 0.95rem;
}
.rec-card .delta { color: #1a7a1a; font-weight: 600; }

/* sidebar polish */
section[data-testid="stSidebar"] h2 { font-size: 1.05rem; }
section[data-testid="stSidebar"] .sidebar-version { color: #6c757d; font-size: 0.8rem; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading risk model…")
def _load_predictor() -> Top25Predictor:
    return Top25Predictor(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def _load_cohort_distributions() -> dict:
    """Load cohort distributions, rescaling stored /1000 quantiles to /100.

    The artefact was precomputed when risk_score lived on a 0–1000 scale.
    The predictor now emits 0–100 to match default-probability percent;
    quantiles are rescaled here so callers can compare scores directly.
    """
    raw = insights.load_precomputed(COHORT_PATH)
    if not raw:
        return raw
    cohorts = raw.get("cohorts", {})
    for cohort in cohorts.values():
        q = cohort.get("score_quantiles") or {}
        cohort["score_quantiles"] = {k: float(v) / 10.0 for k, v in q.items()}
    fb = raw.get("fallback", {}).get("score_quantiles") or {}
    if fb:
        raw["fallback"]["score_quantiles"] = {k: float(v) / 10.0 for k, v in fb.items()}
    return raw


@st.cache_resource(show_spinner=False)
def _load_industry_benchmarks() -> dict:
    return insights.load_precomputed(BENCHMARK_PATH)


@st.cache_resource(show_spinner=False)
def _load_behavioural() -> dict:
    return get_available_behavioral_models()


# ─────────────────────────────────────────────────────────────────────────────
# Behavioural-predictor adapter — translate Top-25 form keys → legacy schema.
# ─────────────────────────────────────────────────────────────────────────────
def _form_to_behavioural_input(form: dict[str, Any]) -> dict[str, Any]:
    """Translate the Top-25 form into the dict the behavioural predictor expects.

    The behavioural-traits model was trained on the legacy 15-field schema. We
    feed it the overlapping fields and let its internal defaults fill the rest.
    """
    out = {
        "age": form.get("age_years"),
        "total_income": form.get("total_income"),
        "years_employed": form.get("years_employed"),
        "family_status": form.get("family_status"),
        "num_children": form.get("num_children"),
        "owns_car": form.get("owns_car"),
        "owns_housing": form.get("owns_housing"),
        # Top-25 has organization_type, but the behavioural predictor's mapping
        # expects an "employment_status" coarse bucket. Pass through and let
        # the predictor's fallback handle unknowns.
        "employment_status": form.get("organization_type"),
    }
    return {k: v for k, v in out.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🏛️ Risk Assessment")
        st.markdown(
            '<p class="sidebar-version">v2.0 · Standard+ tier · 25-field model</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        if st.button(
            "📝 New assessment",
            use_container_width=True,
            type="primary",
            help="Clear inputs and start over.",
        ):
            for k in list(st.session_state.keys()):
                if k != "page":
                    del st.session_state[k]
            st.session_state.page = "assessment"
            st.rerun()

        st.divider()
        st.caption("Learn more")

        if st.button("🔬 Research & benchmarks", use_container_width=True):
            st.session_state.page = "research"
            st.rerun()
        if st.button("🗺️ Roadmap", use_container_width=True):
            st.session_state.page = "roadmap"
            st.rerun()

        st.divider()
        with st.expander("ℹ️ About this app"):
            st.markdown(
                "Powered by a tuned LightGBM model trained on 25 self-reportable "
                "features + 6 derived ratios. Holdout ROC-AUC **0.7146** (Kaggle "
                "median ≈ 0.75). "
                "Source: [GitHub](https://github.com/VytCepas/credit_default_risk_assessment)."
            )

        st.caption(f"🕒 {datetime.now().strftime('%H:%M, %a %d %b')}")


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────
def show_assessment_page() -> None:
    """Single-flow questionnaire + result page."""
    # Hero / intro shown above the form
    st.markdown(
        """
        <div class="hero">
            <h1>Credit Default Risk Assessment</h1>
            <p>Get your risk score in ~5 minutes. Powered by a tuned LightGBM model
            with explainable insights — counter-factual recommendations, cohort
            comparisons, and a loan-affordability sandbox.</p>
            <ul>
                <li>23 quick questions (2 fields are auto-filled)</li>
                <li>Result includes approval probability, decision time, and
                actionable improvement tips</li>
                <li>Your data stays in this browser session — nothing is stored</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    form_data = render_top25_questionnaire()
    if form_data is None:
        return

    # Persist + show results
    st.session_state["form_data"] = form_data
    _render_assessment_result(form_data)


def _render_assessment_result(form_data: dict[str, Any]) -> None:
    predictor = _load_predictor()

    with st.spinner("Computing your risk profile…"):
        result = predictor.predict(form_data)
        approval = insights.approval_with_confidence(predictor, form_data)
        decomposition = insights.risk_decomposition(predictor, form_data)
        cohort = insights.cohort_percentile(
            {**form_data, "__risk_score": result["risk_score"]},
            _load_cohort_distributions(),
        )
        bench = insights.industry_region_benchmark(form_data, _load_industry_benchmarks())
        process = insights.approval_process_time(result["risk_category"])
        rec_max = insights.recommended_max_loan(predictor, form_data)
        counter_factuals = insights.counter_factual_recommendations(predictor, form_data)
        time_to_improve = (
            insights.time_to_improvement(predictor, form_data)
            if result["risk_category"] != "Low"
            else {"already_at_target": True, "months_to_target": 0, "target_tier": "Low"}
        )

    tier = result["risk_category"]
    tier_class = {"Low": "low", "Medium": "medium", "High": "high"}[tier]
    tier_emoji = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}[tier]
    tier_blurb = {
        "Low": "Strong profile — likely approved on standard terms.",
        "Medium": "Borderline profile — may require manual review.",
        "High": "Notable default risk — see ‘Quick wins’ for improvements.",
    }[tier]

    st.markdown(
        f"""
        <div class="tier-card {tier_class}">
            <div class="tier-emoji">{tier_emoji}</div>
            <div>
                <p class="tier-title">{tier} risk</p>
                <p class="tier-sub">{tier_blurb}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gate = result.get("affordability_gate") or {}
    if gate.get("triggered"):
        model_pd = gate.get("model_probability", 0.0)
        final_pd = result["risk_probability"]
        floor = gate.get("affordability_pd_floor")
        floor_note = (
            f"  \n_Default probability raised from the model's_ "
            f"**{model_pd:.1%}** _to_ **{final_pd:.1%}** _via a deterministic "
            f"affordability stress score (floor: {floor:.1%})._"
            if floor is not None and floor > model_pd
            else ""
        )
        if gate.get("overrode_model"):
            st.error(
                "**Affordability gate triggered — tier overridden to High.**  \n"
                f"Model alone would have rated this **{gate.get('model_tier', '?')}** "
                f"(model PD {model_pd:.1%}), but a deterministic affordability "
                "check flagged the loan request as unservicable on the declared "
                "income.  \n  \n"
                f"_Reason:_ {gate.get('reason', '')}"
                f"{floor_note}"
            )
        else:
            st.warning(
                "**Affordability gate also triggered — both signals agree on High.**  \n"
                f"The model rated this **High** (model PD {model_pd:.1%}), and an "
                "independent deterministic affordability check confirms the loan "
                "request is unservicable on the declared income.  \n  \n"
                f"_Reason:_ {gate.get('reason', '')}"
                f"{floor_note}"
            )

    # Headline metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Risk score",
        f"{result['risk_score']} / 100",
        help="Default probability rescaled to a 0–100 scale. Lower is better.",
    )
    col2.metric(
        "Approval probability",
        f"{approval['approval_probability']:.0%}",
        delta=f"± {(approval['ci_upper'] - approval['ci_lower']) / 2:.1%}",
        delta_color="off",
        help=(
            "1 − default probability. Strictly: the model's estimated "
            f"chance you would repay. Confidence band: {approval['confidence_band']}."
        ),
    )
    col3.metric(
        "Default probability",
        f"{result['risk_probability']:.1%}",
        help="The LightGBM model's raw output: P(default within 12 months).",
    )
    col4.metric(
        "Decision time",
        process["expected_time"],
        help=(
            "Tier-based service-level expectation. Low → instant auto-decision; "
            "Medium → 1–2 days for a quick human review; "
            "High → 5+ days for full manual underwriting."
        ),
    )

    _render_metric_explanations(result, approval, process, gate)

    st.markdown(
        f'<div class="callout">Model: <strong>{result["model_name"]}</strong> '
        f"&middot; Holdout ROC-AUC <strong>{result['model_auc']:.4f}</strong> "
        f"&middot; Trained on 25 self-reportable features + 6 derived ratios.</div>",
        unsafe_allow_html=True,
    )

    # Insights tabs
    tab_wins, tab_loan, tab_break, tab_compare, tab_improve, tab_traits = st.tabs([
        "💡 Quick wins",
        "💰 Loan amount?",
        "📊 Risk breakdown",
        "🏘️ How do I compare?",
        "⏱️ Time to improve",
        "🎭 Behavioural traits",
    ])

    with tab_wins:
        _render_quick_wins(counter_factuals)

    with tab_loan:
        _render_loan_sandbox(predictor, form_data, result, rec_max)

    with tab_break:
        _render_risk_breakdown(decomposition)

    with tab_compare:
        _render_cohort_and_benchmark(cohort, bench)

    with tab_improve:
        _render_time_to_improve(time_to_improve)

    with tab_traits:
        _render_behavioural_traits(form_data)

    st.divider()
    if st.button("🔄 Start a new assessment", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Metric explanations
# ─────────────────────────────────────────────────────────────────────────────
def _render_metric_explanations(
    result: dict,
    approval: dict,
    process: dict,
    gate: dict,
) -> None:
    """Explain why each headline number reads the way it does for this applicant."""
    pd_pct = result["risk_probability"] * 100
    score = result["risk_score"]
    approval_pct = approval["approval_probability"] * 100
    tier = result["risk_category"]
    decision = process["expected_time"]

    tier_thresholds = "PD < 6% → Low · 6–15% → Medium · ≥ 15% → High"

    decision_reason = {
        "Low": (
            "**Low risk → instant auto-decision.** The model's confidence is "
            "high enough that no human review is needed for the standard "
            "questionnaire fields."
        ),
        "Medium": (
            "**Medium risk → 1–2 business days.** A loan officer briefly "
            "reviews the application: usually a quick income / employment "
            "verification, then either approves on standard terms or escalates."
        ),
        "High": (
            "**High risk → 5+ business days (manual review).** Full manual "
            "underwriting is required: documentation checks, possibly a credit "
            "bureau pull, and a final committee decision."
        ),
    }.get(tier, "")

    if gate.get("triggered"):
        if gate.get("overrode_model"):
            decision_reason += (
                "  \nThe affordability gate triggered and **overrode** the model — "
                f"it alone would have rated this {gate.get('model_tier', '?')} — "
                "forcing manual review on affordability grounds."
            )
        else:
            decision_reason += (
                "  \nThe affordability gate also triggered, **independently agreeing** "
                "with the model's High verdict — manual review on affordability grounds."
            )

    with st.expander("📖 What do these numbers mean?"):
        st.markdown(
            f"""
**Risk score — {score} / 100**

Your default probability rescaled into a 0–100 scale ({pd_pct:.1f}% × 100 = {score}).
Lower is better. The model's predictions for real applicants almost never go
above ~50 because true default risk caps out around there in the training data.

**Default probability — {pd_pct:.1f}%**

The estimated chance this application would result in a default within
roughly the first 12 months of repayment. This is the single number every
other metric is derived from. In normal cases this is the LightGBM model's
raw output; when the affordability gate fires, the model's PD is replaced
by a deterministic severity-based floor so the number reflects how badly
the loan exceeds affordability ceilings (not just whether it does).

**Approval probability — {approval_pct:.0f}%**

The complement of default probability: `100% − {pd_pct:.1f}% = {approval_pct:.0f}%`.
It is **not** an independent estimate of approval — it's the model's estimated
chance you'd repay, framed positively. The ± band comes from a quick
sensitivity check: we re-score the application with small random perturbations
of the numeric inputs and report the 5th–95th percentile of the resulting
probabilities.

**Decision time — {decision}**

{decision_reason}

Tier thresholds were anchored to the 50th and 90th percentiles of the model's
distribution on the 307k-row training set ({tier_thresholds}).

---

ℹ️ **The three top numbers are three views of one underlying probability.**
They will always satisfy:

- `risk_score = round(default_probability × 100)`
- `approval_probability = 1 − default_probability`

So a change in one is automatically reflected in the other two; the bootstrap
band on approval probability is the only piece of new information.
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# Insight tab renderers
# ─────────────────────────────────────────────────────────────────────────────
def _render_quick_wins(counter_factuals: dict) -> None:
    st.markdown("##### Three realistic changes that would lower your risk score")
    recs = counter_factuals.get("recommendations", [])
    if not recs:
        st.info(
            "No clear single-change improvements found for your profile — your "
            "inputs are already balanced for the loan parameters you provided."
        )
        return
    for rec in recs:
        st.markdown(
            f"""
            <div class="rec-card">
                <strong>{rec['feature']}</strong>:
                <code>{rec['current_value']}</code> → <code>{rec['suggested_value']}</code>
                &nbsp;|&nbsp; score
                {rec['current_score']} → {rec['projected_score']}
                &nbsp; <span class="delta">Δ {rec['delta']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Counter-factual perturbations are limited to mutable features (no demographic "
        "or family-status changes are suggested)."
    )


def _render_loan_sandbox(
    predictor: Top25Predictor,
    form_data: dict,
    result: dict,
    rec_max: dict,
) -> None:
    if rec_max.get("amount") is not None:
        st.success(
            f"💰 Recommended max for **{rec_max['projected_tier']}** tier: "
            f"**€{int(rec_max['amount']):,}** "
            f"(projected score {rec_max['projected_score']})"
        )
        st.caption(rec_max.get("note", ""))
    else:
        st.warning(rec_max.get("note", "No safe loan amount found."))

    st.markdown("##### What if I borrowed differently?")
    slider_default = int(form_data.get("credit_amount") or 500_000)
    new_amount = st.slider(
        "Loan amount (€)",
        min_value=50_000,
        max_value=max(2_000_000, slider_default * 2),
        value=slider_default,
        step=10_000,
        key="loan_slider",
    )
    if new_amount != slider_default:
        with st.spinner("Re-scoring…"):
            scenario = dict(form_data)
            scenario["credit_amount"] = float(new_amount)
            scenario["loan_annuity"] = (
                float(form_data.get("loan_annuity") or 25_000)
                * float(new_amount)
                / slider_default
            )
            new_result = predictor.predict(scenario)
        col_a, col_b = st.columns(2)
        col_a.metric(
            f"At €{new_amount:,}",
            f"{new_result['risk_score']} / 1000",
            delta=new_result["risk_score"] - result["risk_score"],
            delta_color="inverse",
        )
        col_b.metric(
            "New tier",
            new_result["risk_category"],
            help=f"{new_result['risk_probability']:.1%} default probability",
        )


def _render_risk_breakdown(decomposition: dict) -> None:
    """SHAP per-applicant attribution: top features that pushed the prediction.

    Positive SHAP (red) pushes the log-odds toward default; negative (green)
    pulls them toward repaid. Bars are sorted by magnitude.
    """
    features = decomposition.get("features", [])
    if not features:
        st.info(decomposition.get("note", "Risk decomposition unavailable."))
        return

    top_n = 12
    rows = features[:top_n]
    rows = list(reversed(rows))  # plotly horizontal bar reads bottom-up
    labels = [f"{r['feature']} = {_pretty_value(r['value'])}" for r in rows]
    values = [r["shap"] for r in rows]
    colors = ["#d6586e" if v > 0 else "#3aa17e" for v in values]

    fig = px.bar(
        x=values,
        y=labels,
        orientation="h",
        color=colors,
        color_discrete_map="identity",
    )
    fig.update_layout(
        height=460,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        xaxis_title="SHAP contribution (log-odds; → default, ← repaid)",
        yaxis_title="",
    )
    fig.add_vline(x=0, line_width=1, line_color="#999")
    st.plotly_chart(fig, use_container_width=True)

    base = decomposition.get("base_value")
    base_txt = f" Base log-odds: **{base:+.3f}**." if isinstance(base, (int, float)) else ""
    st.caption(
        f"{decomposition.get('method', '')} — each bar is this applicant's "
        f"SHAP value for that feature, in log-odds units. Red increases default "
        f"risk; green decreases it.{base_txt}"
    )


def _pretty_value(v: Any) -> str:
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        if abs(v) >= 1_000:
            return f"{v:,.0f}"
        return f"{v:.3g}"
    return str(v)


def _render_cohort_and_benchmark(cohort: dict, bench: dict) -> None:
    if cohort.get("percentile") is not None:
        st.success(cohort["interpretation"])
        st.caption(
            f"Cohort: **{cohort['cohort_label']}** "
            f"(n = {cohort['n_in_cohort']:,} applicants)"
        )
    else:
        st.info("Cohort distributions are not yet precomputed for your profile.")

    if bench.get("industry_rate") is not None:
        st.markdown("##### Industry & region context")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Your industry's default rate",
            f"{bench['industry_rate']:.1%}",
            help=f"Industry: {bench['industry_label']}",
        )
        col2.metric(
            "Your region's default rate",
            f"{bench['region_rate']:.1%}",
            help=f"Region rating: {bench['region_label']}",
        )
        col3.metric(
            "All-applicants baseline",
            f"{bench.get('population_rate', 0.081):.1%}",
        )
        st.caption(
            "Context only — observed defaults in the training data, not a "
            "personal judgment."
        )


def _render_time_to_improve(time_to_improve: dict) -> None:
    if time_to_improve.get("already_at_target"):
        st.success("🎉 You're already at Low-risk tier — no projection needed.")
        return
    months = time_to_improve.get("months_to_target")
    if months is None:
        st.warning(time_to_improve.get("note", "Target tier not reachable."))
        return
    st.info(
        f"At your current trajectory, you'd reach "
        f"**{time_to_improve['target_tier']}** tier in approximately "
        f"**{months} month{'s' if months != 1 else ''}** "
        f"(projected score {time_to_improve.get('projected_score', '?')})."
    )
    st.caption(time_to_improve.get("caveat", ""))


def _render_behavioural_traits(form_data: dict) -> None:
    behavioural_models = _load_behavioural()
    if not behavioural_models:
        st.info("Behavioural-traits model is not loaded.")
        return
    with st.spinner("Analysing behavioural traits…"):
        traits = predict_behavioral_traits(_form_to_behavioural_input(form_data))
    if "error" in traits:
        st.error(traits["error"])
        return
    BehavioralTraitsDisplay().display_behavioral_traits(traits)


# ─────────────────────────────────────────────────────────────────────────────
# Research & Benchmarks page
# ─────────────────────────────────────────────────────────────────────────────
def show_research_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🔬 Research & Benchmarks</h1>
            <p>How our model compares to the public Kaggle leaderboard and what
            the experiments showed.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    import pandas as pd

    benchmarks = pd.DataFrame(
        [
            ("1st place (Home Aloan, 2018)", 0.806, "Kaggle"),
            ("Top 1 % (~43 teams)", 0.801, "Kaggle"),
            ("Bronze / top 10 %", 0.794, "Kaggle"),
            ("Aguiar public kernel", 0.791, "Kaggle"),
            ("Median submission", 0.75, "Kaggle"),
            ("Application-only LR baseline", 0.70, "Kaggle"),
            ("Production model (Standard+ tier, tuned LightGBM)", 0.7146, "Ours"),
            ("Top-25 + ratios, defaults", 0.7093, "Ours"),
            ("Stacking + calibration", 0.7142, "Ours"),
            ("CTGAN-balanced LightGBM", 0.6882, "Ours"),
            ("E2a — 12-feature baseline", 0.6846, "Ours"),
        ],
        columns=["Model", "ROC-AUC", "Source"],
    )

    fig = px.bar(
        benchmarks.sort_values("ROC-AUC"),
        x="ROC-AUC", y="Model", color="Source",
        orientation="h", range_x=[0.55, 0.85],
        color_discrete_map={"Kaggle": "#b0b0b0", "Ours": "#4c6ef5"},
        title="Our models vs Kaggle Home Credit Default Risk leaderboard",
    )
    fig.update_layout(height=520, yaxis_title="", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        ### Why we're at ~0.71 and not 0.80

        The Kaggle competition winners used the full Home Credit dataset
        (`bureau.csv`, `previous_application.csv`, `installments_payments.csv`,
        …) — six supplementary tables we don't yet have access to.
        `EXT_SOURCE_1/2/3` — external bureau credit scores — alone account for
        ~0.04–0.06 AUC of the gap. We deliberately exclude them because
        applicants cannot answer them themselves.

        Our path to Kaggle-median (~0.75) without growing the form runs through
        the **consented bureau-pull integration** (ADR 0001 §Future, issue #72).
        """
    )

    st.markdown("### Top-solution recipe (status against our Standard+ model)")
    recipe = pd.DataFrame(
        [
            ("Bureau & previous_application aggregations", "+0.04 to +0.06", "🗓 Pending (#47, #72)"),
            ("EXT_SOURCE interactions", "+0.005", "🗓 Pending bureau-pull"),
            ("Application-level ratios (DTI etc.)", "+0.005 to +0.015", "✅ Shipped"),
            ("Installments_payments aggregations", "+0.01 to +0.02", "🗓 Pending"),
            ("LightGBM hyperparameter tuning", "+0.005 to +0.010", "✅ Shipped"),
            ("CTGAN minority-class balancing", "varies", "✅ Evaluated (neutral)"),
            ("Stacking (GBM + LGBM + XGB)", "+0.003 to +0.008", "✅ Evaluated (neutral)"),
            ("Probability calibration (Platt)", "Brier ↓", "✅ Evaluated (neutral)"),
            ("Denoising autoencoder embeddings", "+0.005 to +0.01", "🗓 Top-1% only, deferred"),
        ],
        columns=["Technique", "Expected AUC lift", "Status"],
    )
    st.dataframe(recipe, use_container_width=True, hide_index=True)

    st.caption(
        "References: [Aguiar public kernel](https://github.com/js-aguiar/home-credit-default-competition) · "
        "[Onodera 2nd place](https://github.com/KazukiOnodera/Home-Credit-Default-Risk) · "
        "[CTGAN paper](https://arxiv.org/abs/1907.00503) · "
        "[deepsense.ai blog](https://deepsense.ai/blog/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Roadmap page
# ─────────────────────────────────────────────────────────────────────────────
def show_roadmap_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🗺️ Roadmap</h1>
            <p>What's shipped, what's next, what's strategic.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ✅ Shipped (Practical 3)")
    st.markdown(
        """
        - **Standard+ 25-field tier** — current page (PR #74)
        - **Insights catalogue** — counter-factuals, cohort comparison,
          loan-affordability sandbox, risk decomposition, time-to-improve
          (PR #85)
        - **Tiered questionnaire ADR** (ADR 0001) — Quick / Standard+ / Extended +
          derived-features layer + bureau-pull future track
        - **Marimo notebook port** (Epic 9 partial)
        """
    )

    st.markdown("### 🔄 Sprint 4 — in flight")
    st.markdown(
        """
        - **#47 Bureau-table aggregations** — biggest single AUC lift available
        - **#51 Fairness audit** — demographic parity & equalised odds per tier
        - **LZ-10 marimo migration** — CI integration + decommission `.ipynb`
        """
    )

    st.markdown("### 🗓 Strategic")
    st.markdown(
        """
        - **#72 Consented bureau-pull integration** — adds `EXT_SOURCE_*` to the
          derived layer without growing the form. Largest known accuracy lever
          (~+0.05 AUC).
        - **#84 P-10 Time-to-default model** (survival analysis) — gated on
          bureau-balance data
        - **#71 Extended tier (~100 fields)** — loan-officer mode for high-value
          applicants
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
def _render_footer() -> None:
    st.markdown(
        """
        <hr>
        <div style="text-align:center; color:#6c757d; font-size:0.85rem; padding: 0.6rem 0;">
            🏛️ <strong>Credit Default Risk Assessment</strong> ·
            Standard+ tier (Top-25, tuned LightGBM) ·
            ROC-AUC 0.7146 ·
            Educational / demonstration use only.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _render_sidebar()

    if "page" not in st.session_state:
        st.session_state.page = "assessment"

    page = st.session_state.page
    try:
        if page == "research":
            show_research_page()
        elif page == "roadmap":
            show_roadmap_page()
        else:
            show_assessment_page()
    except FileNotFoundError as e:
        st.error(f"Required asset missing: {e}")
        st.info(
            "Train the Standard+ model by running "
            "`.venv/bin/python scripts/squeeze_top25_accuracy.py` and "
            "`.venv/bin/python scripts/precompute_insights.py`."
        )
    except Exception as e:  # noqa: BLE001 — top-level UI safety net
        logger.exception("Unhandled error in main()")
        st.error(f"Unexpected error: {e}")
        if st.button("🔄 Reload"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    _render_footer()


if __name__ == "__main__":
    main()
