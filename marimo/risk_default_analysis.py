"""Marimo reactive notebook — Risk default analysis.

Reactive port of `notebooks/risk_default_analysis.ipynb`. Same dataset, same
split, plus the Practical 3 model-expansion experiments (E1, E2a, E4, E5) and
a per-stage view of the Top-25 squeeze summary.

Run with:
    .venv/bin/marimo edit marimo/risk_default_analysis.py
Read-only:
    .venv/bin/marimo run  marimo/risk_default_analysis.py
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
        # Home Credit Default Risk — Marimo Edition

        Reactive port of `notebooks/risk_default_analysis.ipynb`.

        **Kaggle leaderboard reference** (Home Credit Default Risk, 7,198 teams, 2018):

        | Tier | ROC-AUC |
        | --- | --- |
        | 1st place ("Home Aloan") | ~0.806 |
        | Top 1 % | ~0.801 |
        | Bronze / top 10 % | ~0.794 |
        | Aguiar public kernel | ~0.791 |
        | Median submission | ~0.75 |
        | Application-only LR baseline | ~0.70 |
        | **Our production Top-25 (Standard+)** | **0.7146** |
        | Production 15-field GBM (Practical 2) | 0.6272 |
        """
    )
    return


@app.cell
def _():
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "marimo" else Path.cwd()
    sys.path.insert(0, str(PROJECT_ROOT))

    DATA_DIRECTORY = PROJECT_ROOT / "data"
    RESULTS_DIRECTORY = PROJECT_ROOT / "scripts" / "results"
    RANDOM_STATE = 0
    return (
        DATA_DIRECTORY,
        PROJECT_ROOT,
        RESULTS_DIRECTORY,
        RANDOM_STATE,
        np,
        pd,
        roc_auc_score,
        time,
        train_test_split,
    )


@app.cell
def _(DATA_DIRECTORY, mo, pd):
    application_df = pd.read_parquet(DATA_DIRECTORY / "application_train.parquet")
    application_df.columns = application_df.columns.str.lower()
    mo.md(
        f"""
        **Dataset loaded:** `application_train.parquet`

        - Rows: {application_df.shape[0]:,}
        - Columns: {application_df.shape[1]}
        - Default rate: {application_df['target'].mean():.1%}
        - EXT_SOURCE_1/2/3 present: `{all(f in application_df.columns for f in ['ext_source_1','ext_source_2','ext_source_3'])}`
        """
    )
    return (application_df,)


@app.cell
def _(application_df, np):
    eps = 1e-9
    app_eng = application_df.copy()
    app_eng["days_employed"] = app_eng["days_employed"].replace(365243, np.nan)

    # Engineered ratio features (Kaggle top-solution playbook)
    app_eng["dti"] = app_eng["amt_annuity"] / (app_eng["amt_income_total"] + eps)
    app_eng["credit_to_income"] = app_eng["amt_credit"] / (app_eng["amt_income_total"] + eps)
    app_eng["annuity_to_credit"] = app_eng["amt_annuity"] / (app_eng["amt_credit"] + eps)
    app_eng["years_employed_ratio"] = (-app_eng["days_employed"]) / (
        (-app_eng["days_birth"]) + eps
    )
    app_eng["income_per_family_member"] = app_eng["amt_income_total"] / (
        app_eng["cnt_fam_members"] + eps
    )
    # EXT_SOURCE interaction (unconstrained set only)
    app_eng["ext_2_x_3"] = app_eng["ext_source_2"] * app_eng["ext_source_3"]
    app_eng["ext_source_mean"] = app_eng[
        ["ext_source_1", "ext_source_2", "ext_source_3"]
    ].mean(axis=1)

    return (app_eng,)


@app.cell
def _(mo):
    mo.md(
        """
        ## E1 — Unconstrained baseline (LightGBM on all numeric features)

        Reveals the AUC ceiling reachable with `application_train` alone. The
        questionnaire constraint excludes `EXT_SOURCE_*`; here we **include**
        them to quantify the cost of the constraint.
        """
    )
    return


@app.cell
def _(RANDOM_STATE, app_eng, np, roc_auc_score, time, train_test_split):
    from lightgbm import LGBMClassifier

    numeric_cols = (
        app_eng.select_dtypes(include=[np.number])
        .columns.drop(["sk_id_curr", "target"])
        .tolist()
    )

    X_uc = app_eng[numeric_cols].copy().fillna(
        app_eng[numeric_cols].median(numeric_only=True)
    )
    y_full = app_eng["target"].astype(int)

    X_tr_uc, X_te_uc, y_tr_uc, y_te_uc = train_test_split(
        X_uc, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full
    )

    lgbm_e1 = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1,
    )

    t0_e1 = time.perf_counter()
    lgbm_e1.fit(X_tr_uc, y_tr_uc)
    e1_time = round(time.perf_counter() - t0_e1, 1)
    e1_auc = round(roc_auc_score(y_te_uc, lgbm_e1.predict_proba(X_te_uc)[:, 1]), 4)

    e1_result = {
        "experiment": "E1 — unconstrained baseline",
        "features": len(numeric_cols),
        "train_time_s": e1_time,
        "roc_auc": e1_auc,
    }
    e1_result
    return LGBMClassifier, e1_result, y_full


@app.cell
def _(mo):
    mo.md(
        """
        ## E2a — Engineered ratios on the questionnaire surface

        Five ratio features derived from columns the production app already
        collects. Zero-data-cost feature engineering: the application form does
        not need to change.
        """
    )
    return


@app.cell
def _(
    LGBMClassifier,
    RANDOM_STATE,
    app_eng,
    roc_auc_score,
    time,
    train_test_split,
    y_full,
):
    questionnaire_numeric = [
        "cnt_children", "amt_income_total", "amt_credit", "amt_annuity",
        "cnt_fam_members", "days_birth", "days_employed",
    ]
    ratio_features = [
        "dti", "credit_to_income", "annuity_to_credit",
        "years_employed_ratio", "income_per_family_member",
    ]

    X_e2a = app_eng[questionnaire_numeric + ratio_features].copy()
    X_e2a = X_e2a.fillna(X_e2a.median(numeric_only=True))

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_e2a, y_full, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_full,
    )
    lgbm_e2a = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1,
    )
    t0_e2a = time.perf_counter()
    lgbm_e2a.fit(X_tr2, y_tr2)
    e2a_time = round(time.perf_counter() - t0_e2a, 1)
    e2a_auc = round(roc_auc_score(y_te2, lgbm_e2a.predict_proba(X_te2)[:, 1]), 4)
    e2a_result = {
        "experiment": "E2a — Questionnaire + 5 ratios",
        "features": X_e2a.shape[1],
        "train_time_s": e2a_time,
        "roc_auc": e2a_auc,
    }
    e2a_result
    return X_te2, X_tr2, e2a_result, y_te2, y_tr2


@app.cell
def _(mo):
    mo.md(
        """
        ## E4 — CTGAN minority-class synthetic balancing

        Replaces `SMOTETomek` (linear interpolation + Tomek cleaning) with
        **CTGAN** (Conditional Tabular GAN; Xu et al. NeurIPS 2019). CTGAN
        models the joint distribution of all columns, handling discrete
        columns natively via conditional generation.

        Reference: <https://arxiv.org/abs/1907.00503>
        """
    )
    return


@app.cell
def _(
    LGBMClassifier,
    RANDOM_STATE,
    X_te2,
    X_tr2,
    roc_auc_score,
    time,
    y_te2,
    y_tr2,
):
    try:
        from ctgan import CTGAN
        import pandas as _pd
        import numpy as _np

        minority_train = X_tr2[y_tr2 == 1].copy()
        minority_count_target = int((y_tr2 == 0).sum())
        minority_sample = minority_train.sample(
            n=min(5000, len(minority_train)),
            random_state=RANDOM_STATE,
        )

        ctgan_t0 = time.perf_counter()
        ctgan = CTGAN(epochs=50, verbose=False, cuda=False)
        ctgan.fit(minority_sample.reset_index(drop=True))
        n_to_generate = minority_count_target - int((y_tr2 == 1).sum())
        synth = ctgan.sample(n_to_generate)
        ctgan_time = round(time.perf_counter() - ctgan_t0, 1)

        X_balanced = _pd.concat(
            [X_tr2, synth.reset_index(drop=True)], ignore_index=True
        )
        y_balanced = _np.concatenate(
            [y_tr2.values, _np.ones(len(synth), dtype=int)]
        )

        lgbm_e4 = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1,
        )
        lgbm_e4.fit(X_balanced, y_balanced)
        e4_auc = round(roc_auc_score(y_te2, lgbm_e4.predict_proba(X_te2)[:, 1]), 4)
        e4_result = {
            "experiment": "E4 — CTGAN-balanced LightGBM",
            "ctgan_train_time_s": ctgan_time,
            "synthetic_minority_added": int(n_to_generate),
            "roc_auc": e4_auc,
        }
    except ImportError:
        e4_result = {
            "experiment": "E4 — CTGAN-balanced LightGBM",
            "status": "ctgan not installed; run `pip install ctgan` and re-execute",
        }
    e4_result
    return (e4_result,)


@app.cell
def _(mo):
    mo.md(
        """
        ## E5 — Stacking ensemble + probability calibration

        Stack LightGBM + GBM (+ optional XGBoost) with a Logistic Regression
        meta-learner, then calibrate the meta-learner's probabilities with
        `CalibratedClassifierCV`. Calibration does not improve ROC-AUC (it is
        a monotone transform) but lowers Brier score — critical when the
        score is shown to loan officers as a percentage.
        """
    )
    return


@app.cell
def _(LGBMClassifier, RANDOM_STATE, X_te2, X_tr2, roc_auc_score, y_te2, y_tr2):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    base_estimators = [
        (
            "gbm",
            GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=RANDOM_STATE,
            ),
        ),
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbose=-1,
            ),
        ),
    ]
    try:
        from xgboost import XGBClassifier
        base_estimators.append(
            (
                "xgb",
                XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    eval_metric="auc", verbosity=0,
                ),
            ),
        )
    except ImportError:
        pass

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=3, n_jobs=-1, passthrough=False,
    )
    stack.fit(X_tr2, y_tr2)
    stack_proba = stack.predict_proba(X_te2)[:, 1]
    e5_auc = round(roc_auc_score(y_te2, stack_proba), 4)

    calibrator = CalibratedClassifierCV(
        FrozenEstimator(stack), method="sigmoid", cv=5,
    )
    calibrator.fit(X_tr2, y_tr2)
    cal_proba = calibrator.predict_proba(X_te2)[:, 1]
    e5_brier = round(brier_score_loss(y_te2, cal_proba), 4)
    e5_brier_uncal = round(brier_score_loss(y_te2, stack_proba), 4)

    e5_result = {
        "experiment": "E5 — Stacking + calibration",
        "stack_roc_auc": e5_auc,
        "brier_uncalibrated": e5_brier_uncal,
        "brier_calibrated": e5_brier,
        "base_models": [name for name, _ in base_estimators],
    }
    e5_result
    return (e5_result,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Top-25 squeeze pipeline — summary

        Loaded from `scripts/results/squeeze_summary.json`. The squeeze
        pipeline lives in `scripts/squeeze_top25_accuracy.py` and produces
        the production Standard+ model at `src/assets/top25_risk_model.pkl`.
        """
    )
    return


@app.cell
def _(RESULTS_DIRECTORY, mo, pd):
    import json as _json
    with open(RESULTS_DIRECTORY / "squeeze_summary.json") as _f:
        squeeze = _json.load(_f)

    stage_rows = [
        ("Production GBM (15 fields)",   squeeze["stages"]["production_reference"]["auc"]),
        ("S1 — Top-25 only",             squeeze["stages"]["stage1_top25_only"]["auc"]),
        ("S1+ratios — Top-25 + 6 ratios", squeeze["stages"]["S1_top25_plus_ratios"]["auc"]),
        ("S2 — RandomizedSearchCV",      squeeze["stages"]["S2_randomized_search"]["auc"]),
        ("S3 — CTGAN-balanced",          squeeze["stages"]["S3_ctgan_balanced"]["auc"]),
        ("S4 — Stacking + calibration",  squeeze["stages"]["S4_stacking"]["stack_auc"]),
    ]
    stage_df = pd.DataFrame(stage_rows, columns=["Stage", "ROC-AUC"])
    stage_df["Δ vs production"] = (
        stage_df["ROC-AUC"] - stage_df["ROC-AUC"].iloc[0]
    ).round(4)
    mo.ui.table(stage_df)
    return squeeze, stage_df


@app.cell
def _(e1_result, e2a_result, e4_result, e5_result, mo, pd):
    rows = []
    for r in (e1_result, e2a_result, e4_result, e5_result):
        rows.append(
            {
                "Experiment": r.get("experiment", "?"),
                "ROC-AUC": r.get("roc_auc")
                or r.get("stack_roc_auc")
                or "—",
                "Notes": (
                    r.get("status")
                    or f"features={r.get('features','?')}, time={r.get('train_time_s','?')}s"
                ),
            }
        )
    summary = pd.DataFrame(rows)
    mo.ui.table(summary)
    return (summary,)


if __name__ == "__main__":
    app.run()
