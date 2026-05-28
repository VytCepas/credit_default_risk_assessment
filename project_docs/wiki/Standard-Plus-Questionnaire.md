# Standard+ Questionnaire (25-field tier)

The production app collects **22 user-typed answers** and **auto-fills 2** of
the model's 25 features. One UI dropdown (`city_size`) maps to **two** model
features via training-data medians, so the user only sees 22 questions.

## Field map

| Section | UI field | Model feature(s) | Notes |
|---|---|---|---|
| **Personal** (5) | Gender | `gender` | Male / Female |
|  | Age | `age` | Years; converted to `days_birth` server-side |
|  | Family status | `family_status` | Single / Married / Civil marriage / Separated / Widow |
|  | Number of children | `num_children` | Integer |
|  | Family members in household | `num_family_members` | Integer |
| **Employment** (4) | Years employed | `years_employed` | Float; `days_employed` server-side |
|  | Organisation type | `organization_type` | 58 categories from Kaggle taxonomy |
|  | Occupation type | `occupation_type` | 18 categories |
|  | Has work phone | `has_work_phone` | Boolean |
| **Loan** (4) | Contract type | `contract_type` | Cash loans / Revolving loans |
|  | Credit amount | `credit_amount` | Currency unit |
|  | Loan annuity | `loan_annuity` | Annual repayment |
|  | Goods price | `goods_price` | Asset value the loan funds |
| **Financial** (1) | Total income | `total_income` | Net monthly income |
| **Assets** (3) | Owns car | `owns_car` | Boolean |
|  | Car age | `car_age` | Years; 0 if no car |
|  | Owns realty | `owns_realty` | Boolean |
| **Residence** (4) | Years at current address | `years_at_address` | Float |
|  | Years since ID change | `years_since_id_change` | Float |
|  | City size *(one UI field)* | `region_population_relative` **and** `region_rating_client_w_city` | Maps via training-data medians — saves one question |
|  | Works in a different city | `works_in_different_city` | Boolean |
| **Other** (1) | Has landline | `has_landline` | Boolean |
| **Auto-filled** | (hidden) | `hour_appr_process_start` | Server timestamp |
|  | (hidden) | `weekday_appr_process_start` | Server timestamp |

**Total user-visible questions: 22.** **Total model features: 25.**
(`city_size` covers 2 features; 2 are auto-filled from server time.)

## Why these 25?

Stage-1 of the squeeze pipeline (`scripts/select_top25_features.py`) ranks
all 38 candidate self-reportable columns from `application_train` by
LightGBM gain importance over a 5-fold stratified CV. The top 25 are kept;
the rest are dropped. The full ranking is saved to
`scripts/results/top25_features.json`.

## Why "self-reportable"?

A field is "self-reportable" if a non-expert user can answer it in a web form.
Hard examples that we **exclude**:

- `EXT_SOURCE_1/2/3` — external bureau credit scores. Predictive but unknown to the user.
- `AMT_REQ_CREDIT_BUREAU_*` — count of recent bureau inquiries. Requires a bureau pull.
- `DAYS_LAST_PHONE_CHANGE`, `OWN_CAR_AGE` for non-car-owners — fragile.
- Social-circle defaults — sourced from internal Home Credit data.

This constraint is what defines the [ADR 0001](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/adr/0001_tiered_questionnaire.md) tier strategy.

## Server-side derivations

The 6 ratios listed in [Modeling Pipeline](Modeling-Pipeline#feature-engineering)
are computed at inference inside `Top25Predictor` — invisible to the user
but counted as features by the model.

## See also

- [`src/components/questionnaire_top25.py`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/src/components/questionnaire_top25.py) — the Streamlit form
- [`models/top25_predictor.py`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/models/top25_predictor.py) — the inference wrapper
- [`scripts/results/top25_features.json`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/scripts/results/top25_features.json) — the Stage-1 ranking
