"""Streamlit form for the 25-field "Standard+" tier (ADR 0001, issue #68).

Collects 23 user-typed answers; the 2 timestamp-derived features
(``hour_appr_process_start``, ``weekday_appr_process_start``) are filled
in by :class:`Top25Predictor` at inference time.

The form is grouped into 6 sections to keep the cognitive load
manageable; expected completion time is ~5 minutes.
"""
from __future__ import annotations

import streamlit as st


# Mapping of city-size choice to (region_population_relative, region_rating_client_w_city).
# Bucket medians come from a groupby on the training data (see
# scripts/select_top25_features.py and the inline correlation analysis).
CITY_SIZE_OPTIONS: dict[str, tuple[float, int]] = {
    "Capital / Major city (e.g. Vilnius)":   (0.046, 1),
    "Mid-size city (e.g. Kaunas, Klaipėda)": (0.019, 2),
    "Small town / village":                  (0.013, 3),
}

ORGANIZATION_TYPES = [
    "Business Entity Type 3", "Business Entity Type 2", "Business Entity Type 1",
    "Self-employed", "Government", "School", "Trade: type 7", "Trade: type 3",
    "Trade: type 2", "Medicine", "Construction", "Kindergarten", "Industry: type 3",
    "Industry: type 9", "Industry: type 11", "Transport: type 4", "Bank",
    "Police", "Military", "Postal", "Agriculture", "Restaurant",
    "Services", "University", "Security", "Hotel", "Religion", "Realtor",
    "Cleaning", "Insurance", "Telecom", "Emergency", "Culture", "Electricity",
    "Mobile", "Legal Services", "Advertising", "Other",
]

OCCUPATION_TYPES = [
    "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
    "High skill tech staff", "Accountants", "Medicine staff", "Cooking staff",
    "Security staff", "Cleaning staff", "Private service staff", "Low-skill Laborers",
    "Waiters/barmen staff", "Secretaries", "Realty agents",
    "HR staff", "IT staff", "Other",
]

FAMILY_STATUSES = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
CONTRACT_TYPES = ["Cash loans", "Revolving loans"]
GENDERS = ["Female", "Male"]


def _yesno(label: str, key: str, default: bool = False, help: str | None = None) -> bool:
    return st.radio(
        label, options=["Yes", "No"], index=0 if default else 1,
        key=key, horizontal=True, help=help,
    ) == "Yes"


def render_top25_questionnaire() -> dict | None:
    """Render the Standard+ form. Returns the form dict on submit, else None."""
    st.markdown(
        '<div class="section-header">📋 Standard+ Application (25 questions, ~5 min)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-box">
        This is the <strong>Standard+</strong> tier (ADR 0001). It collects 23 questions
        you can answer yourself (the application time and weekday are filled in
        automatically). Expected accuracy: <strong>~0.69 ROC-AUC</strong> versus
        ~0.63 on the legacy 15-question form.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("top25_questionnaire", clear_on_submit=False):
        # ---- Personal ----
        st.markdown("### 👤 Personal")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", GENDERS, key="t25_gender")
        with col2:
            age = st.number_input("Age", min_value=18, max_value=80, value=30, key="t25_age")
        with col3:
            family_status = st.selectbox(
                "Family status", FAMILY_STATUSES, key="t25_family_status",
            )

        col1, col2 = st.columns(2)
        with col1:
            num_children = st.number_input(
                "Number of children", min_value=0, max_value=20, value=0, key="t25_children",
            )
        with col2:
            num_family = st.number_input(
                "Number of family members (including yourself)",
                min_value=1, max_value=20, value=1, key="t25_family",
            )

        # ---- Employment ----
        st.markdown("### 💼 Employment")
        col1, col2 = st.columns(2)
        with col1:
            years_employed = st.number_input(
                "Years at current employer",
                min_value=0.0, max_value=60.0, value=3.0, step=0.5,
                key="t25_years_employed",
            )
        with col2:
            has_work_phone = _yesno(
                "Do you have a work phone?", "t25_work_phone", default=False,
                help="A direct work number, not a personal mobile.",
            )
        col1, col2 = st.columns(2)
        with col1:
            organization_type = st.selectbox(
                "Employer industry", ORGANIZATION_TYPES,
                key="t25_organization_type",
                help="Industry / sector of your employer.",
            )
        with col2:
            occupation_type = st.selectbox(
                "Your role / occupation", OCCUPATION_TYPES,
                key="t25_occupation_type",
            )

        # ---- Loan ----
        st.markdown("### 💰 Loan")
        col1, col2 = st.columns(2)
        with col1:
            contract_type = st.selectbox(
                "Loan contract type", CONTRACT_TYPES, key="t25_contract_type",
            )
        with col2:
            total_income = st.number_input(
                "Annual income",
                min_value=0, max_value=10_000_000, value=150_000, step=1_000,
                key="t25_income",
            )
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_amount = st.number_input(
                "Loan amount requested",
                min_value=0, max_value=10_000_000, value=500_000, step=10_000,
                key="t25_credit",
            )
        with col2:
            loan_annuity = st.number_input(
                "Estimated monthly payment",
                min_value=0, max_value=200_000, value=25_000, step=500,
                key="t25_annuity",
            )
        with col3:
            goods_price = st.number_input(
                "Asset purchase price (for purchase loans; 0 if not applicable)",
                min_value=0, max_value=10_000_000, value=500_000, step=10_000,
                key="t25_goods_price",
            )

        # ---- Assets ----
        st.markdown("### 🏠 Assets")
        col1, col2, col3 = st.columns(3)
        with col1:
            owns_car = _yesno("Do you own a car?", "t25_owns_car", default=False)
        with col2:
            car_age = st.number_input(
                "Age of your car (years; 0 if none)",
                min_value=0, max_value=70, value=0, key="t25_car_age",
            )
        with col3:
            owns_realty = _yesno(
                "Do you own real estate?", "t25_owns_realty", default=False,
            )

        # ---- Residence ----
        st.markdown("### 📍 Residence")
        col1, col2 = st.columns(2)
        with col1:
            years_at_address = st.number_input(
                "Years at current address",
                min_value=0.0, max_value=80.0, value=5.0, step=0.5,
                key="t25_years_address",
            )
        with col2:
            years_id = st.number_input(
                "Years since your ID was last issued / renewed",
                min_value=0.0, max_value=20.0, value=3.0, step=0.5,
                key="t25_years_id",
                help="Lithuanian personal IDs are renewed every 10 years.",
            )

        col1, col2 = st.columns(2)
        with col1:
            city_size_label = st.selectbox(
                "Size of the city/region you live in",
                list(CITY_SIZE_OPTIONS.keys()), key="t25_city_size",
            )
        with col2:
            different_city = _yesno(
                "Do you work in a different city from where you live?",
                "t25_diff_city", default=False,
            )

        # ---- Other ----
        st.markdown("### 📞 Other")
        has_landline = _yesno(
            "Do you have a landline / home phone?", "t25_landline", default=False,
        )

        submitted = st.form_submit_button("🎯 Get Risk Assessment", type="primary")

    if not submitted:
        return None

    pop_rel, city_rating = CITY_SIZE_OPTIONS[city_size_label]

    return {
        # Personal
        "gender": gender,
        "age_years": age,
        "num_children": num_children,
        "num_family_members": num_family,
        "family_status": family_status,
        # Employment
        "years_employed": years_employed,
        "organization_type": organization_type,
        "occupation_type": occupation_type,
        "has_work_phone": has_work_phone,
        # Loan
        "contract_type": contract_type,
        "credit_amount": credit_amount,
        "loan_annuity": loan_annuity,
        "goods_price": goods_price,
        # Financial
        "total_income": total_income,
        # Assets
        "owns_car": owns_car,
        "car_age_years": car_age if owns_car else None,
        "owns_housing": owns_realty,
        # Residence
        "years_since_id_change": years_id,
        "years_at_address": years_at_address,
        "region_population_relative": pop_rel,
        "city_rating": city_rating,
        "works_in_different_city": different_city,
        # Other
        "has_landline": has_landline,
    }
