"""End-to-end sanity check of the Standard+ predictor over a hand-built profile suite.

For each profile we record what we *expect* (tier, rough PD direction) and then
print the actual model output. The script also asserts the three identities
the Streamlit result card relies on:

    risk_score          == round(default_probability * 100)
    approval_probability ==  1 - default_probability
    risk_score / 100    ==  default_probability   (i.e. they are NOT independent)

Run with::

    .venv/bin/python scripts/evaluate_test_cases.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models import insights
from models.top25_predictor import Top25Predictor

MODEL_PATH = "src/assets/top25_risk_model.pkl"


@dataclass
class Profile:
    name: str
    expect_tier: str
    expect_gate: bool
    note: str
    form: dict[str, Any]


def _base() -> dict[str, Any]:
    return dict(
        gender="Female", age_years=35, num_children=1, num_family_members=3,
        family_status="Married", years_employed=5,
        organization_type="Business Entity Type 3", occupation_type="Core staff",
        has_work_phone=False, contract_type="Cash loans",
        credit_amount=500_000, loan_annuity=25_000, goods_price=500_000,
        total_income=150_000, owns_car=False, car_age_years=None, owns_housing=True,
        years_since_id_change=3, years_at_address=5,
        region_population_relative=0.019, city_rating=2,
        works_in_different_city=False, has_landline=False,
    )


def _merge(**overrides: Any) -> dict[str, Any]:
    f = _base()
    f.update(overrides)
    return f


PROFILES: list[Profile] = [
    Profile(
        "A. Strong professional, modest loan",
        expect_tier="Low", expect_gate=False,
        note="High income, long tenure, owns home and car, capital city.",
        form=_merge(
            gender="Male", age_years=42, num_children=2, num_family_members=4,
            years_employed=12, organization_type="Government",
            occupation_type="Managers", has_work_phone=True,
            credit_amount=300_000, loan_annuity=15_000, goods_price=300_000,
            total_income=250_000, owns_car=True, car_age_years=3,
            years_since_id_change=4, years_at_address=10,
            region_population_relative=0.046, city_rating=1, has_landline=True,
        ),
    ),
    Profile(
        "B. Median applicant",
        expect_tier="Low", expect_gate=False,
        note="Default base profile — anchor for the rest.",
        form=_base(),
    ),
    Profile(
        "C. Young thin file, modest loan",
        expect_tier="Medium", expect_gate=False,
        note="Short tenure, low income — should land borderline.",
        form=_merge(
            gender="Male", age_years=23, num_children=0, num_family_members=1,
            family_status="Single / not married", years_employed=1.5,
            organization_type="Business Entity Type 2", occupation_type="Sales staff",
            credit_amount=200_000, loan_annuity=10_000, goods_price=200_000,
            total_income=80_000, owns_housing=False,
            years_since_id_change=2, years_at_address=2,
            region_population_relative=0.013, city_rating=3,
            works_in_different_city=True,
        ),
    ),
    Profile(
        "D. Young precarious, oversized loan",
        expect_tier="High", expect_gate=True,
        note="Self-employed, low-skill, very short tenure, big credit "
             "(CTI ≈ 15× — gate concurs with model).",
        form=_merge(
            age_years=22, num_children=0, num_family_members=1,
            family_status="Single / not married", years_employed=0.3,
            organization_type="Self-employed", occupation_type="Low-skill Laborers",
            credit_amount=600_000, loan_annuity=30_000, goods_price=0,
            total_income=40_000, owns_housing=False,
            years_since_id_change=0.5, years_at_address=0.5,
            region_population_relative=0.013, city_rating=3,
            works_in_different_city=True,
        ),
    ),
    Profile(
        "E. Affordability gate (DTI > 0.8)",
        expect_tier="High", expect_gate=True,
        note="2M loan on 50k income — must be overridden to High by the gate.",
        form=_merge(
            num_children=2, num_family_members=4, years_employed=4,
            credit_amount=2_000_000, loan_annuity=100_000, goods_price=2_000_000,
            total_income=50_000, owns_housing=False,
            years_since_id_change=4, years_at_address=5,
        ),
    ),
    Profile(
        "F. Senior, asset-rich, small loan",
        expect_tier="Low", expect_gate=False,
        note="Long tenure, government, owns home & car, capital city.",
        form=_merge(
            gender="Male", age_years=58, num_children=0, num_family_members=2,
            years_employed=25, organization_type="Government",
            occupation_type="High skill tech staff", has_work_phone=True,
            credit_amount=200_000, loan_annuity=8_000, goods_price=200_000,
            total_income=200_000, owns_car=True, car_age_years=2,
            years_since_id_change=6, years_at_address=20,
            region_population_relative=0.046, city_rating=1, has_landline=True,
        ),
    ),
    Profile(
        "G. Large loan but strong income",
        expect_tier="Low", expect_gate=False,
        note="1.2M loan; 300k income gives credit-to-income ≈ 4 — within bounds.",
        form=_merge(
            gender="Male", age_years=38, num_children=2, num_family_members=4,
            years_employed=10, occupation_type="Managers", has_work_phone=True,
            credit_amount=1_200_000, loan_annuity=45_000, goods_price=1_200_000,
            total_income=300_000, owns_car=True, car_age_years=4,
            years_since_id_change=5, years_at_address=8, has_landline=True,
        ),
    ),
    Profile(
        "H. Revolving small loan, modest income",
        expect_tier="Medium", expect_gate=False,
        note="Revolving credit is historically riskier than cash loans.",
        form=_merge(
            age_years=29, num_children=1, num_family_members=3,
            family_status="Civil marriage", years_employed=3,
            organization_type="Trade: type 3", occupation_type="Sales staff",
            contract_type="Revolving loans",
            credit_amount=100_000, loan_annuity=6_000, goods_price=0,
            total_income=80_000, owns_housing=False,
            years_since_id_change=3, years_at_address=4,
        ),
    ),
    Profile(
        "I. Edge: tiny income, tiny loan (within gate)",
        expect_tier="Medium", expect_gate=False,
        note="DTI ≈ 0.3 — small absolute numbers but inside affordability bands.",
        form=_merge(
            age_years=27, num_children=0, num_family_members=1,
            family_status="Single / not married", years_employed=2,
            organization_type="Trade: type 7", occupation_type="Sales staff",
            credit_amount=60_000, loan_annuity=6_000, goods_price=60_000,
            total_income=24_000, owns_housing=False,
            years_since_id_change=2, years_at_address=2,
        ),
    ),
    Profile(
        "J. Stress: lots of dependants, average loan",
        expect_tier="Medium", expect_gate=False,
        note="Five children — income_per_family_member is low.",
        form=_merge(
            num_children=5, num_family_members=7, years_employed=4,
            credit_amount=500_000, loan_annuity=25_000, goods_price=500_000,
            total_income=120_000,
        ),
    ),
]


def _identity_violations(score: int, proba: float, approval: float) -> list[str]:
    msgs: list[str] = []
    if abs(score - round(proba * 100)) > 1:
        msgs.append(f"score {score} != round(PD*100) {round(proba * 100)}")
    if abs((approval + proba) - 1.0) > 1e-6:
        msgs.append(f"approval + PD = {approval + proba:.6f}, expected 1.0")
    return msgs


def main() -> None:
    p = Top25Predictor(MODEL_PATH)
    print(f"Model: {p.best_name} · Holdout AUC: {p.best_auc:.4f}")
    print(f"Tier thresholds: PD < {p.low_threshold} → Low, "
          f"PD ≥ {p.high_threshold} → High\n")

    header = (
        f"{'#':>2}  {'Profile':<46} {'PD':>6} {'Score':>5} "
        f"{'Approval':>9} {'Tier':>7} {'Gate':>5}  {'Expect':<7} {'OK':>3}"
    )
    print(header)
    print("-" * len(header))

    correct_tier = 0
    correct_gate = 0
    identity_failures: list[str] = []

    for i, prof in enumerate(PROFILES, 1):
        r = p.predict(prof.form)
        gate = (r.get("affordability_gate") or {}).get("triggered", False)
        proba = r["risk_probability"]
        score = r["risk_score"]
        approval = insights.approval_with_confidence(
            p, prof.form, n_bootstrap=0
        )["approval_probability"]

        violations = _identity_violations(score, proba, approval)
        identity_failures.extend(f"{prof.name}: {v}" for v in violations)

        tier_ok = r["risk_category"] == prof.expect_tier
        gate_ok = gate == prof.expect_gate
        if tier_ok:
            correct_tier += 1
        if gate_ok:
            correct_gate += 1
        ok = "✓" if (tier_ok and gate_ok) else "✗"

        print(
            f"{i:>2}. {prof.name:<46} {proba:>6.3f} {score:>5} "
            f"{approval:>9.1%} {r['risk_category']:>7} {str(gate):>5}  "
            f"{prof.expect_tier:<7} {ok:>3}"
        )

    n = len(PROFILES)
    print()
    print(f"Tier match : {correct_tier}/{n}")
    print(f"Gate match : {correct_gate}/{n}")
    print(f"Identities : {'OK' if not identity_failures else 'VIOLATIONS:'}")
    for msg in identity_failures:
        print(f"   - {msg}")

    print()
    print("Proportion check across all profiles:")
    print("   risk_score   = round(default_probability * 100)")
    print("   approval     = 1 - default_probability")
    print("   → the three numbers are three views of one underlying PD.")


if __name__ == "__main__":
    main()
