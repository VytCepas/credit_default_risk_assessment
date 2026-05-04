# Stage 1

**Project title:**
AI-Based Credit Default Risk Prediction System

**Sector:**
Finance

**Project idea:**
The system uses client's personal data (income, age, employment, family status, etc.) to predict the probability that a client is eligible to receive a loan.

The solution can be applied in:

- Banks
- Fintech companies
- Online loan lending platforms

## Business Model

- A financial institution provides loans to customers
- Before approval, each client is evaluated
- The system calculates a risk score
- Based on the score, the institution:
  - approves the loan
  - rejects the loan
  - adjusts loan conditions (for example, interest rate)

## Value Proposition

- Reduce default risk for loaning companies
- Automate and fast decision-making
- Speed up loan approval process
- Clients can check their loan contract terms on their own before reaching out to the company

## Industry Overview (Finance / Lending)

- Has high competition
- Strong risk management requirements
- Strict regulatory environment

**Key trends:**

- Integration of AI in credit scoring systems
- Automation of decision making systems
- Real-time risk evaluation

## Market Situation

- Traditional banks rely on older scoring methods
- Fintech companies leverage AI for faster decisions
- Customers expect:
  - Quick loan approvals
  - Minimal paperwork

## Competitors

**Main competitors:**

- Traditional banks (Swedbank, SEB)
- Fintech companies (Revolut)
- Credit bureaus (Bobutės paskola)

**Competitive advantage — AI models can:**

- Improve prediction accuracy
- Faster processing of larger datasets
- Adapt to new patterns faster
- Reduce workload for workers

---

## Three Major Challenges

### 1. Market Disruption

**Problem:** Fintech companies are transforming the lending market by offering:

- Faster services
- Simplified processes

**Impact:** Traditional credit scoring approaches might become outdated in the near future.

### 2. Technological Change

**Problem:** Rapid advancements in AI and data processing.

**So:**

- Models require continuous updates
- Increasing data volume and complexity
- Need for model monitoring (MLOps)

### 3. Competitive Pressure

**Problem:** Strong competition between:

- Banks
- Fintech companies
- Lending platforms

---

# Stage 2

## Problem Analysis and Goals

### Problem Analysis

Financial institutions may face issues such as:

- Manual evaluation of clients which consumes much time
- Traditional scoring models are less accurate and outdated
- High risk of loan sometimes leads to financial losses
- Inconsistent decision making due to human factors

### Goals

The main goals of the system:

- Improve risk prediction accuracy
- Automate decision making process
- Speed up loan approval
- Reduce employee workload

---

## Solutions to Business Problems

| Problem | Solution | Expected Impact |
|---------|----------|-----------------|
| Manual risk assessment | AI-based risk prediction model | Faster and automated decisions |
| Low prediction accuracy | Machine learning model trained on historical data | Improved risk evaluation |
| Slow loan approval | Real-time scoring system | Better customer experience |
| High employee workload | Automated decision support system | Increased efficiency |

---

## Current Documents

| Document | Description | Usage |
|----------|-------------|-------|
| Loan application form | Provided personal and financial data by client | Input for AI model |
| Credit bureau report | External credit history from financial institutions | Risk assessment |
| Previous loan records | Historical loan applications and decisions | Behavioral analysis |
| Payment history | Records of past repayments and delays | Default prediction |
| Employment and income data | Income level and job stability | Financial reliability |

---

## Product Vision, Goals, Success Factors (OKR/KPI)

### Product Vision

To develop an AI-powered credit risk assessment system which is fast, accurate and automated.

### Goals

- Build a reliable risk prediction model
- Integrate AI into loan approval workflow
- Improve customer experience
- Reduce operational costs

### Success Factors (OKR/KPI)

**OKRs:**

- Maintain ROC-AUC ≥ 0.65
- Enable automated risk scoring for loan applications
- Identify high-risk clients consistently

**KPIs:**

- ROC-AUC score
- Default prediction rate
- Percentage of automated decisions
- Processing time per loan application

---

## AI Use Cases

### 1. Credit Risk Prediction

AI predicts the probability of loan default based on client data (income, age, employment, credit amount).

**Value:** Improves credit decision accuracy and reduces financial risk.

### 2. Automated Loan Approval

AI evaluates applications and automatically approves, rejects, or flags them for review based on risk score.

**Value:** Speeds up decision making and at the same time reduces manual work.

### 3. Customer Risk Segmentation

AI groups clients into low, medium, and high-risk categories based on default probability.

**Value:** Enables better pricing and automated personalized offers.

---

## ML Model Context

### Available Data Analysis

Dataset: **Home Credit Default Risk** — publicly available on Kaggle.

| Feature | Type | Description | Relevance |
|---------|------|-------------|-----------|
| `AMT_CREDIT` | Numerical | Total loan amount requested | Core credit risk indicator |
| `AMT_INCOME_TOTAL` | Numerical | Annual income of applicant | Repayment capacity |
| `DAYS_BIRTH` | Numerical | Age of applicant (days from today, negative) | Demographic risk factor |
| `DAYS_EMPLOYED` | Numerical | Employment length (days) | Job stability |
| `CNT_FAM_MEMBERS` | Numerical | Number of family members | Financial obligations |
| `CNT_CHILDREN` | Numerical | Number of children | Financial obligations |
| `AMT_ANNUITY` | Numerical | Loan annuity payment amount | Repayment burden |
| `CODE_GENDER` | Binary | Applicant gender (M/F) | Demographic feature |
| `FLAG_OWN_CAR` | Binary | Car ownership | Asset indicator |
| `FLAG_OWN_REALTY` | Binary | Real estate ownership | Asset indicator |
| `NAME_CONTRACT_TYPE` | Binary | Cash loan vs. revolving loan | Loan type risk |
| `NAME_INCOME_TYPE` | Categorical | Employment category | Income stability |
| `NAME_EDUCATION_TYPE` | Categorical | Highest education level | Socioeconomic indicator |
| `NAME_FAMILY_STATUS` | Categorical | Marital status | Demographic factor |
| `NAME_HOUSING_TYPE` | Categorical | Housing situation | Stability indicator |
| `TARGET` | Binary label | 1 = defaulted, 0 = repaid | Prediction target |

**Dataset statistics:**
- Total samples: 307,511
- Default rate: ~8.1% (class imbalance addressed with SMOTETomek)
- Final feature count after encoding: 32 (7 numerical + 4 binary + 21 one-hot encoded categorical)

### Required Decision-Making Analysis

The model must support the following decision points:

| Decision | Input | Output | Threshold |
|----------|-------|--------|-----------|
| Approve / Reject loan | Client application data | Default probability (0–1) | 0.37 (optimised) |
| Assign risk tier | Default probability | Low / Medium / High risk category | Low <0.3 / Med 0.3–0.6 / High >0.6 |
| Personalise loan terms | Risk tier | Interest rate adjustment suggestion | Based on tier |
| Flag for manual review | Model confidence | Review flag | Borderline cases 0.3–0.45 |

### Expected Business Value

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| Cost reduction | Fewer manual credit analyst hours | 30–50% reduction in assessment time |
| Revenue growth | More accurate approvals = fewer missed good clients | Estimated 5–10% increase in approved volume |
| Risk reduction | Fewer defaults due to better screening | Reduction in non-performing loan rate |
| Customer experience | Faster decisions (real-time vs. days) | Higher applicant satisfaction |

---

# Stage 3: Event Storming

> **Status:** Placeholder — to be completed in Miro or Prooph Board before 05.09.
>
> **Tool:** [Miro Event Storming Template](https://miro.com/miroverse/event-storming/)

## Proposed Event Flow

The following events, commands, and actors form the basis of the Event Storming session. Transfer these to sticky notes in your chosen tool.

### Actors (Yellow)

- **Loan Applicant** — submits application
- **Credit Analyst** — reviews flagged cases
- **System / AI Model** — scores and classifies
- **Loan Officer** — makes final approval decision

### Commands / Activities (Blue)

1. Submit loan application form
2. Validate and preprocess input data
3. Run credit risk prediction model
4. Calculate risk score (0–1000)
5. Classify applicant into risk tier (Low / Medium / High)
6. Generate SHAP explanation for decision
7. Auto-approve (Low risk) / Auto-reject (High risk)
8. Flag application for manual review (Medium / borderline)
9. Credit Analyst reviews flagged case
10. Loan Officer issues final decision
11. Notify applicant of outcome
12. Log decision to audit trail

### Risk Hotspots (Red)

- Model bias in demographic features (gender, age)
- Class imbalance in training data leading to poor minority detection
- Regulatory compliance for automated decisions (EU AI Act / GDPR)
- Model drift over time — requires periodic retraining
- Lack of explainability could make decisions non-auditable

### Opportunities (Green)

- Integrate external credit bureau data to improve accuracy
- Add real-time income verification via open banking APIs
- Deploy MLOps pipeline for automatic retraining
- Extend to behavioural risk profiling (already partially built)
- Offer applicant-facing self-service risk check tool

## User Story Map (Draft)

> Transfer to Miro as a horizontal story map with swim lanes per actor.

| Applicant | System | Analyst | Officer |
|-----------|--------|---------|---------|
| Submit form | Validate data | — | — |
| — | Score application | — | — |
| — | Classify risk tier | — | — |
| — | Auto-approve / reject | — | — |
| — | Flag for review | Review case | — |
| — | — | — | Issue decision |
| Receive notification | Log to audit | — | — |

---

# Stage 4: TA-1 Defense Preparation

**Defense date: 05.14**

## Checklist

- [ ] All team members confirmed for attendance
- [ ] Upload `.txt` file with team details to Moodle → 1 practical work
- [ ] Report finalised and uploaded (GitHub Wiki / this document)
- [ ] Presentation slides prepared (see below)
- [ ] Event Storming board completed in Miro/Prooph Board
- [ ] Record a short walkthrough of the Event Storming activity

## Suggested Slide Structure

1. **Title slide** — Project name, team, date
2. **Problem statement** — What we solve and why it matters
3. **Business model** — Industry, competitors, market situation
4. **Three major challenges** — Market disruption, tech change, competitive pressure
5. **AI use cases** — 3 identified use cases with expected value
6. **ML model overview** — Dataset, features, approach
7. **Event Storming result** — Screenshot of the Miro/Prooph board
8. **OKR/KPI targets** — What does success look like?
9. **Next steps** — TA2 plan

## Team

| Name | Role | GitHub |
|------|------|--------|
| Vytautas Cepas | Data Scientist | @VytCepas |
| Laurynas Zalaga | Data Scientist | @Gitlaurynas |
