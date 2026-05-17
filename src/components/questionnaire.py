"""
Questionnaire form component for the Credit Default Risk Assessment Streamlit app.

Collects the 15 input fields required by the risk prediction pipeline and passes
them to QuestionnaireToFeatures for preprocessing before model inference.

Field groups
------------
Personal Information : gender, age
Financial Information : total_income, employment_status, years_employed
Education and Family : education_level, family_status, num_children, num_family_members
Assets               : owns_car, owns_housing, housing_type
Loan Details         : contract_type, credit_amount, loan_annuity

All options in selectbox fields match the OneHotEncoder categories fitted during training.
Adding or removing options here will cause an encoder mismatch and raise a ValueError.

Related issue: #19
"""
import streamlit as st
from typing import Any, Optional


class QuestionnaireForm:
    """Streamlit form collecting 15 user inputs for credit risk scoring.

    Renders labelled widgets (selectbox, number_input) for each field,
    validates required fields on submission, and returns a raw dict
    passed directly to ``QuestionnaireToFeatures.transform()``.
    """

    def __init__(self):
        self.questions = self._define_questions()

    def _define_questions(self) -> dict[str, dict[str, Any]]:
        """Return the ordered field definitions used to render the form.

        Each entry maps a field key to its widget configuration.
        Keys must match the ``questionnaire_mapping`` in ``QuestionnaireToFeatures``.
        """
        return {
            # Personal Information
            "gender": {
                "label": "What is your gender?",
                "type": "selectbox",
                "options": ["Female", "Male"],
                "required": True,
                "section": "Personal Information",
            },
            "age": {
                "label": "What is your age?",
                "type": "number_input",
                "min_value": 18,
                "max_value": 80,
                "value": 30,
                "required": True,
                "section": "Personal Information",
            },
            # Financial Information
            "total_income": {
                "label": "What is your total annual income (in local currency)?",
                "type": "number_input",
                "min_value": 0,
                "max_value": 10000000,
                "value": 150000,
                "required": True,
                "section": "Financial Information",
            },
            "employment_status": {
                "label": "What is your employment status?",
                "type": "selectbox",
                "options": [
                    "Working",
                    "State servant",
                    "Commercial associate",
                    "Pensioner",
                    "Unemployed",
                    "Student",
                    "Businessman",
                    "Maternity leave",
                ],
                "required": True,
                "section": "Financial Information",
            },
            "years_employed": {
                "label": "How many years have you been employed in your current job?",
                "type": "number_input",
                "min_value": 0,
                "max_value": 50,
                "value": 5,
                "required": True,
                "section": "Financial Information",
            },
            # Education and Family
            "education_level": {
                "label": "What is your highest level of education?",
                "type": "selectbox",
                "options": [
                    "Lower secondary",
                    "Secondary / secondary special",
                    "Incomplete higher",
                    "Higher education",
                    "Academic degree",
                ],
                "required": True,
                "section": "Education and Family",
            },
            "family_status": {
                "label": "What is your family status?",
                "type": "selectbox",
                "options": [
                    "Single / not married",
                    "Married",
                    "Civil marriage",
                    "Widow",
                    "Separated",
                ],
                "required": True,
                "section": "Education and Family",
            },
            "num_children": {
                "label": "How many children do you have?",
                "type": "number_input",
                "min_value": 0,
                "max_value": 20,
                "value": 0,
                "required": True,
                "section": "Education and Family",
            },
            "num_family_members": {
                "label": "How many family members live with you (including yourself)?",
                "type": "number_input",
                "min_value": 1,
                "max_value": 20,
                "value": 2,
                "required": True,
                "section": "Education and Family",
            },
            # Assets and Housing
            "owns_car": {
                "label": "Do you own a car?",
                "type": "selectbox",
                "options": ["Yes", "No"],
                "required": True,
                "section": "Assets and Housing",
            },
            "owns_housing": {
                "label": "Do you own a house or apartment?",
                "type": "selectbox",
                "options": ["Yes", "No"],
                "required": True,
                "section": "Assets and Housing",
            },
            "housing_type": {
                "label": "What is your housing situation?",
                "type": "selectbox",
                "options": [
                    "House / apartment",
                    "With parents",
                    "Municipal apartment",
                    "Rented apartment",
                    "Office apartment",
                    "Co-op apartment",
                ],
                "required": True,
                "section": "Assets and Housing",
            },
            # Loan Information
            "contract_type": {
                "label": "What type of loan are you applying for?",
                "type": "selectbox",
                "options": ["Cash loans", "Revolving loans"],
                "required": True,
                "section": "Loan Information",
            },
            "credit_amount": {
                "label": "What is the credit amount you are requesting (in local currency)?",
                "type": "number_input",
                "min_value": 1000,
                "max_value": 10000000,
                "value": 200000,
                "required": True,
                "section": "Loan Information",
            },
            "loan_annuity": {
                "label": "What loan annuity (monthly payment) can you afford (in local currency)?",
                "type": "number_input",
                "min_value": 0,
                "max_value": 1000000,
                "value": 0,
                "required": False,
                "section": "Loan Information",
                "help": "Optional: enter 0 to let the app estimate a monthly payment from the requested credit amount",
            },
        }

    def render_form(self) -> Optional[dict[str, Any]]:
        """
        Render the complete questionnaire form

        Returns:
            dict containing all form responses or None if not submitted
        """
        st.title("🏦 Home Credit Risk Assessment")
        st.markdown("---")
        st.markdown("""
        **Welcome to the Home Credit Risk Assessment Tool**
        
        This questionnaire takes approximately 5-10 minutes to complete and helps the model estimate your loan risk.
        The loan annuity field is optional; leave it at 0 if you want the app to estimate a monthly payment for you.
        """)

        if "form_data" not in st.session_state:
            st.session_state.form_data = {}

        with st.form("risk_assessment_form"):
            responses = {}

            sections = {}
            for key, question in self.questions.items():
                section = question["section"]
                if section not in sections:
                    sections[section] = []
                sections[section].append((key, question))

            for section_name, section_questions in sections.items():
                st.subheader(f"📋 {section_name}")

                for question_key, question_config in section_questions:
                    response = self._render_question(question_key, question_config)
                    responses[question_key] = response

                st.markdown("---")

            _, col2, _ = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button(
                    "🔍 Assess Risk", type="primary", use_container_width=True
                )

            if submitted:
                validation_errors = self._validate_responses(responses)

                if validation_errors:
                    st.error("Please correct the following errors:")
                    for error in validation_errors:
                        st.error(f"• {error}")
                    return None
                else:
                    st.session_state.form_data = responses
                    return responses

        return None

    def _render_question(
        self, question_key: str, question_config: dict[str, Any]
    ) -> Any:
        """Render a single question based on its configuration"""

        label = question_config["label"]
        if question_config.get("required", False):
            label += " *"

        help_text = question_config.get("help", None)

        if question_config["type"] == "selectbox":
            options = question_config["options"]
            if question_config.get("required", False):
                options = ["Please select..."] + options
                index = 0
            else:
                index = 0 if len(options) > 0 else None

            value = st.selectbox(
                label, options=options, index=index, key=question_key, help=help_text
            )

            if value == "Please select...":
                return None
            return value

        elif question_config["type"] == "number_input":
            return st.number_input(
                label,
                min_value=question_config.get("min_value", None),
                max_value=question_config.get("max_value", None),
                value=question_config.get("value", 0),
                key=question_key,
                help=help_text,
            )

        elif question_config["type"] == "text_input":
            return st.text_input(
                label,
                value=question_config.get("value", ""),
                key=question_key,
                help=help_text,
            )

        else:
            st.error(f"Unknown question type: {question_config['type']}")
            return None

    def _validate_responses(self, responses: dict[str, Any]) -> list[str]:
        """Validate form responses and return list of errors"""
        errors = []

        for question_key, question_config in self.questions.items():
            if question_config.get("required", False):
                response = responses.get(question_key)

                if response is None or response == "" or response == "Please select...":
                    errors.append(
                        f"{question_config['label'].replace(' *', '')} is required"
                    )

                elif question_key == "age" and (response < 18 or response > 80):
                    errors.append("Age must be between 18 and 80 years")

                elif question_key == "total_income" and response <= 0:
                    errors.append("Total income must be greater than 0")

                elif question_key == "credit_amount" and response < 1000:
                    errors.append("Credit amount must be at least 1,000")

        if "num_children" in responses and "num_family_members" in responses:
            if responses["num_children"] > responses["num_family_members"]:
                errors.append("Number of children cannot exceed total family members")

        if "credit_amount" in responses and "loan_annuity" in responses:
            if (
                responses["loan_annuity"]
                and responses["loan_annuity"] * 12 > responses["credit_amount"]
            ):
                errors.append("Annual loan payments cannot exceed the credit amount")

        return errors

    def get_question_info(self, question_key: str) -> dict[str, Any]:
        """Get information about a specific question"""
        return self.questions.get(question_key, {})

    def get_all_questions(self) -> dict[str, dict[str, Any]]:
        """Get all questions configuration"""
        return self.questions

    def display_summary(self, responses: dict[str, Any]):
        """Display a summary of all responses"""
        st.subheader("📊 Response Summary")

        sections = {}
        for key, question in self.questions.items():
            section = question["section"]
            if section not in sections:
                sections[section] = []
            sections[section].append((key, question))

        for section_name, section_questions in sections.items():
            with st.expander(f"{section_name}", expanded=False):
                for question_key, question_config in section_questions:
                    if question_key in responses:
                        value = responses[question_key]
                        if value is not None:
                            label = question_config["label"].replace(" *", "")
                            if question_key == "loan_annuity" and value in (
                                0,
                                0.0,
                                None,
                            ):
                                st.write(f"**{label}:** Auto-estimated by the app")
                            else:
                                st.write(f"**{label}:** {value}")


def create_questionnaire_form() -> QuestionnaireForm:
    """Factory function to create questionnaire form"""
    return QuestionnaireForm()
