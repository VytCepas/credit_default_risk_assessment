import streamlit as st
from pathlib import Path
import logging

from src.components.questionnaire import create_questionnaire_form
from src.components.results import create_results_display
from src.components.behavioral_traits import create_behavioral_traits_display
from src.models.risk_predictor import (
    load_risk_predictor,
    get_available_models,
    predict_with_explanations,
)
from src.models.behavioral_predictor import (
    predict_behavioral_traits,
    get_behavioral_model_info,
)

st.set_page_config(
    page_title="Home Credit Risk Assessment",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


AVAILABLE_MODELS = get_available_models()


def load_custom_css():
    """Load custom CSS styling"""
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e86c1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: #f8f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    
    .metric-container {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 0.5rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #ddd;
        margin-top: 3rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def display_header():
    """Display application header and introduction"""
    st.markdown(
        '<div class="main-header">🏛️ Home Credit Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
        <div class="info-box">
        <h4>Welcome to our intelligent loan risk assessment system</h4>
        <p>This tool uses advanced machine learning to evaluate loan default risk based on your personal and financial information. The assessment takes approximately 5-10 minutes to complete.</p>
        
        <p><strong>How it works:</strong></p>
        <ul>
            <li>📝 Complete the comprehensive questionnaire</li>
            <li>🤖 Our AI model analyzes your information</li>
            <li>📊 Receive detailed risk assessment and recommendations</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


def display_model_status():
    """Display model loading status"""
    with st.sidebar:
        st.header("System Status")

        # Show available models
        if AVAILABLE_MODELS:
            st.success(f"✅ {len(AVAILABLE_MODELS)} model(s) available")

            with st.expander("📋 Available Models"):
                for model_key, model_info in AVAILABLE_MODELS.items():
                    model_path = Path(model_info["path"])
                    if model_path.exists():
                        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
                        st.write(
                            f"{model_info['icon']} **{model_info['display_name']}**"
                        )
                        st.write(f"Size: {model_size:.1f} MB")
                        st.write("---")

        else:
            st.error("❌ No models found")
            st.warning("Please train models first using the training script")


def create_navigation_sidebar():
    """Create navigation sidebar for multi-page structure"""
    with st.sidebar:
        st.markdown("---")

        if st.button(
            "🔄 Clear Session & Start Over",
            use_container_width=True,
            help="Clear all cached data and start fresh",
        ):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.header("📊 Assessment Pages")

        if "current_page" not in st.session_state:
            st.session_state.current_page = "questionnaire"

        if st.button("📝 Questionnaire", use_container_width=True):
            st.session_state.current_page = "questionnaire"
            st.rerun()

        if st.session_state.get("assessment_completed", False):
            st.markdown("**Assessment Results:**")

            if len(AVAILABLE_MODELS) > 1:
                if st.button("📊 Model Comparison", use_container_width=True):
                    st.session_state.current_page = "comparison"
                    st.rerun()

            for model_key, model_info in AVAILABLE_MODELS.items():
                button_label = f"{model_info['icon']} {model_info['display_name']}"
                if st.button(button_label, use_container_width=True):
                    st.session_state.current_page = model_key
                    st.rerun()

        st.markdown("---")
        st.markdown("**🔮 Planned Features:**")
        if st.button("🚀 Future Enhancements", use_container_width=True):
            st.session_state.current_page = "future_features"
            st.rerun()


def show_future_features_page():
    """Display planned features and enhancements"""
    st.markdown(
        '<div class="section-header">🚀 Planned Features & Enhancements</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-box">
    <p>Our platform roadmap includes advanced analytics and ML capabilities currently in development.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Core Feature Enhancements
    st.markdown("### � Core Enhancements (In Development)")

    features = [
        {
            "icon": "💰",
            "name": "Financial Health Analysis",
            "desc": "Optimized financial ratios and stability metrics",
            "status": "Q1 2026",
        },
        {
            "icon": "📊",
            "name": "Statistical Inference Analysis",
            "desc": "Comprehensive hypothesis testing and validation",
            "status": "Q2 2026",
        },
        {
            "icon": "🌍",
            "name": "Geodemographic Insights",
            "desc": "Regional patterns and demographic clustering",
            "status": "Q2 2026",
        },
        {
            "icon": "🚨",
            "name": "Warnings & Red Flags System",
            "desc": "1M+ anomaly patterns and fraud detection",
            "status": "Q1 2026",
        },
        {
            "icon": "🔄",
            "name": "What-if Analysis Tooling",
            "desc": "Interactive scenario testing and optimization",
            "status": "Q1 2026",
        },
    ]

    cols = st.columns(2)
    for idx, feature in enumerate(features):
        with cols[idx % 2]:
            st.markdown(
                f"""
            <div style="
                border: 2px dashed #e0e0e0;
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
                background-color: #fafafa;
            ">
                <h4 style="margin: 0 0 8px 0;">{feature["icon"]} {feature["name"]}</h4>
                <p style="color: #666; margin: 0 0 8px 0; font-size: 0.9em;">{feature["desc"]}</p>
                <p style="margin: 0; font-size: 0.85em;"><strong>ETA:</strong> {feature["status"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Additional ML Models
    st.markdown("---")
    st.markdown("### 🤖 Additional ML Models (Planned)")

    models = [
        {"icon": "🚨", "name": "Fraud Detection Model", "eta": "Q1 2026"},
        {"icon": "💰", "name": "Income Verification Model", "eta": "Q2 2026"},
        {"icon": "💳", "name": "Credit Limit Optimizer", "eta": "Q2 2026"},
        {"icon": "⚠️", "name": "Early Warning System", "eta": "Q3 2026"},
        {"icon": "🎯", "name": "Cross-sell Propensity", "eta": "Q3 2026"},
    ]

    cols = st.columns(3)
    for idx, model in enumerate(models):
        with cols[idx % 3]:
            st.markdown(
                f"""
            <div style="
                text-align: center;
                padding: 10px;
                border: 1px dashed #ddd;
                border-radius: 6px;
                margin: 5px 0;
                background-color: #f9f9f9;
            ">
                <div style="font-size: 1.5em;">{model["icon"]}</div>
                <div style="font-size: 0.85em; font-weight: bold; margin: 5px 0;">{model["name"]}</div>
                <div style="font-size: 0.75em; color: #888;">{model["eta"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    if st.button("⬅️ Back to Assessment", use_container_width=True):
        if st.session_state.get("assessment_completed", False):
            st.session_state.current_page = "comparison"
        else:
            st.session_state.current_page = "questionnaire"
        st.rerun()


def show_questionnaire_page():
    """Display the questionnaire page"""
    st.markdown(
        '<div class="section-header">Complete Assessment Questionnaire</div>',
        unsafe_allow_html=True,
    )

    questionnaire = create_questionnaire_form()
    responses = questionnaire.render_form()

    if responses:
        previous_responses = st.session_state.get("questionnaire_data", {})
        responses_changed = previous_responses != responses

        if responses_changed:
            logger.info("New responses detected - clearing previous results")
            st.session_state.model_results = {}
            st.session_state.assessment_completed = False

        st.session_state.questionnaire_data = responses

        if not st.session_state.get("model_results") or responses_changed:
            st.session_state.model_results = {}

        progress_text = st.empty()
        progress_bar = st.progress(0)

        try:
            total_models = len(AVAILABLE_MODELS)

            for i, (model_key, model_info) in enumerate(AVAILABLE_MODELS.items()):
                progress = (i + 1) / total_models
                progress_text.text(
                    f"🔍 Processing {model_info['display_name']} ({i + 1}/{total_models})..."
                )
                progress_bar.progress(progress)

                predictor = load_risk_predictor(model_info["path"])

                logger.info(
                    f"Processing responses for {model_info['display_name']}: {responses}"
                )

                prediction_results, shap_values, processed_features = (
                    predict_with_explanations(predictor, responses)
                )

                logger.info(
                    f"Prediction results for {model_info['display_name']}: Risk Score {prediction_results.get('risk_score', 'Unknown')}, Probability {prediction_results.get('risk_probability', 'Unknown')}"
                )

                st.session_state.model_results[model_key] = {
                    "prediction_results": prediction_results,
                    "shap_values": shap_values,
                    "processed_features": processed_features,
                    "model_instance": predictor,
                    "model_info": model_info,
                    "input_snapshot": dict(responses),  # Store copy of inputs
                }

            progress_text.empty()
            progress_bar.empty()

            st.session_state.assessment_completed = True
            st.success("✅ Assessment completed successfully!")

            if AVAILABLE_MODELS:
                first_model = list(AVAILABLE_MODELS.keys())[0]
                st.session_state.current_page = first_model

            st.rerun()

        except Exception as e:
            progress_text.empty()
            progress_bar.empty()

            st.error(f"❌ Assessment failed: {str(e)}")
            logger.error(f"Prediction error: {str(e)}")

            if st.checkbox("Show debug information"):
                st.exception(e)


def show_model_results_page(model_key: str):
    """Display results for a specific model"""
    if model_key not in AVAILABLE_MODELS:
        st.error(f"❌ Model '{model_key}' not found")
        return

    if not st.session_state.get("assessment_completed", False):
        st.warning("⚠️ Please complete the questionnaire first")
        return

    model_info = AVAILABLE_MODELS[model_key]
    model_results = st.session_state.model_results.get(model_key)

    if not model_results:
        st.error(f"❌ No results available for {model_info['display_name']}")
        return

    st.markdown(
        f'<div class="section-header">{model_info["icon"]} {model_info["display_name"]} Results</div>',
        unsafe_allow_html=True,
    )

    st.info(f"📋 **Model Description:** {model_info['description']}")

    results_display = create_results_display()
    results_display.display_results(
        prediction_results=model_results["prediction_results"],
        shap_values=model_results["shap_values"],
        processed_features=model_results["processed_features"],
    )

    # Add Behavioral Traits Analysis
    try:
        behavioral_traits = predict_behavioral_traits(
            st.session_state.questionnaire_data
        )
        behavioral_display = create_behavioral_traits_display()
        behavioral_display.display_behavioral_traits(behavioral_traits)
    except Exception as e:
        st.warning(f"⚠️ Behavioral traits analysis unavailable: {str(e)}")
        logger.warning(f"Behavioral traits prediction failed: {e}")

    with st.expander("📋 View Questionnaire Responses", expanded=False):
        questionnaire = create_questionnaire_form()
        questionnaire.display_summary(st.session_state.questionnaire_data)

    with st.expander("🔍 Debug: Input/Output Verification", expanded=False):
        st.markdown(
            "**This panel helps verify the model is receiving different inputs and producing different outputs.**"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Key Inputs:**")
            input_snapshot = model_results.get("input_snapshot", {})
            debug_inputs = {
                "Age": input_snapshot.get("age", "N/A"),
                "Income": f"${input_snapshot.get('total_income', 0):,.0f}",
                "Credit": f"${input_snapshot.get('credit_amount', 0):,.0f}",
                "Employment": input_snapshot.get("employment_status", "N/A"),
                "Education": input_snapshot.get("education_level", "N/A"),
            }
            for key, value in debug_inputs.items():
                st.text(f"{key}: {value}")

        with col2:
            st.markdown("**Model Output:**")
            pred = model_results["prediction_results"]
            st.text(f"Risk Score: {pred.get('risk_score', 'N/A')}/1000")
            st.text(f"Probability: {pred.get('risk_probability', 0):.3f}")
            st.text(f"Category: {pred.get('risk_category', 'N/A')}")

        st.info(
            "💡 **Tip:** If you see the same risk score after changing inputs, click '🔄 Clear Session & Start Over' in the sidebar and try again."
        )

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Back to Questionnaire"):
            st.session_state.current_page = "questionnaire"
            st.rerun()

    with col3:
        model_keys = list(AVAILABLE_MODELS.keys())
        current_index = model_keys.index(model_key)

        if current_index < len(model_keys) - 1:
            next_model_key = model_keys[current_index + 1]
            next_model_info = AVAILABLE_MODELS[next_model_key]
            if st.button(f"➡️ {next_model_info['display_name']}"):
                st.session_state.current_page = next_model_key
                st.rerun()


def show_model_comparison_page():
    """Display comparison of all model results"""
    if not st.session_state.get("assessment_completed", False):
        st.warning("⚠️ Please complete the questionnaire first")
        return

    st.markdown(
        '<div class="section-header">📊 Model Comparison Overview</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "🔍 **Comparison Overview:** Compare risk assessments from all available models to get a comprehensive view of your loan application."
    )

    comparison_data = []

    for model_key, model_results in st.session_state.model_results.items():
        model_info = AVAILABLE_MODELS[model_key]
        prediction_results = model_results["prediction_results"]

        comparison_data.append(
            {
                "Model": f"{model_info['icon']} {model_info['display_name']}",
                "Risk Level": prediction_results.get("risk_category", "Unknown"),
                "Risk Score": f"{prediction_results.get('risk_score', 0)}/1000",
                "Probability": f"{prediction_results.get('risk_probability', 0):.1%}",
                "Description": model_info["description"],
            }
        )

    if comparison_data:
        import pandas as pd

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("### 📈 Risk Level Distribution")

        col1, col2 = st.columns(2)

        with col1:
            risk_levels = [item["Risk Level"] for item in comparison_data]
            risk_counts = pd.Series(risk_levels).value_counts()

            import plotly.express as px

            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution Across Models",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            risk_scores = [float(item["Risk Score"]) for item in comparison_data]
            model_names = [item["Model"] for item in comparison_data]

            fig_bar = px.bar(
                x=model_names,
                y=risk_scores,
                title="Risk Scores by Model",
                labels={"x": "Model", "y": "Risk Score"},
            )
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Detailed Model Analysis")
    st.write("Click on any model below to view detailed analysis and explanations:")

    cols = st.columns(min(len(AVAILABLE_MODELS), 3))

    for i, (model_key, model_info) in enumerate(AVAILABLE_MODELS.items()):
        col_idx = i % len(cols)
        with cols[col_idx]:
            if st.button(
                f"{model_info['icon']} {model_info['display_name']}",
                key=f"goto_{model_key}",
                use_container_width=True,
            ):
                st.session_state.current_page = model_key
                st.rerun()

    st.markdown("---")
    if st.button("⬅️ Back to Questionnaire"):
        st.session_state.current_page = "questionnaire"
        st.rerun()


def main():
    """Main application function with multi-page structure"""
    load_custom_css()
    display_header()
    display_model_status()
    create_navigation_sidebar()

    if "assessment_completed" not in st.session_state:
        st.session_state.assessment_completed = False
    if "questionnaire_data" not in st.session_state:
        st.session_state.questionnaire_data = None
    if "model_results" not in st.session_state:
        st.session_state.model_results = {}
    if "current_page" not in st.session_state:
        st.session_state.current_page = "questionnaire"

    if not AVAILABLE_MODELS:
        st.error("🚨 **No Models Available**")
        st.markdown(
            """
        <div class="error-box">
        <p>No machine learning models are available. Please follow these steps:</p>
        <ol>
            <li>Navigate to the <code>streamlit_app/src/models/</code> directory</li>
            <li>Run <code>python train_model.py</code> to train the models</li>
            <li>Ensure model files are saved in the project root directory</li>
            <li>Refresh this page once training is complete</li>
        </ol>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Refresh Page"):
            st.rerun()
        return

    current_page = st.session_state.current_page

    if current_page == "questionnaire":
        show_questionnaire_page()
    elif current_page == "comparison":
        show_model_comparison_page()
    elif current_page == "future_features":
        show_future_features_page()
    elif current_page in AVAILABLE_MODELS:
        show_model_results_page(current_page)
    else:
        st.session_state.current_page = "questionnaire"
        st.rerun()

    with st.sidebar:
        st.markdown("---")
        if st.session_state.get("assessment_completed", False):
            if st.button("🔄 Start New Assessment", use_container_width=True):
                # Reset session state
                st.session_state.assessment_completed = False
                st.session_state.questionnaire_data = None
                st.session_state.model_results = {}
                st.session_state.current_page = "questionnaire"
                st.rerun()

    display_footer()


def display_footer():
    """Display application footer"""
    st.markdown(
        """
    <div class="footer">
    <hr>
    <p><strong>🏛️ Home Credit Risk Assessment System</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p><em>This tool is for educational and demonstration purposes. 
    Actual loan decisions should involve comprehensive review by qualified professionals.</em></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def handle_errors():
    """Global error handler"""

    def error_handler(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.error(f"Application error: {str(e)}")

                with st.expander("🔍 Error Details"):
                    st.exception(e)

                if st.button("🔄 Restart Application"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

        return wrapper

    return error_handler


@handle_errors()
def protected_main():
    main()


if __name__ == "__main__":
    try:
        protected_main()
    except Exception as e:
        st.error(f"🚨 Critical application error: {str(e)}")
        logger.critical(f"Critical error: {str(e)}")

        st.markdown(
            """
        **Something went wrong!** 
        
        Please try refreshing the page. If the problem persists, check:
        - All required packages are installed
        - The model file (home_credit_model.pkl) exists in the project root
        - All source files are present
        """
        )

        if st.button("🔄 Refresh Page"):
            st.rerun()
