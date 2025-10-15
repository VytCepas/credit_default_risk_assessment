import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


class ResultsDisplay:
    """
    Simple risk assessment results display
    """

    def __init__(self):
        self.risk_colors = {
            "Very Low Risk": "green",
            "Low Risk": "lightgreen",
            "Medium Risk": "orange",
            "High Risk": "red",
            "Very High Risk": "darkred",
            "Error": "gray",
        }

    def display_results(
        self,
        prediction_results: dict[str, str | int | float],
        shap_values=None,
        processed_features=None,
    ):
        """
        Display simple risk assessment results

        Args:
            prediction_results: Dictionary containing prediction results
            shap_values: SHAP values for feature importance
            processed_features: Processed feature data
        """
        st.markdown("---")
        st.title("Risk Assessment Results")

        if "error" in prediction_results:
            st.error(f"Error: {prediction_results['error']}")
            return

        risk_score = prediction_results.get("risk_score", 0)
        risk_category = prediction_results.get("risk_category", "Unknown")
        risk_probability = prediction_results.get("risk_probability", 0.0)

        st.subheader("Risk Assessment")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Risk Score",
                f"{risk_score}/1000",
                help="Risk score from 0 (lowest risk) to 1000 (highest risk)",
            )

        with col2:
            st.metric(
                "Risk Probability",
                f"{risk_probability:.1%}",
                help="Probability of default",
            )

        with col3:
            color = self.risk_colors.get(risk_category, "gray")
            st.markdown(
                f"""
                <div style="text-align: center; padding: 10px; border-radius: 10px; 
                    background-color: {color}; color: white; font-weight: bold; font-size: 18px;">
                    {risk_category}
                </div>
                """,
                unsafe_allow_html=True,
            )

        recommendation = prediction_results.get(
            "recommendation", "No recommendation available"
        )
        st.info(f"**Recommendation:** {recommendation}")

        if shap_values is not None and processed_features is not None:
            self._display_feature_importance(shap_values, processed_features)

    def _display_feature_importance(self, shap_values, processed_features):
        """Display simple feature importance with positive/negative impact"""
        st.subheader("🎯 Feature Importance Analysis")
        st.markdown(
            "Understanding which features have the most impact on your risk assessment:"
        )

        try:
            if isinstance(shap_values, list) and len(shap_values) > 0:
                feature_names = shap_values
                if isinstance(processed_features, list):
                    importance_df = pd.DataFrame(
                        {
                            "feature": feature_names,
                            "shap_value": [0] * len(feature_names),
                            "abs_importance": range(len(feature_names), 0, -1),
                        }
                    ).head(10)
                else:
                    st.warning("Unexpected format for SHAP values")
                    return
            elif isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
                    shap_values = shap_values[0]

                feature_names = (
                    processed_features.columns.tolist()
                    if hasattr(processed_features, "columns")
                    else processed_features
                    if isinstance(processed_features, list)
                    else [f"Feature_{i}" for i in range(len(shap_values))]
                )

                importance_df = (
                    pd.DataFrame(
                        {
                            "feature": feature_names,
                            "shap_value": shap_values,
                            "abs_importance": np.abs(shap_values),
                        }
                    )
                    .sort_values("abs_importance", ascending=False)
                    .head(10)
                )
            else:
                st.warning(f"Unexpected shap_values type: {type(shap_values)}")
                return

            importance_df["feature_display"] = importance_df["feature"].apply(
                lambda x: x.replace("_", " ").title() if isinstance(x, str) else str(x)
            )

            st.markdown("**Top 10 Most Important Features:**")
            st.caption(
                "🔴 Red bars increase risk | 🟢 Green bars decrease risk | Bar length shows impact strength"
            )

            colors = [
                "rgba(220, 20, 60, 0.8)" if x > 0 else "rgba(46, 139, 87, 0.8)"
                for x in importance_df["shap_value"]
            ]

            fig = go.Figure(
                go.Bar(
                    x=importance_df["shap_value"],
                    y=importance_df["feature_display"],
                    orientation="h",
                    marker=dict(
                        color=colors,
                        line=dict(color="rgba(0, 0, 0, 0.3)", width=1),
                    ),
                    text=[f"{x:+.4f}" for x in importance_df["shap_value"]],
                    textposition="outside",
                    textfont=dict(size=12, color="black"),
                    hovertemplate="<b>%{y}</b><br>"
                    + "Impact: %{x:.4f}<br>"
                    + "<extra></extra>",
                )
            )

            fig.update_layout(
                title={
                    "text": "Feature Impact on Risk Assessment",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 18, "color": "#1f77b4"},
                },
                xaxis_title="Impact on Risk Score",
                yaxis_title="",
                height=max(400, len(importance_df) * 50),
                yaxis={
                    "categoryorder": "total ascending",
                    "tickfont": {"size": 12},
                },
                xaxis={"tickfont": {"size": 11}, "zeroline": True},
                plot_bgcolor="rgba(240, 240, 240, 0.5)",
                paper_bgcolor="white",
                margin=dict(l=20, r=80, t=60, b=60),
                showlegend=False,
            )

            fig.add_vline(
                x=0, line_width=2, line_dash="dash", line_color="rgba(0, 0, 0, 0.4)"
            )

            fig.add_annotation(
                x=importance_df["shap_value"].max() * 0.7,
                y=len(importance_df) - 0.5,
                text="Increases Risk →",
                showarrow=False,
                font=dict(size=11, color="rgba(220, 20, 60, 0.8)"),
            )
            fig.add_annotation(
                x=importance_df["shap_value"].min() * 0.7,
                y=len(importance_df) - 0.5,
                text="← Decreases Risk",
                showarrow=False,
                font=dict(size=11, color="rgba(46, 139, 87, 0.8)"),
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("ℹ️ How to interpret this chart"):
                st.markdown(
                    """
                **Understanding Feature Impact:**
                
                - **Horizontal bars** show which factors had the most influence on your risk score
                - **Red bars (→)** indicate features that *increased* your risk
                - **Green bars (←)** indicate features that *decreased* your risk
                - **Longer bars** = stronger impact on the final risk assessment
                - **Numbers** show the exact SHAP value (impact magnitude)
                
                **Example interpretations:**
                - A long red bar for "Total Income" means higher income increased risk in your case
                - A long green bar for "Years Employed" means employment duration decreased risk
                - Features closer to the top had the most significant impact
                """
                )

        except Exception as e:
            st.error(f"Could not display feature importance: {e}")
            st.info("Debug information:")
            st.write(f"- shap_values type: {type(shap_values)}")
            st.write(f"- processed_features type: {type(processed_features)}")
            if isinstance(shap_values, list):
                st.write(
                    f"- shap_values sample: {shap_values[:3] if len(shap_values) > 3 else shap_values}"
                )
            if isinstance(processed_features, list):
                st.write(
                    f"- processed_features sample: {processed_features[:3] if len(processed_features) > 3 else processed_features}"
                )


def create_results_display() -> ResultsDisplay:
    """Factory function to create results display"""
    return ResultsDisplay()
