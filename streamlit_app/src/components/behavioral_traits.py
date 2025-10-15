import streamlit as st
from typing import Dict, Any


class BehavioralTraitsDisplay:
    """
    Display behavioral traits analysis results with visualizations
    """

    def __init__(self):
        self.trait_colors = {
            "job_stability": "#1f77b4",
            "payment_behavior": "#ff7f0e", 
            "responsibility": "#2ca02c"
        }
        
        self.trait_icons = {
            "job_stability": "📊",
            "payment_behavior": "💳",
            "responsibility": "📞"
        }
        
        self.trait_descriptions = {
            "job_stability": "Employment consistency and income reliability",
            "payment_behavior": "Credit usage and payment patterns",
            "responsibility": "Demographic indicators of reliability"
        }

    def display_behavioral_traits(self, traits_results: Dict[str, Any]):
        """
        Display comprehensive behavioral traits analysis
        
        Args:
            traits_results: Dictionary containing behavioral traits scores
        """
        if "error" in traits_results:
            st.error(f"Behavioral Analysis Error: {traits_results['error']}")
            return

        st.markdown("---")
        st.title("🎭 Behavioral Traits Analysis")

        job_stability = traits_results.get("job_stability", 0)
        payment_behavior = traits_results.get("payment_behavior", 0)
        responsibility = traits_results.get("responsibility", 0)
        overall_score = traits_results.get("overall_behavioral_score", 0)
        
        # Overall Score Display
        self._display_overall_score(overall_score)
        
        # Individual Traits Display
        self._display_individual_traits(job_stability, payment_behavior, responsibility)
        
        # Behavioral Insights
        self._display_behavioral_insights(traits_results)

    def _display_overall_score(self, overall_score: float):
        """Display overall behavioral score with interpretation"""
        
        if overall_score >= 70:
            category = "Excellent"
            color = "#2ca02c"
            emoji = "✅"
        elif overall_score >= 50:
            category = "Good" 
            color = "#ff7f0e"
            emoji = "🟡"
        elif overall_score >= 30:
            category = "Average"
            color = "#d62728"
            emoji = "🟠"
        else:
            category = "Below Average"
            color = "#8c564b"
            emoji = "🔴"
        
        st.markdown("### Overall Behavioral Score")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, {color}22 0%, {color}44 100%);
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                border: 2px solid {color};
                margin: 1rem 0;
            ">
                <h1 style="color: {color}; margin: 0; font-size: 3rem;">{overall_score:.1f}</h1>
                <h3 style="color: {color}; margin: 0.5rem 0;">{emoji} {category}</h3>
                <p style="color: #666; margin: 0;">Behavioral Profile Score</p>
            </div>
            """, unsafe_allow_html=True)

    def _display_individual_traits(self, job_stability: float, payment_behavior: float, responsibility: float):
        """Display individual trait scores"""
        
        st.markdown("### Individual Trait Scores")
        
        traits_data = [
            ("job_stability", "Job Stability", job_stability),
            ("payment_behavior", "Payment Behavior", payment_behavior), 
            ("responsibility", "Responsibility", responsibility)
        ]
        
        cols = st.columns(3)
        
        for i, (trait_key, trait_name, score) in enumerate(traits_data):
            with cols[i]:
                color = self.trait_colors[trait_key]
                icon = self.trait_icons[trait_key]
                description = self.trait_descriptions[trait_key]
                
                if score >= 60:
                    status = "Strong"
                    status_color = "#2ca02c"
                elif score >= 40:
                    status = "Moderate"
                    status_color = "#ff7f0e"
                else:
                    status = "Needs Attention"
                    status_color = "#d62728"
                
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 1.5rem;
                    border-radius: 10px;
                    border-left: 5px solid {color};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 1rem;
                ">
                    <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{icon} {trait_name}</h4>
                    <h2 style="color: {color}; margin: 0.5rem 0;">{score:.1f}/100</h2>
                    <p style="color: {status_color}; font-weight: bold; margin: 0.5rem 0;">{status}</p>
                    <p style="color: #666; font-size: 0.9rem; margin: 0;">{description}</p>
                </div>
                """, unsafe_allow_html=True)

    def _display_behavioral_insights(self, traits_results: Dict[str, Any]):
        """Display behavioral insights and recommendations"""
        
        st.markdown("### 🔍 Behavioral Insights")
        
        job_stability = traits_results.get("job_stability", 0)
        payment_behavior = traits_results.get("payment_behavior", 0)
        responsibility = traits_results.get("responsibility", 0)
        overall_score = traits_results.get("overall_behavioral_score", 0)
        
        insights = []
        recommendations = []
        
        if job_stability >= 60:
            insights.append("✅ Strong employment stability indicates reliable income patterns")
        else:
            insights.append("⚠️ Employment stability shows room for improvement")
            recommendations.append("Consider building longer employment history or stable income sources")
        
        if payment_behavior >= 60:
            insights.append("✅ Positive payment behavior patterns suggest good credit management")
        else:
            insights.append("⚠️ Payment behavior indicates potential credit risk factors")
            recommendations.append("Focus on consistent payment history and credit utilization management")
        
        if responsibility >= 60:
            insights.append("✅ High responsibility indicators suggest reliable financial behavior")
        else:
            insights.append("⚠️ Responsibility metrics show areas for strengthening")
            recommendations.append("Building assets and stable family/housing situations can improve this score")
        
        if overall_score >= 70:
            insights.append("🎯 Overall profile suggests excellent behavioral traits for lending")
        elif overall_score >= 50:
            insights.append("🎯 Overall profile shows good behavioral traits with minor areas for improvement")
        else:
            insights.append("🎯 Overall profile indicates significant opportunity for behavioral improvement")
        
        if insights:
            st.markdown("**Key Insights:**")
            for insight in insights:
                st.markdown(f"- {insight}")
        
        if recommendations:
            st.markdown("**Recommendations:**")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        st.markdown("---")
        st.info("""
        **About Behavioral Traits Analysis:**
        
        This analysis uses machine learning models trained on 300,000+ credit applications to predict behavioral patterns that correlate with positive financial outcomes. Unlike traditional credit scoring, this approach evaluates:
        
        - **Job Stability**: Employment consistency and income reliability patterns
        - **Payment Behavior**: Credit usage and payment history indicators  
        - **Responsibility**: Demographic factors associated with financial reliability
        
        Higher scores indicate behavioral traits associated with lower default risk and better financial outcomes.
        """)


def create_behavioral_traits_display():
    """Factory function to create behavioral traits display component"""
    return BehavioralTraitsDisplay()