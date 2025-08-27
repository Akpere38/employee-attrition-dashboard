# model.py
from shap import Explainer
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

class LeavePredictor:
    def __init__(self, model_path="models/lgbm_model.pkl"):
        self.model = joblib.load(model_path)

    def render(self):
        st.subheader("🤖Employee Attrition Prediction (LightGBM)")

        # --- User Inputs ---
        joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2018)
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ever_benched = st.selectbox("Ever Benched", ["No", "Yes"])
        exp_current_domain = st.slider("Experience in Current Domain (years)", 0, 20, 3)

        education = st.selectbox("Education", ["Bachelors", "Masters", "PHD"])
        city = st.selectbox("City", ["Bangalore", "New Delhi", "Pune"])
        payment_tier = st.selectbox("Payment Tier", [1, 2, 3])

        # --- Encoding ---
        input_dict = {
            "JoiningYear": joining_year,
            "Age": age,
            "Gender": 1 if gender == "Male" else 0,
            "EverBenched": 1 if ever_benched == "Yes" else 0,
            "ExperienceInCurrentDomain": exp_current_domain,
            "Education_Bachelors": 1 if education == "Bachelors" else 0,
            "Education_Masters": 1 if education == "Masters" else 0,
            "Education_PHD": 1 if education == "PHD" else 0,
            "City_Bangalore": 1 if city == "Bangalore" else 0,
            "City_New Delhi": 1 if city == "New Delhi" else 0,
            "City_Pune": 1 if city == "Pune" else 0,
            "PaymentTier_1": 1 if payment_tier == 1 else 0,
            "PaymentTier_2": 1 if payment_tier == 2 else 0,
            "PaymentTier_3": 1 if payment_tier == 3 else 0,
        }

        input_df = pd.DataFrame([input_dict])

        # --- Prediction ---
        if st.button("Predict"):
            proba = self.model.predict_proba(input_df)[0][1]  # Probability of Leave
            pred = self.model.predict(input_df)[0]

            st.write(f"### Prediction: {'🚨 Leave' if pred == 1 else '✅ Stay'}")
            st.write(f"**Probability of Leaving:** {proba*100:.2f}%")

            # --- Gauge / Meter ---
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={'text': "Attrition Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if proba > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "lightcoral"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Display shap explainability 
            import shap
            import matplotlib.pyplot as plt

            X_background = pd.read_csv("data/X_train.csv")  
            # Optionally sample to make it smaller
            # X_background = X_background.sample(100, random_state=42)

            # Create explainer
            explainer = shap.Explainer(self.model, X_background)

            # Get SHAP values for current input
            shap_values_current = explainer.shap_values(input_df)

            # Binary vs multiclass
            if isinstance(shap_values_current, list):
                shap_vals = shap_values_current[1]       # positive class
                expected_val = explainer.expected_value[1]
            else:
                shap_vals = shap_values_current
                expected_val = explainer.expected_value

            # # Visualize SHAP values
            # fig, ax = plt.subplots()
            # shap.summary_plot(shap_vals, ax=ax)
            # st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8,6))
            shap.plots._waterfall.waterfall_legacy(
                expected_val,
                shap_vals[0],           # first (and only) row
                feature_names=input_df.columns,
                features=input_df.iloc[0],
                show=False
            )

            st.pyplot(fig)
            st.write("This chart explains how each feature contributed to the predicted probability for your input.")

            st.write("This chart explains how each feature contributed to the predicted probability for your input.")

           

        # Model evaluation
        import matplotlib.pyplot as plt
        st.subheader("LightGBM Model Evaluation Metrics")

        # Use columns to display metrics side by side
        col1, col2, col3 = st.columns(3)
        col1.metric("ROC-AUC", "0.844")
        col2.metric("PR-AUC", "0.839")
        col3.metric("F1-score (Leave=1)", "0.728")

        col4, col5, col6 = st.columns(3)
        col4.metric("Brier Score", "0.149")
        col5.metric("Accuracy", "0.79")
        col6.metric("Recall (Leave=1)", "0.72")

        plt.savefig("confusion_matrix.png")
        st.image("lgbm_cm.png", caption="Confusion Matrix")

