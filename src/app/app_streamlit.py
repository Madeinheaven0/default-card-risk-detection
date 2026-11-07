import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Default Risk Prediction")
st.markdown(
    "Fill in the client credit profile details below to evaluate the default risk in real-time."
)


# ==========================================
# MODEL & PIPELINE LOADING (Cached)
# ==========================================
@st.cache_resource
def load_credit_model():
    """Loads the trained pipeline (preprocessor + XGBoost) from local joblib."""
    # Adjust path according to your project structure
    model_path = Path("models/xgbclassifier_best_model.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None


model = load_credit_model()

if model is None:
    st.warning(
        "⚠️ Local model file not found (`models/xgbclassifier_best_model.joblib`). Make sure you have run the training script.")

# Input Form
with st.form("prediction_form"):
    st.subheader("📊 Credit Parameters and History")

    col1, col2 = st.columns(2)

    with col1:
        pre_since_opened = st.number_input("Months since account opened (pre_since_opened)", min_value=0, value=12)
        pre_since_confirmed = st.number_input("Months since confirmation (pre_since_confirmed)", min_value=0, value=4)
        pre_loans_credit_limit = st.number_input("Credit limit (pre_loans_credit_limit)", min_value=0.0, value=50000.0)
        pre_loans_credit_cost_rate = st.number_input("Credit cost rate", min_value=0.0, value=4.0)
        pre_loans_outstanding = st.number_input("Outstanding loans amount", min_value=0.0, value=10000.0)
        paym_mean_status = st.number_input("Mean payment status", value=0.0)
        paym_count_incidents = st.number_input("Count of payment incidents", min_value=0, value=1)
        paym_incidents_over_mean = st.number_input("Incidents over mean ratio", value=1.0)
        total_zero_loans_count = st.number_input("Total zero loans count", min_value=0, value=5)
        paym_trend_3m_vs_12m = st.number_input("Payment trend (3m vs 12m)", value=0.0)
        paym_hist_mean_1_23 = st.number_input("Historical mean (months 1-23)", value=0.0)

    with col2:
        total_delinquency_score = st.number_input("Total delinquency score", min_value=0, value=3)
        count_paym_status_0_24m = st.number_input("Count of status 0 (24m)", min_value=0, value=20)
        count_paym_status_2_24m = st.number_input("Count of status 2 (24m)", min_value=0, value=0)
        count_paym_status_3_24m = st.number_input("Count of status 3 (24m)", min_value=0, value=3)
        enc_loans_account_holder_type = st.selectbox("Account holder type", [1, 2, 3], index=0)
        enc_loans_credit_status = st.selectbox("Credit status", [1, 2, 3], index=2)
        enc_loans_credit_type = st.selectbox("Credit type", [1, 2, 3, 4], index=3)
        enc_loans_account_cur = st.selectbox("Account currency", [1, 2], index=0)
        pclose_flag = st.selectbox("Close flag (pclose_flag)", [0, 1], index=0)
        paym_latest_status = st.number_input("Latest payment status", value=0)
        paym_max_status = st.number_input("Max payment status", value=3)

    submit_button = st.form_submit_button(label="Run Risk Prediction 🚀")

if submit_button:
    if model is None:
        st.error("Cannot run prediction: Model is not loaded.")
    else:
        # Build input DataFrame
        input_data = pd.DataFrame([{
            "pre_since_opened": pre_since_opened,
            "pre_since_confirmed": pre_since_confirmed,
            "pre_loans_credit_limit": pre_loans_credit_limit,
            "pre_loans_credit_cost_rate": pre_loans_credit_cost_rate,
            "paym_mean_status": paym_mean_status,
            "paym_count_incidents": paym_count_incidents,
            "paym_incidents_over_mean": paym_incidents_over_mean,
            "total_zero_loans_count": total_zero_loans_count,
            "paym_trend_3m_vs_12m": paym_trend_3m_vs_12m,
            "paym_hist_mean_1_23": paym_hist_mean_1_23,
            "total_delinquency_score": total_delinquency_score,
            "count_paym_status_0_24m": count_paym_status_0_24m,
            "count_paym_status_2_24m": count_paym_status_2_24m,
            "count_paym_status_3_24m": count_paym_status_3_24m,
            "enc_loans_account_holder_type": enc_loans_account_holder_type,
            "enc_loans_credit_status": enc_loans_credit_status,
            "enc_loans_credit_type": enc_loans_credit_type,
            "enc_loans_account_cur": enc_loans_account_cur,
            "pclose_flag": pclose_flag,
            "paym_latest_status": paym_latest_status,
            "paym_max_status": paym_max_status,
            "pre_loans_outstanding": pre_loans_outstanding
        }])

        # Compute derived features required by the pipeline (same as add_credit_financial_ratios_pandas)
        input_data['credit_utilization_ratio'] = input_data['pre_loans_outstanding'] / (
                    input_data['pre_loans_credit_limit'] + 1e-5)
        input_data['delinquency_per_incident'] = input_data['total_delinquency_score'] / (
                    input_data['paym_count_incidents'] + 1)
        input_data['status_trend_diff'] = input_data['paym_latest_status'] - input_data['paym_hist_mean_1_23']
        input_data['severe_incidents_ratio'] = input_data['count_paym_status_3_24m'] / (
                    input_data['paym_count_incidents'] + 1)
        input_data['credit_stress_score'] = input_data['credit_utilization_ratio'] * (
                    input_data['paym_count_incidents'] + 1)
        input_data['delinquency_velocity'] = input_data['paym_trend_3m_vs_12m'] * (input_data['paym_latest_status'] + 1)
        input_data['maturity_risk_ratio'] = input_data['pre_loans_outstanding'] / (input_data['pre_since_opened'] + 1)
        input_data['zero_loans_cleanliness_index'] = float(total_zero_loans_count)

        try:
            with st.spinner("Analyzing risk profile..."):
                # Prediction probability for default (class 1)
                prob = float(model.predict_proba(input_data)[0][1])

                # Define risk level based on threshold (e.g. 0.5 or optimal threshold)
                risk_level = "High Risk" if prob >= 0.5 else "Low Risk"

            st.success("Prediction completed successfully!")

            # Display metrics
            col_res1, col_res2 = st.columns(2)
            col_res1.metric(label="Default Probability", value=f"{prob * 100:.2f} %")
            col_res2.metric(label="Risk Level", value=risk_level)

            # Visual progress bar for probability
            st.progress(prob)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")