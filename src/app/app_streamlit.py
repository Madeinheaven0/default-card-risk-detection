import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Default Risk Prediction")
st.markdown(
    "Fill in the client credit profile details below to evaluate the default risk in real-time via the FastAPI backend.")

# FastAPI endpoint URL
API_URL = "http://api:8000/predict/"

with st.form("prediction_form"):
    st.subheader("📊 Credit Parameters and History")

    col1, col2 = st.columns(2)

    with col1:
        pre_since_opened = st.number_input("Months since account opened (pre_since_opened)", min_value=0, value=0)
        pre_since_confirmed = st.number_input("Months since confirmation (pre_since_confirmed)", min_value=0, value=4)
        pre_loans_credit_limit = st.number_input("Credit limit (pre_loans_credit_limit)", min_value=0.0, value=11.0)
        pre_loans_credit_cost_rate = st.number_input("Credit cost rate", min_value=0.0, value=4.0)
        pre_loans_outstanding = st.number_input("Outstanding loans amount", min_value=0.0, value=3.0)
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
    # Build payload mapping to Pydantic schema
    payload = {
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
    }

    try:
        with st.spinner("Analyzing risk profile..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            prob = result.get("default_probability", 0.0)
            risk_level = result.get("risk_level", "Unknown")

            st.success("Prediction completed successfully!")

            # Display metrics
            col_res1, col_res2 = st.columns(2)
            col_res1.metric(label="Default Probability", value=f"{prob * 100:.2f} %")
            col_res2.metric(label="Risk Level", value=risk_level)

            # Visual progress bar for probability
            st.progress(float(prob))

        else:
            st.error(f"API Error ({response.status_code}): {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI server. Make sure the server is running.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")