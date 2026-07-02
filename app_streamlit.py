import json
import os

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

CLASS_NAMES = {0: "Poor", 1: "Standard", 2: "Good"}
LABEL_COLORS = {"Poor": "#FF4B4B", "Standard": "#FFA500", "Good": "#21BA45"}

LOAN_TYPES = [
    "Student Loan", "Mortgage Loan", "Debt Consolidation Loan",
    "Payday Loan", "Credit-Builder Loan", "Personal Loan",
    "Home Equity Loan", "Auto Loan", "Not Specified",
]

OCCUPATIONS = [
    "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
    "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
    "Accountant", "Musician", "Mechanic", "Writer", "Architect",
]

PAYMENT_BEHAVIOURS = [
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments",
    "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
]


class SageMakerPredictor:

    def __init__(self):
        self.endpoint_name = ENDPOINT_NAME
        self.region = REGION
        self.runtime = boto3.client("sagemaker-runtime", region_name=self.region)

    def predict(self, features):
        payload = {"instances": [features]}

        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )

        return json.loads(response["Body"].read().decode("utf-8"))


@st.cache_resource
def load_predictor():
    return SageMakerPredictor()


predictor = load_predictor()

st.set_page_config(page_title="Credit Score Predictor", layout="wide")

st.title("Credit Score Prediction")
st.caption("Classes: Poor · Standard · Good")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Customer Data Input")

    with st.form("input_form"):

        st.subheader("Personal & Financial Info")

        age = st.number_input("Age", 18, 100, 35)
        occupation = st.selectbox("Occupation", OCCUPATIONS)
        annual_income = st.number_input("Annual Income (USD)", 0.0, 500_000.0, 50_000.0, step=1000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary (USD)", 0.0, 50_000.0, 4_000.0, step=100.0)

        st.subheader("Credit Accounts")

        num_bank_accounts = st.slider("Number of Bank Accounts", 0, 20, 3)
        num_credit_cards = st.slider("Number of Credit Cards", 0, 20, 2)
        num_of_loan = st.slider("Number of Loans", 0, 20, 2)
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])

        st.subheader("Payment Behaviour")

        interest_rate = st.slider("Interest Rate (%)", 0, 60, 15)
        delay_from_due = st.slider("Avg. Delay from Due Date (days)", 0, 100, 5)
        num_delayed_payments = st.slider("Number of Delayed Payments", 0, 50, 3)
        payment_of_min = st.selectbox("Pays Minimum Amount?", ["Yes", "No"])
        payment_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOURS)

        st.subheader("Debt & Utilisation")

        outstanding_debt = st.number_input("Outstanding Debt (USD)", 0.0, 100_000.0, 1_000.0, step=100.0)
        changed_credit_limit = st.number_input("Changed Credit Limit (USD)", -10_000.0, 50_000.0, 0.0, step=100.0)
        num_credit_inquiries = st.slider("Number of Credit Inquiries", 0, 20, 2)

        st.subheader("History & EMI")

        credit_history_months = st.slider("Credit History Age (months)", 0, 500, 120)
        total_emi = st.number_input("Total EMI per Month (USD)", 0.0, 10_000.0, 300.0, step=50.0)
        amount_invested = st.number_input("Amount Invested Monthly (USD)", 0.0, 10_000.0, 200.0, step=50.0)
        monthly_balance = st.number_input("Monthly Balance (USD)", 0.0, 50_000.0, 500.0, step=50.0)

        st.subheader("Loan Types Held")

        selected_loans = st.multiselect("Select all that apply", LOAN_TYPES, default=[])

        submitted = st.form_submit_button("Predict Credit Score", use_container_width=True)

if submitted:

    loan_values = [1 if lt in selected_loans else 0 for lt in LOAN_TYPES]

    features = [
        age, annual_income, monthly_salary,
        num_bank_accounts, num_credit_cards, interest_rate,
        num_of_loan, delay_from_due, num_delayed_payments,
        changed_credit_limit, num_credit_inquiries, outstanding_debt,
        total_emi, credit_history_months, amount_invested, monthly_balance,
        payment_of_min,
        *loan_values,
        credit_mix,
        occupation,
        payment_behaviour,
    ]

    try:
        result = predictor.predict(features)

    except NoCredentialsError:
        st.error("AWS credentials tidak ditemukan.")
        st.stop()

    except ClientError as e:
        st.error(f"AWS Error: {e.response['Error'].get('Message', str(e))}")
        st.stop()

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

    label_idx = int(result["predictions"][0])
    label = CLASS_NAMES[label_idx]
    color = LABEL_COLORS.get(label, "#888")

    st.session_state.history.append({
        "Age": age,
        "Occupation": occupation,
        "Credit Mix": credit_mix,
        "Outstanding Debt": outstanding_debt,
        "Prediction": label,
        "Loan Types": ", ".join(selected_loans) if selected_loans else "-",
    })

    st.markdown(
        f"""
        <div style='background:{color}22; border:2px solid {color};
                    border-radius:12px; padding:20px; text-align:center; margin-bottom:20px'>
            <h1 style='color:{color}; margin:0'>Credit Score: {label}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        
        st.subheader("Input Summary")

        summary = {
            "Age": age,
            "Occupation": occupation,
            "Annual Income": f"${annual_income:,.0f}",
            "Monthly Salary": f"${monthly_salary:,.0f}",
            "Credit Mix": credit_mix,
            "Outstanding Debt": f"${outstanding_debt:,.0f}",
            "Credit History": f"{credit_history_months} months",
            "Delayed Payments": num_delayed_payments,
        }

        summary_df = pd.DataFrame(summary.items(), columns=["Feature", "Value"]).astype(str)
        st.table(summary_df)

    with col2:
        
        if "probabilities" in result:
            st.subheader("Prediction Probability")

            proba = result["probabilities"][0]
            prob_df = pd.DataFrame({"Probability (%)": [p * 100 for p in proba]}, index=["Poor", "Standard", "Good"],)
            
            st.bar_chart(prob_df)

        else:
            st.info("Probability scores not available.")