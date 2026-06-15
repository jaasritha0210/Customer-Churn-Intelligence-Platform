import streamlit as st
import pandas as pd
import joblib
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
st.header("Customer Information")

gender = st.selectbox("Gender", ["Male", "Female"])

senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

partner = st.selectbox("Partner", ["No", "Yes"])

dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure_months = st.number_input(
    "Tenure Months",
    min_value=0,
    max_value=72,
    value=12
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Mailed check",
        "Electronic check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

# Feature Engineering

total_services = (
    (1 if online_security == "Yes" else 0)
    + (1 if online_backup == "Yes" else 0)
    + (1 if device_protection == "Yes" else 0)
    + (1 if tech_support == "Yes" else 0)
    + (1 if streaming_tv == "Yes" else 0)
    + (1 if streaming_movies == "Yes" else 0)
)

security_services_count = (
    (1 if online_security == "Yes" else 0)
    + (1 if online_backup == "Yes" else 0)
    + (1 if device_protection == "Yes" else 0)
    + (1 if tech_support == "Yes" else 0)
)

streaming_services_count = (
    (1 if streaming_tv == "Yes" else 0)
    + (1 if streaming_movies == "Yes" else 0)
)

if tenure_months <= 12:
    tenure_group = "0-12"
elif tenure_months <= 24:
    tenure_group = "12-24"
elif tenure_months <= 48:
    tenure_group = "24-48"
else:
    tenure_group = "48-72"

avg_revenue_per_month = total_charges / (tenure_months + 1)

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        'Gender': [gender],
        'Senior Citizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'Tenure Months': [tenure_months],
        'Phone Service': [phone_service],
        'Multiple Lines': [multiple_lines],
        'Internet Service': [internet_service],
        'Online Security': [online_security],
        'Online Backup': [online_backup],
        'Device Protection': [device_protection],
        'Tech Support': [tech_support],
        'Streaming TV': [streaming_tv],
        'Streaming Movies': [streaming_movies],
        'Contract': [contract],
        'Paperless Billing': [paperless_billing],
        'Payment Method': [payment_method],
        'Monthly Charges': [monthly_charges],
        'Total Charges': [total_charges],
        'Total Services': [total_services],
        'Security Services Count': [security_services_count],
        'Streaming Services Count': [streaming_services_count],
        'Tenure Group': [tenure_group],
        'Avg Revenue Per Month': [avg_revenue_per_month]
    })
    st.write(input_data.columns)
    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")

    st.write(f"Churn Probability: {probability:.2%}")
