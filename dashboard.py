# dashboard.py (Corrected Final Version)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Load Model and Columns---
try:
    model = joblib.load('churn_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model or column file not found. Please run the notebook to train and save them first.")
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

# --- Page Configuration ---
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# --- Load Data ---
df = load_data()

# --- Sidebar ---
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Data Analysis & KPIs", "Live Churn Prediction"])

# --- Page 1: Project Overview ---
if page == "Project Overview":
    st.title("Project: Customer Churn Prediction using Machine Learning")
    st.header("Software Requirements Document (SRD) Summary")
    st.markdown("...") # Content is the same as before

# --- Page 2: Data Analysis & KPIs ---
elif page == "Data Analysis & KPIs":
    st.title("Data Analysis and Key Performance Indicators (KPIs)")
    # Content is the same as before
    st.header("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    total_customers = df.shape[0]
    churned_customers = df[df['Churn'] == 'Yes'].shape[0]
    churn_rate = (churned_customers / total_customers) * 100
    col1.metric("Total Customers", f"{total_customers}")
    col2.metric("Churned Customers", f"{churned_customers}")
    col3.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
    st.markdown("---")
    st.header("Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tenure vs. Churn")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette={'Yes': '#FF5733', 'No': '#33C1FF'}, ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Monthly Charges vs. Churn")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', palette={'Yes': '#FF5733', 'No': '#33C1FF'}, ax=ax)
        st.pyplot(fig)

# --- Page 3: Live Churn Prediction ---
elif page == "Live Churn Prediction":
    st.title("Live Customer Churn Prediction")
    st.write("Enter the customer's details below to get a churn prediction.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            # We'll collect a smaller, more manageable set of important features from the user
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=10.0)

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            dependents = st.radio("Has Dependents?", ["Yes", "No"])
            partner = st.radio("Has a Partner?", ["Yes", "No"])

        submit_button = st.form_submit_button(label='Predict Churn')

    if submit_button:
        # Create a dictionary of the one-hot encoded features based on user input
        input_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges, # Added TotalCharges
            # We assume some defaults for features not in the form for simplicity
            'SeniorCitizen': 0,
            'gender_Male': 0,
            'PhoneService_Yes': 1,
            'MultipleLines_No phone service': 0,
            'MultipleLines_Yes': 0,
            'OnlineBackup_No internet service': 0,
            'OnlineBackup_Yes': 0,
            'DeviceProtection_No internet service': 0,
            'DeviceProtection_Yes': 0,
            'StreamingTV_No internet service': 0,
            'StreamingTV_Yes': 0,
            'StreamingMovies_No internet service': 0,
            'StreamingMovies_Yes': 0,
            'PaymentMethod_Credit card (automatic)': 0,
            'PaymentMethod_Electronic check': 0,
            'PaymentMethod_Mailed check': 0,
            # Set values based on user input
            'Partner_Yes': 1 if partner == 'Yes' else 0,
            'Dependents_Yes': 1 if dependents == 'Yes' else 0,
            'PaperlessBilling_Yes': 1 if paperless_billing == 'Yes' else 0,
            'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
            'InternetService_No': 1 if internet_service == 'No' else 0,
            'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
            'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
            'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
            'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
            'Contract_One year': 1 if contract == 'One year' else 0,
            'Contract_Two year': 1 if contract == 'Two year' else 0,
        }

        # Create a dataframe from the dictionary
        input_df = pd.DataFrame([input_data])

        # CRITICAL STEP: Reindex the dataframe to match the model's training columns
        # This adds any missing columns with a value of 0 and ensures the order is identical.
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"This customer is likely to CHURN. (Confidence: {prediction_proba[0][1]*100:.2f}%)")
        else:
            st.success(f"This customer is likely to NOT CHURN. (Confidence: {prediction_proba[0][0]*100:.2f}%)")
