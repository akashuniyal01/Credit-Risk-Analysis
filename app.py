import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD MODEL AND PREPROCESSORS
# =========================

model = joblib.load('model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

# Load dataset for EDA
df = pd.read_csv("credit_risk_dataset.csv")


# =========================
# PAGE TITLE
# =========================

st.title("💳 Credit Risk Modelling")


# =========================
# TWO BUTTONS
# =========================

if "page" not in st.session_state:
    st.session_state.page = "prediction"


col1, col2 = st.columns(2)

with col1:
    if st.button("🔮 Prediction", use_container_width=True):
        st.session_state.page = "prediction"

with col2:
    if st.button("📊 Data Analysis", use_container_width=True):
        st.session_state.page = "analysis"


st.divider()


# ==========================================================
# PREDICTION PAGE
# ==========================================================

if st.session_state.page == "prediction":

    st.header("🔮 Credit Risk Prediction")

    # Q1: Age
    age = st.slider(
        "Age",
        18,
        100,
        25
    )

    # Q2: Income
    income = st.number_input(
        "Enter your personal income",
        min_value=0.0,
        value=25000.0,
        step=1000.0
    )

    # Q3: Home Ownership
    person_home_ownership = st.selectbox(
        "Person Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "Other"]
    )

    # Q4: Employment Length
    person_emp_length = st.slider(
        "Person Employment Length",
        1,
        50,
        25
    )

    # Q5: Loan Intent
    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    # Q6: Loan Grade
    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    # Q7: Loan Amount
    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=25000.0,
        step=1000.0
    )

    # Q8: Interest Rate
    loan_int_rate = st.slider(
        "Loan Interest Rate (%)",
        0.0,
        30.0,
        10.0,
        0.1
    )

    # Q9: Loan Percent Income
    loan_percent_income = st.slider(
        "Loan Percent Income",
        0.0,
        1.0,
        0.2,
        0.01
    )

    # Q10: Previous Default
    cb_person_default_on_file = st.radio(
        "Default on File?",
        ["Y", "N"]
    )

    # Q11: Credit History
    cb_person_cred_hist_length = st.slider(
        "Credit History Length (years)",
        0,
        30,
        5,
        1
    )

    # =========================
    # PREDICT BUTTON
    # =========================

    if st.button(
        "🚨 Predict Loan Status",
        use_container_width=True
    ):

        # User input
        input_data = {
            "person_age": age,
            "person_income": income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length
        }

        input_df = pd.DataFrame([input_data])

        # One-hot encoding
        input_df = pd.get_dummies(input_df)

        # Match columns expected by imputer
        input_df = input_df.reindex(
            columns=imputer.feature_names_in_,
            fill_value=0
        )

        # Apply imputer
        input_df = imputer.transform(input_df)

        input_df = pd.DataFrame(
            input_df,
            columns=imputer.feature_names_in_
        )

        # Apply scaler
        input_df = scaler.transform(input_df)

        input_df = pd.DataFrame(
            input_df,
            columns=scaler.feature_names_in_
        )

        # Remove loan_status if present
        if "loan_status" in input_df.columns:
            input_df = input_df.drop(
                columns=["loan_status"]
            )

        # Prediction
        prediction = model.predict(input_df)

        # Result
        if prediction[0] == 1:

            st.error("⚠️ High Credit Risk")

            st.write(
                "The applicant is likely to default on the loan."
            )

        else:

            st.success("✅ Low Credit Risk")

            st.write(
                "The applicant is unlikely to default on the loan."
            )


# ==========================================================
# DATA ANALYSIS PAGE
# ==========================================================

elif st.session_state.page == "analysis":

    st.header("📊 Data Analysis")

    st.write(
        "Explore the credit risk dataset using the following visualizations."
    )

    # -------------------------
    # Dataset overview
    # -------------------------

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Total Features",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )

    # -------------------------
    # Loan Status
    # -------------------------

    st.subheader("Loan Status Distribution")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="loan_status",
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Age
    # -------------------------

    st.subheader("Age Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x="person_age",
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Income
    # -------------------------

    st.subheader("Income Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x="person_income",
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Loan Amount
    # -------------------------

    st.subheader("Loan Amount Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x="loan_amnt",
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Loan Grade vs Status
    # -------------------------

    st.subheader("Loan Grade vs Loan Status")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="loan_grade",
        hue="loan_status",
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Home Ownership
    # -------------------------

    st.subheader("Home Ownership vs Loan Status")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="person_home_ownership",
        hue="loan_status",
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Loan Intent
    # -------------------------

    st.subheader("Loan Intent vs Loan Status")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x="loan_intent",
        hue="loan_status",
        ax=ax
    )

    plt.xticks(rotation=30)

    st.pyplot(fig)

    # -------------------------
    # Correlation
    # -------------------------

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)
