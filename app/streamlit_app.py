import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_xgboost.joblib"
ENCODER_PATH = BASE_DIR / "models" / "agegroup_encoder.joblib"

# ---------------------------------------------------------
# Load model and encoder
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

model, encoder = load_artifacts()

# Locked threshold selected during validation
THRESHOLD = 0.45

# Exact feature order used during model training
CATEGORICAL_COLS = ["AgeGroup"]

# This reproduces the same ordering used in the notebooks:
# X.columns.difference(["AgeGroup"]).tolist()
NUMERIC_COLS = [
    "DebtRatio",
    "Has90DaysLate",
    "HasAnyDelinquency",
    "HighUtilization",
    "IncomeMissing",
    "IncomePerCreditLine",
    "MonthlyIncome",
    "NumberOfDependents",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "RevolvingUtilizationOfUnsecuredLines",
    "TotalCreditLines",
    "TotalDelinquencies",
    "age",
]

AGE_GROUPS = ["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81+"]

# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------
def build_features(
    utilization,
    age,
    days_30_59,
    debt_ratio,
    monthly_income,
    open_credit_lines,
    days_90,
    real_estate_loans,
    days_60_89,
    dependents,
    age_group,
):
    # Original + engineered features exactly as used in the project
    high_utilization = int(utilization > 1)

    total_delinquencies = (
        days_30_59 +
        days_60_89 +
        days_90
    )

    total_credit_lines = (
        open_credit_lines +
        real_estate_loans
    )

    has_90_days_late = int(days_90 > 0)

    has_any_delinquency = int(total_delinquencies > 0)

    # The model was trained after median imputation.
    # The UI requires income, so IncomeMissing is 0 here.
    income_missing = 0

    income_per_credit_line = (
        monthly_income /
        max(total_credit_lines, 1)
    )

    data = {
        "DebtRatio": debt_ratio,
        "Has90DaysLate": has_90_days_late,
        "HasAnyDelinquency": has_any_delinquency,
        "HighUtilization": high_utilization,
        "IncomeMissing": income_missing,
        "IncomePerCreditLine": income_per_credit_line,
        "MonthlyIncome": monthly_income,
        "NumberOfDependents": dependents,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTime30-59DaysPastDueNotWorse": days_30_59,
        "NumberOfTime60-89DaysPastDueNotWorse": days_60_89,
        "NumberOfTimes90DaysLate": days_90,
        "NumberOfRealEstateLoansOrLines": real_estate_loans,
        "RevolvingUtilizationOfUnsecuredLines": utilization,
        "TotalCreditLines": total_credit_lines,
        "TotalDelinquencies": total_delinquencies,
        "age": age,
        "AgeGroup": age_group,
    }

    # Correct the exact project column name
    data["NumberRealEstateLoansOrLines"] = data.pop(
        "NumberOfRealEstateLoansOrLines"
    )

    df = pd.DataFrame([data])

    # Numeric features
    X_num = df[NUMERIC_COLS].to_numpy()

    # AgeGroup one-hot encoding using the fitted encoder
    X_cat = encoder.transform(df[CATEGORICAL_COLS])

    # Exact same ordering as training
    X_final = np.hstack([X_num, X_cat])

    return X_final


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("💳 Credit Default Prediction")
st.markdown(
    "### XGBoost-based credit risk assessment"
)
st.write(
    "Enter the customer's financial and credit information "
    "to estimate the probability of serious delinquency within two years."
)

st.divider()

# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=21,
        max_value=109,
        value=45,
        step=1,
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=5000.0,
        step=100.0,
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

    age_group = st.selectbox(
        "Age Group",
        AGE_GROUPS,
        index=2,
    )

with col2:
    utilization = st.number_input(
        "Revolving Utilization of Unsecured Lines",
        min_value=0.0,
        value=0.20,
        step=0.01,
        format="%.4f",
        help="Ratio of total balance on unsecured credit lines to available credit.",
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        value=0.40,
        step=0.01,
        format="%.4f",
    )

    open_credit_lines = st.number_input(
        "Open Credit Lines and Loans",
        min_value=0,
        max_value=100,
        value=5,
        step=1,
    )

with col3:
    real_estate_loans = st.number_input(
        "Real Estate Loans or Lines",
        min_value=0,
        max_value=100,
        value=1,
        step=1,
    )

    days_30_59 = st.number_input(
        "Times 30–59 Days Past Due",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
    )

    days_60_89 = st.number_input(
        "Times 60–89 Days Past Due",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
    )

    days_90 = st.number_input(
        "Times 90+ Days Late",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
    )

st.divider()

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("🔍 Predict Credit Risk", type="primary", use_container_width=True):

    # Make sure AgeGroup matches the selected age
    if 21 <= age <= 30:
        age_group = "21-30"
    elif 31 <= age <= 40:
        age_group = "31-40"
    elif 41 <= age <= 50:
        age_group = "41-50"
    elif 51 <= age <= 60:
        age_group = "51-60"
    elif 61 <= age <= 70:
        age_group = "61-70"
    elif 71 <= age <= 80:
        age_group = "71-80"
    else:
        age_group = "81+"

    X_input = build_features(
        utilization=utilization,
        age=age,
        days_30_59=days_30_59,
        debt_ratio=debt_ratio,
        monthly_income=monthly_income,
        open_credit_lines=open_credit_lines,
        days_90=days_90,
        real_estate_loans=real_estate_loans,
        days_60_89=days_60_89,
        dependents=dependents,
        age_group=age_group,
    )

    probability = float(model.predict_proba(X_input)[0, 1])
    prediction = int(probability >= THRESHOLD)

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Default Probability",
            f"{probability:.2%}",
        )

    with result_col2:
        st.metric(
            "Decision Threshold",
            f"{THRESHOLD:.2f}",
        )

    with result_col3:
        if prediction == 1:
            st.error("⚠️ Higher Default Risk")
        else:
            st.success("✅ Lower Default Risk")

    if prediction == 1:
        st.warning(
            "The model predicts a higher likelihood of serious delinquency "
            "within two years."
        )
    else:
        st.success(
            "The model predicts a lower likelihood of serious delinquency "
            "within two years."
        )

    # Risk indicator
    st.progress(
        min(max(probability, 0.0), 1.0),
        text=f"Predicted probability: {probability:.2%}",
    )

    st.caption(
        "This prediction is generated by the trained XGBoost model using "
        "the project's selected classification threshold of 0.45."
    )

# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------
with st.expander("Model Information"):
    st.write("**Model:** XGBoost")
    st.write("**Primary optimization metric:** F1")
    st.write("**Classification threshold:** 0.45")
    st.write("**Test F1:** 0.4332")
    st.write("**Test ROC-AUC:** 0.8622")
    st.info(
        "This application is a machine-learning demonstration and should "
        "not be used as the sole basis for real credit decisions."
    )
