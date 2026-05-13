"""
Credit Risk Predictor - Streamlit App
Run: streamlit run app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ---------------- Safe launch ----------------
if __name__ == "__main__" and "STREAMLIT_RUNNING" not in os.environ:
    import subprocess
    env = os.environ.copy()
    env["STREAMLIT_RUNNING"] = "1"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.abspath(__file__),
    ], env=env)
    sys.exit()

# ---------------- Base dir ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
    return model, scaler, features

model, scaler, feature_names = load_model()

# ---------------- Options ----------------
HOME_OPTIONS = ["RENT", "OWN", "MORTGAGE", "OTHER"]

INTENT_OPTIONS = [
    "EDUCATION", "MEDICAL", "VENTURE",
    "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT",
]

GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]

HOME_MAP = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}

INTENT_MAP = {
    "DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5,
}

GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}

DEFAULT_MAP = {"No": 0, "Yes": 1}

# ---------------- UI ----------------
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("💳 Credit Risk Predictor")
st.write("Enter applicant details to predict loan risk")

# ---------------- Feature builder ----------------
def build_row(age, income, home, emp_length, intent, grade,
              loan_amnt, int_rate, pct_income, default_hist, cred_hist):

    row = [
        age, income, HOME_MAP[home], emp_length,
        INTENT_MAP[intent], GRADE_MAP[grade],
        loan_amnt, int_rate, pct_income,
        DEFAULT_MAP[default_hist], cred_hist,
    ]

    # engineered features (must match training)
    row.append(income / (loan_amnt + 1))
    row.append(age / (emp_length + 1))
    row.append((int_rate * pct_income) / (cred_hist + 1))

    return row

# ---------------- Result UI ----------------
def show_result(pred, proba):

    safe = proba[0] * 100
    risk = proba[1] * 100

    if pred == 0:
        st.markdown(f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background:linear-gradient(135deg,#00b09b,#96c93d);
            color:white;
            text-align:center;
            margin-top:20px;
        ">
            <h2>✅ LOW RISK</h2>
            <h3>{safe:.1f}% Safe Probability</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background:linear-gradient(135deg,#e53935,#ff6f00);
            color:white;
            text-align:center;
            margin-top:20px;
        ">
            <h2>⚠️ HIGH RISK</h2>
            <h3>{risk:.1f}% Default Risk</h3>
        </div>
        """, unsafe_allow_html=True)

# ---------------- Inputs (FIXED RANGES) ----------------

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=25)
    income = st.number_input("Income", min_value=5000, max_value=500000, value=50000)
    home = st.selectbox("Home Ownership", HOME_OPTIONS)
    emp_length = st.number_input("Employment Length", min_value=0, max_value=50, value=1)

with col2:
    intent = st.selectbox("Loan Purpose", INTENT_OPTIONS)
    grade = st.selectbox("Grade", GRADE_OPTIONS)
    loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=50000, value=5000)
    int_rate = st.slider("Interest Rate", min_value=5.0, max_value=25.0, value=10.0)

with col3:
    pct_income = st.slider("Loan % Income", min_value=0.0, max_value=1.0, value=0.1)
    default_hist = st.selectbox("Previous Default", ["No", "Yes"])
    cred_hist = st.slider("Credit History", min_value=2, max_value=30, value=5)

# ---------------- Predict ----------------
if st.button("🔍 Predict Risk"):

    row = build_row(
        age, income, home, emp_length,
        intent, grade, loan_amnt,
        int_rate, pct_income,
        default_hist, cred_hist
    )

    df = pd.DataFrame([row], columns=feature_names)
    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]

    show_result(pred, proba)