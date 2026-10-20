import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered",
)

# ---------- Light custom styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 820px;}
    div.stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
    }
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model files ----------
model = joblib.load('knn_model.pkl')
scaler = joblib.load("scaler.pkl")
features = joblib.load('features.pkl')

# ---------- Header ----------
st.title("❤️ Heart Disease Risk Prediction")
st.caption("Built by Akshatt  •  Powered by a K-Nearest Neighbors model")
st.markdown("Fill in the patient details below and click **Predict** to see the risk assessment.")
st.divider()

# ---------- Inputs ----------
st.subheader("👤 Personal Details")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 40)
with col2:
    sex = st.selectbox("Sex", ['M', 'F'], help="M = Male, F = Female")

st.subheader("🩺 Clinical Measurements")
col3, col4 = st.columns(2)
with col3:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL", [0, 1],
        help="1 = Yes, 0 = No",
    )
with col4:
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)

st.subheader("📈 Heart & Exercise Tests")
col5, col6 = st.columns(2)
with col5:
    chest_pain = st.selectbox(
        "Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'],
        help="ATA = Atypical Angina, NAP = Non-Anginal Pain, TA = Typical Angina, ASY = Asymptomatic",
    )
    resting_ecg = st.selectbox(
        "Resting ECG", ["Normal", "ST", "LVH"],
        help="ST = ST-T wave abnormality, LVH = Left ventricular hypertrophy",
    )
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina", ["Y", "N"],
        help="Y = Yes, N = No",
    )
with col6:
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox(
        "ST Slope", ["Up", "Flat", "Down"],
        help="Slope of the peak exercise ST segment",
    )

st.divider()

# ---------- Prediction ----------
if st.button("🔍 Predict", type="primary"):
    with st.spinner("Analyzing..."):
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }
        input_df = pd.DataFrame([raw_input])

        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[features]

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")


