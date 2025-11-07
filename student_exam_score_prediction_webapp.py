# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 20:18:44 2025

@author: tosindataginius
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


st.set_page_config(page_title="Student Exam Score Predictor", layout="wide", initial_sidebar_state="expanded")

# --- UI styling ---
st.markdown(
    """
    <style>
    .app-header {display:flex; align-items:center; gap:16px}
    .title {font-size:28px; font-weight:700}
    .subtitle {color:#6b7280}
    .card {background:#fff;padding:18px;border-radius:12px;box-shadow:0 6px 18px rgba(15,23,42,0.06)}
    .small {font-size:13px; color:#6b7280}
    .muted {color:#6b7280}
    </style>
    """,
    unsafe_allow_html=True,
)

# Model Path
#MODEL_PATH  = r"C:\Users\USER\Downloads\Machine\student_exam_score_prediction_app\student_exam_score_prediction.joblib"
MODEL_PATH = "student_exam_score_prediction.joblib"

@st.cache_data
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None, f"Model file not found at {path}. Place your joblib model beside the app."
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


def infer_feature_names_from_model(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None


def default_feature_explanation(col):
    col_l = col.lower()
    if "study" in col_l or "hours" in col_l:
        return "Estimated weekly study hours — higher values often correlate with better exam performance."
    if "attendance" in col_l:
        return "Attendance percentage — better class attendance typically improves scores."
    if "prev" in col_l or "score" in col_l:
        return "Previous academic performance — strong predictor of future scores."
    if "sleep" in col_l:
        return "Average sleep hours per night — both insufficient and excessive sleep can impact performance."
    if "age" in col_l:
        return "Age of the student in years."
    if "gender" in col_l:
        return "Encoded gender (e.g., 0 = female, 1 = male)."
    if "parent" in col_l or "socio" in col_l or "income" in col_l:
        return "Socioeconomic factors (family support, parental education, income) that influence educational outcomes."
    return "Feature used by the model; typically related to student background, study behaviour, or prior performance."

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1995/1995604.png", width=110)
    st.title("Exam Score Predictor")
    st.markdown("---")
    st.write("<div class='small'>Model automatically loads from local joblib file.</div>", unsafe_allow_html=True)

# --- Load model ---
model, load_err = load_model(MODEL_PATH)

# --- Main layout ---
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("<div class='app-header'><div class='title'>Student Exam Score Predictor</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Enter student attributes to get a predicted exam score. The app auto-detects features when possible.</div>", unsafe_allow_html=True)

    # Feature handling
    feature_names = None
    if model is not None:
        feature_names = infer_feature_names_from_model(model)

    st.markdown("---")
    if feature_names:
        st.success(f"Model loaded — detected {len(feature_names)} feature(s).")
    else:
        if model is None:
            st.warning(load_err or "No model loaded. Place 'student_exam_score_prediction.joblib' beside this app.")
        else:
            st.info("Model loaded but no feature names were detectable. Provide them manually if needed.")

    if feature_names is None:
        feature_names = [
            "study_hours_per_week",
            "attendance_pct",
            "previous_score",
            "sleep_hours",
            "parental_education_level",
            "age",
        ]
        st.info("No features detected. Using default feature set for demonstration.")

    # Input form
    st.subheader("Student inputs")
    with st.form("predict_form"):
        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            col = cols[i % 3]
            with col:
                lower = feat.lower()
                if any(k in lower for k in ["pct", "percent", "attendance"]):
                    val = st.number_input(feat, min_value=0.0, max_value=100.0, value=75.0, step=1.0)
                elif any(k in lower for k in ["hours", "study", "sleep"]):
                    val = st.number_input(feat, min_value=0.0, max_value=200.0, value=10.0, format="%.1f")
                elif any(k in lower for k in ["score", "previous"]):
                    val = st.number_input(feat, min_value=0.0, max_value=100.0, value=60.0)
                elif any(k in lower for k in ["gender", "encoded", "binary", "is_"]):
                    val = st.selectbox(feat, options=[0,1])
                else:
                    val = st.number_input(feat, value=0.0, format="%.2f")
                input_vals[feat] = val
        predict_button = st.form_submit_button("Predict exam score")

    input_df = pd.DataFrame([input_vals])

    if predict_button:
        if model is None:
            st.error(load_err or "No model available.")
        else:
            try:
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted exam score: {prediction:.2f}")

                st.markdown("---")
                st.subheader("Input summary")
                st.table(input_df.T)

                st.info("No uncertainty estimate available for this model. Consider using ensemble regressors for prediction intervals.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model & Features")
    if model is None:
        st.warning("No model loaded")
    else:
        st.write(f"Model type: {type(model)}")
        fn = infer_feature_names_from_model(model)
        if fn:
            st.write("Detected feature names:")
            for f in fn:
                st.write(f"- **{f}**: {default_feature_explanation(f)}")
        else:
            st.write("Feature names not detectable. Using defaults.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("This app provides predicted exam scores based on a saved ML model. Always validate predictions before using them for important decisions.")
