import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="PLA Sepsis Predictor", layout="centered")

# ==============================
# Load model bundle
# ==============================
bundle = joblib.load("best_model_bundle.joblib")

best_model = bundle["best_model"]
preprocessor = bundle["preprocessor"]
boruta_support = bundle["boruta_support"]
scaler_rfe = bundle["scaler_rfe"]
rfecv_support = bundle["rfecv_support"]

input_feature_cols = bundle["input_feature_cols"]
uni_selected_cols = bundle["uni_selected_cols"]
corr_selected_cols = bundle["corr_selected_cols"]

# ==============================
# Final displayed variables
# ==============================
DISPLAY_COLS = [
    "Lymphocytes",
    "Na",
    "NPR",
    "IL_6",
    "Procalcitonin",
    "CREA"
]

# More user-friendly labels
DISPLAY_LABELS = {
    "Lymphocytes": "Lymphocyte Count (×10⁹/L)",
    "Na": "Serum Sodium (mmol/L)",
    "NPR": "Neutrophil-to-Platelet Ratio",
    "IL_6": "Interleukin-6 (pg/mL)",
    "Procalcitonin": "Procalcitonin (ng/mL)",
    "CREA": "Creatinine (μmol/L)"
}

# Suggested default values
DEFAULT_VALUES = {
    "Lymphocytes": 0.34,
    "Na": 136.0,
    "NPR": 0.0308,
    "IL_6": 58.86,
    "Procalcitonin": 6.976,
    "CREA": 83.0
}

# Optional lower bounds
MIN_VALUES = {
    "Lymphocytes": 0.0,
    "Na": 0.0,
    "NPR": 0.0,
    "IL_6": 0.0,
    "Procalcitonin": 0.0,
    "CREA": 0.0
}

# ==============================
# Page title and note
# ==============================
st.title("PLA In-hospital Sepsis Risk Predictor")

st.caption(
    "Please enter baseline admission values for the six variables below. "
    "This tool is intended for research use and does not replace clinical judgment."
)

st.divider()

# ==============================
# Input section
# ==============================
st.subheader("Input Variables")

user_input = {}
for col in DISPLAY_COLS:
    user_input[col] = st.number_input(
        DISPLAY_LABELS[col],
        min_value=MIN_VALUES[col],
        value=float(DEFAULT_VALUES[col]),
        format="%.4f"
    )

# ==============================
# Prediction
# ==============================
if st.button("Predict", use_container_width=True):
    # Build a full raw input frame matching the original training input space
    raw_df = pd.DataFrame([{col: np.nan for col in input_feature_cols}])

    for col in DISPLAY_COLS:
        raw_df.loc[0, col] = user_input[col]

    # Follow the exact training-time processing pipeline
    x = raw_df[uni_selected_cols].copy()
    x = x[corr_selected_cols].copy()

    x_pre = preprocessor.transform(x)
    x_boruta = np.array(x_pre)[:, boruta_support]
    x_scaled = scaler_rfe.transform(x_boruta)
    x_sel = x_scaled[:, rfecv_support]

    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(x_sel)[0, 1]
    else:
        raw_score = best_model.decision_function(x_sel)[0]
        prob = 1 / (1 + np.exp(-raw_score))

    pred = int(prob >= 0.5)

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Probability",
            value=f"{prob:.3f}"
        )

    with col2:
        st.metric(
            label="Predicted Classification",
            value="Sepsis" if pred == 1 else "Non-sepsis"
        )

    if pred == 1:
        st.error(
            f"High-risk prediction: the estimated probability of in-hospital sepsis is {prob:.3f}."
        )
    else:
        st.success(
            f"Lower-risk prediction: the estimated probability of in-hospital sepsis is {prob:.3f}."
        )

    st.caption(
        "Classification threshold: 0.50. "
        "Results should be interpreted together with the overall clinical context."
    )
