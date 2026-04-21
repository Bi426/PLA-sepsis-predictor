import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="PLA Sepsis Predictor", layout="centered")

st.title("PLA In-hospital Sepsis Risk Predictor")

# 读取模型包
bundle = joblib.load("best_model_bundle.joblib")

best_model = bundle["best_model"]
preprocessor = bundle["preprocessor"]
boruta_support = bundle["boruta_support"]
scaler_rfe = bundle["scaler_rfe"]
rfecv_support = bundle["rfecv_support"]

input_feature_cols = bundle["input_feature_cols"]
uni_selected_cols = bundle["uni_selected_cols"]
corr_selected_cols = bundle["corr_selected_cols"]

# ===== 这里只保留最终网页显示的 6 个变量 =====
DISPLAY_COLS = [
    "Lymphocytes",
    "Na",
    "NPR",
    "IL_6",
    "Procalcitonin",
    "CREA"
]

st.write("Please enter the required variables below:")

user_input = {}
for col in DISPLAY_COLS:
    user_input[col] = st.number_input(col, value=0.0, format="%.4f")

if st.button("Predict"):
    # 先构造训练时完整原始输入框架，其余变量用缺失占位
    raw_df = pd.DataFrame([{col: np.nan for col in input_feature_cols}])

    # 填入网页上这 6 个变量
    for col in DISPLAY_COLS:
        raw_df.loc[0, col] = user_input[col]

    # 按训练时相同流程处理
    x = raw_df[uni_selected_cols].copy()
    x = x[corr_selected_cols].copy()

    x_pre = preprocessor.transform(x)
    x_boruta = np.array(x_pre)[:, boruta_support]
    x_scaled = scaler_rfe.transform(x_boruta)
    x_sel = x_scaled[:, rfecv_support]

    # 预测
    if hasattr(best_model, "predict_proba"):
        prob = best_model.predict_proba(x_sel)[0, 1]
    else:
        raw_score = best_model.decision_function(x_sel)[0]
        prob = 1 / (1 + np.exp(-raw_score))

    pred = int(prob >= 0.5)

    st.subheader("Prediction result")
    st.write(f"Predicted probability of in-hospital sepsis: **{prob:.3f}**")
    st.write(f"Predicted class: **{'Sepsis' if pred == 1 else 'Non-sepsis'}**")

    if pred == 1:
        st.error("High-risk prediction")
    else:
        st.success("Lower-risk prediction")
