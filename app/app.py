import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


st.set_page_config(
    page_title="Clinical Risk Prediction – Cirrhosis",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "..", "models")

model_path = os.path.join(models_dir, "rf_model.pkl")
model = joblib.load(model_path)

st.title("🧪 Clinical Mortality Risk Prediction")
st.markdown("""
This tool estimates **mortality risk** in patients with Primary Biliary Cirrhosis  
based on clinical and laboratory parameters.

⚠️IMPORTANT NOTE: *Educational use only. Not for clinical decision making.*
""")

st.divider()

st.header("Patient Clinical Data")

age = st.slider("Age (years)", 20, 90, 55)
sex = st.selectbox("Sex", ["F", "M"])
bilirubin = st.number_input("Bilirubin (mg/dL)", 0.1, 50.0, 2.5)
albumin = st.number_input("Albumin (g/dL)", 1.0, 5.0, 3.2)
alk_phos = st.number_input("Alkaline Phosphatase", 200, 5000, 1200)
sgot = st.number_input("SGOT / AST", 10, 500, 90)
platelets = st.number_input("Platelets (×10⁹/L)", 50, 500, 180)
tryglicerides = st.number_input("Tryglicerides", 10, 500, 90)
prothrombin = st.number_input("Prothrombin", 10, 500, 90)
stage = st.number_input("Stage", 1, 4, 2)
copper = st.number_input("Copper", 10, 500, 90)
cholesterol = st.number_input("Cholesterol", 10, 500, 90)
hepatomegaly = st.selectbox("Hepatomegaly", ["N", "Y"])
spiders = st.selectbox("Spiders", ["N", "Y"])

ascites = st.selectbox("Ascites (Fluid in abdomen)", ["N", "Y"])
edema = st.selectbox("Edema (Swelling)", ["N", "S", "Y"])

drug = st.selectbox(
    "Treatment",
    ["D-penicillamine", "Placebo"]
)

input_df = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "Bilirubin": bilirubin,
    "Albumin": albumin,
    "Alk_Phos": alk_phos,
    "SGOT": sgot,
    "Platelets": platelets,
    "Ascites": ascites,
    "Edema": edema,
    "Drug": drug,
    "Tryglicerides": tryglicerides,
    "Prothrombin": prothrombin,
    "Stage": stage,
    "Copper": copper,
    "Cholesterol": cholesterol,
    "Hepatomegaly": hepatomegaly,
    "Spiders": spiders
}])

if st.button("Predict Risk"):
    risk = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Predicted Mortality Risk")

    st.metric(
        label="Estimated Risk",
        value=f"{risk*100:.1f} %"
    )

    if risk < 0.2:
        st.success("Low estimated risk")
    elif risk < 0.5:
        st.warning("Moderate estimated risk")
    else:
        st.error("High estimated risk")

    st.markdown("""
    ### Interpretation
    This prediction reflects population-level patterns learned from historical
    clinical data and should always be interpreted in clinical context.
    """)

st.divider()

st.caption(
    "Author: Sebastian Lijewski, PhD | Applied Data Science in Medicine & Pharma"
)
