import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "premswan/engine-predictive-maintenance-model")
MODEL_FILENAME = "best_engine_maintenance_model.joblib"
FEATURE_COLUMNS = ["Engine_RPM", "Lub_Oil_Pressure", "Fuel_Pressure", "Coolant_Pressure", "Lub_Oil_Temperature", "Coolant_Temperature"]

LABEL_MAP = {
    0: "Normal / Healthy",
    1: "Maintenance Required / Faulty"
}


@st.cache_resource
def load_model():
    token = os.getenv("HF_TOKEN")
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        token=token
    )
    return joblib.load(model_path)


MODEL = load_model()

st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

st.title("Engine Predictive Maintenance Classifier")
st.write(
    "Enter engine sensor readings. The app loads the registered model from "
    "Hugging Face Model Hub and predicts whether maintenance is required."
)

with st.form("prediction_form"):
    st.subheader("Sensor Inputs")
    sensor_values = {}
    cols = st.columns(2)
    for idx, feature in enumerate(FEATURE_COLUMNS):
        with cols[idx % 2]:
            sensor_values[feature] = st.number_input(
                label=feature,
                value=0.0,
                format="%.6f"
            )

    submitted = st.form_submit_button("Predict Engine Condition")

if submitted:
    input_df = pd.DataFrame([sensor_values], columns=FEATURE_COLUMNS)
    prediction = int(MODEL.predict(input_df)[0])

    if hasattr(MODEL, "predict_proba"):
        probability_maintenance = float(MODEL.predict_proba(input_df)[0, 1])
    else:
        probability_maintenance = None

    prediction_label = LABEL_MAP.get(prediction, str(prediction))

    st.subheader("Prediction Output")
    st.metric("Predicted Engine Condition", prediction_label)

    if probability_maintenance is not None:
        st.metric("Probability of Maintenance/Faulty Class", "%.4f" % probability_maintenance)

    if prediction == 1:
        st.error("Recommended action: Schedule inspection or preventive maintenance before continued operation.")
    else:
        st.success("Recommended action: Continue normal operation and keep monitoring sensor readings.")

    st.subheader("Input DataFrame Used for Inference")
    st.dataframe(input_df, use_container_width=True)
