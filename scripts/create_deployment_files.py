# ============================================================
# scripts/create_deployment_files.py
# ============================================================
# Creates Docker/Streamlit deployment files for Hugging Face Space.

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required.")

api = HfApi(token=HF_TOKEN)
hf_user = api.whoami(token=HF_TOKEN)["name"]
MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", f"{hf_user}/engine-predictive-maintenance-model")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

try:
    metadata_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="model_metadata.json", token=HF_TOKEN)
    feature_cols = json.loads(Path(metadata_path).read_text(encoding="utf-8"))["feature_columns"]
except Exception:
    feature_cols = [
        "Engine_RPM",
        "Lub_Oil_Pressure",
        "Fuel_Pressure",
        "Coolant_Pressure",
        "Lub_Oil_Temperature",
        "Coolant_Temperature",
    ]

(DEPLOYMENT_DIR / "requirements.txt").write_text(
    "streamlit\npandas\nnumpy\nscikit-learn\njoblib\nhuggingface_hub\nxgboost\n",
    encoding="utf-8"
)

docker = """FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
"""
(DEPLOYMENT_DIR / "Dockerfile").write_text(docker, encoding="utf-8")

app_py_template = """import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "__MODEL_REPO_ID__")
MODEL_FILENAME = "best_engine_maintenance_model.joblib"
FEATURE_COLUMNS = __FEATURE_COLUMNS__

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
"""

app_py = (
    app_py_template
    .replace("__MODEL_REPO_ID__", MODEL_REPO_ID)
    .replace("__FEATURE_COLUMNS__", json.dumps(feature_cols))
)

(DEPLOYMENT_DIR / "app.py").write_text(app_py, encoding="utf-8")
print("Streamlit deployment files created in:", DEPLOYMENT_DIR)
