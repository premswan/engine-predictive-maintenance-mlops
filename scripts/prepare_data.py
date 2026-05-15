# ============================================================
# scripts/prepare_data.py
# ============================================================
# Loads raw data from Hugging Face, cleans it, splits train/test, and uploads outputs.

import os
import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required.")

api = HfApi(token=HF_TOKEN)
hf_user = api.whoami(token=HF_TOKEN)["name"]
DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", f"{hf_user}/engine-predictive-maintenance-data")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_RENAME_MAP = {"Engine rpm": "Engine_RPM", "Lub oil pressure": "Lub_Oil_Pressure", "Fuel pressure": "Fuel_Pressure", "Coolant pressure": "Coolant_Pressure", "lub oil temp": "Lub_Oil_Temperature", "Coolant temp": "Coolant_Temperature", "Engine Condition": "Engine_Condition"}
TARGET_COL = "Engine_Condition"
RANDOM_STATE = 42

def standardize_columns(df):
    df = df.copy().rename(columns=COLUMN_RENAME_MAP)
    cleaned = []
    for col in df.columns:
        col = re.sub(r"[^A-Za-z0-9]+", "_", str(col).strip())
        cleaned.append(re.sub(r"_+", "_", col).strip("_"))
    df.columns = cleaned
    return df

def load_hf_csv(repo_id, file_name):
    try:
        return load_dataset(repo_id, data_files=file_name, split="train", token=HF_TOKEN)
    except TypeError:
        return load_dataset(repo_id, data_files=file_name, split="train", use_auth_token=HF_TOKEN)

raw_df = load_hf_csv(DATASET_REPO_ID, "engine_data.csv").to_pandas()
clean_df = standardize_columns(raw_df).drop_duplicates()
clean_df = clean_df.drop(columns=[c for c in clean_df.columns if clean_df[c].isna().all()])
constant_cols = [c for c in clean_df.columns if c != TARGET_COL and clean_df[c].nunique(dropna=True) <= 1]
clean_df = clean_df.drop(columns=constant_cols)
if TARGET_COL not in clean_df.columns:
    raise ValueError(f"Target column {TARGET_COL} not found.")
for col in clean_df.columns:
    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
clean_df = clean_df.dropna(subset=[TARGET_COL])
clean_df[TARGET_COL] = clean_df[TARGET_COL].astype(int)
clean_df = clean_df[clean_df[TARGET_COL].isin([0, 1])].reset_index(drop=True)
train_df, test_df = train_test_split(clean_df, test_size=0.20, random_state=RANDOM_STATE, stratify=clean_df[TARGET_COL])
paths = {"clean_engine_data.csv": DATA_DIR / "clean_engine_data.csv", "train.csv": DATA_DIR / "train.csv", "test.csv": DATA_DIR / "test.csv"}
clean_df.to_csv(paths["clean_engine_data.csv"], index=False)
train_df.to_csv(paths["train.csv"], index=False)
test_df.to_csv(paths["test.csv"], index=False)
for repo_file, local_path in paths.items():
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=repo_file, repo_id=DATASET_REPO_ID, repo_type="dataset", commit_message=f"Upload {repo_file}", token=HF_TOKEN)
print("Prepared and uploaded clean/train/test datasets.")
