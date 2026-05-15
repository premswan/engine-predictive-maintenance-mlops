# ============================================================
# scripts/register_dataset.py
# ============================================================
# Creates/verifies the Hugging Face dataset repository and raw CSV availability.

import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required.")

api = HfApi(token=HF_TOKEN)
hf_user = api.whoami(token=HF_TOKEN)["name"]
DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", f"{hf_user}/engine-predictive-maintenance-data")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = DATA_DIR / "engine_data.csv"

api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True, private=False, token=HF_TOKEN)

if RAW_CSV.exists():
    api.upload_file(path_or_fileobj=str(RAW_CSV), path_in_repo="engine_data.csv", repo_id=DATASET_REPO_ID, repo_type="dataset", commit_message="Upload raw engine dataset", token=HF_TOKEN)
    print(f"Uploaded raw dataset to {DATASET_REPO_ID}")
else:
    try:
        hf_hub_download(repo_id=DATASET_REPO_ID, filename="engine_data.csv", repo_type="dataset", token=HF_TOKEN)
        print(f"Raw dataset already available in {DATASET_REPO_ID}")
    except Exception as exc:
        raise FileNotFoundError("Raw engine_data.csv not found in GitHub or Hugging Face Dataset Hub.") from exc
