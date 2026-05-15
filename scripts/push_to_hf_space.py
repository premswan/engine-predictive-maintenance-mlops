# ============================================================
# scripts/push_to_hf_space.py
# ============================================================
# Pushes generated deployment files to Hugging Face Space.

import os
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required.")
api = HfApi(token=HF_TOKEN)
hf_user = api.whoami(token=HF_TOKEN)["name"]
SPACE_REPO_ID = os.getenv("HF_SPACE_REPO_ID", f"{hf_user}/engine-predictive-maintenance-space")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
api.create_repo(repo_id=SPACE_REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True, private=False, token=HF_TOKEN)
api.upload_folder(folder_path=str(DEPLOYMENT_DIR), repo_id=SPACE_REPO_ID, repo_type="space", path_in_repo=".", commit_message="Deploy predictive maintenance Streamlit app", token=HF_TOKEN)
print(f"Space updated: https://huggingface.co/spaces/{SPACE_REPO_ID}")
