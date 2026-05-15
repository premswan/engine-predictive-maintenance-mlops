import os
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO_ID = os.getenv("HF_SPACE_REPO_ID", "premswan/engine-predictive-maintenance-space")
DEPLOYMENT_DIR = Path(__file__).resolve().parent

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required to push files to Hugging Face Space.")

api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    private=False,
    token=HF_TOKEN
)

api.upload_folder(
    folder_path=str(DEPLOYMENT_DIR),
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    path_in_repo=".",
    commit_message="Deploy predictive maintenance Streamlit app",
    token=HF_TOKEN
)

print(f"Deployment files pushed to: https://huggingface.co/spaces/{SPACE_REPO_ID}")
