# ============================================================
# scripts/train_and_register_model.py
# ============================================================
# Trains allowed algorithms, tunes hyperparameters, evaluates, and registers best model.

import os
import re
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required.")
api = HfApi(token=HF_TOKEN)
hf_user = api.whoami(token=HF_TOKEN)["name"]
DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", f"{hf_user}/engine-predictive-maintenance-data")
MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", f"{hf_user}/engine-predictive-maintenance-model")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model_artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL = "Engine_Condition"
RANDOM_STATE = 42
COLUMN_RENAME_MAP = {"Engine rpm": "Engine_RPM", "Lub oil pressure": "Lub_Oil_Pressure", "Fuel pressure": "Fuel_Pressure", "Coolant pressure": "Coolant_Pressure", "lub oil temp": "Lub_Oil_Temperature", "Coolant temp": "Coolant_Temperature", "Engine Condition": "Engine_Condition"}

def standardize_columns(df):
    df = df.copy().rename(columns=COLUMN_RENAME_MAP)
    cols = []
    for col in df.columns:
        col = re.sub(r"[^A-Za-z0-9]+", "_", str(col).strip())
        cols.append(re.sub(r"_+", "_", col).strip("_"))
    df.columns = cols
    return df

def load_hf_csv(repo_id, file_name):
    try:
        return load_dataset(repo_id, data_files=file_name, split="train", token=HF_TOKEN)
    except TypeError:
        return load_dataset(repo_id, data_files=file_name, split="train", use_auth_token=HF_TOKEN)

train_df = standardize_columns(load_hf_csv(DATASET_REPO_ID, "train.csv").to_pandas())
test_df = standardize_columns(load_hf_csv(DATASET_REPO_ID, "test.csv").to_pandas())
feature_cols = [c for c in train_df.columns if c != TARGET_COL]
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL].astype(int)
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL].astype(int)
models_and_params = {
    "Decision_Tree": {"estimator": DecisionTreeClassifier(random_state=RANDOM_STATE), "params": {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 2, 5]}},
    "Bagging": {"estimator": BaggingClassifier(random_state=RANDOM_STATE, n_jobs=-1), "params": {"model__n_estimators": [50, 100], "model__max_samples": [0.8, 1.0]}},
    "Random_Forest": {"estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), "params": {"model__n_estimators": [100, 200], "model__max_depth": [8, None], "model__class_weight": [None, "balanced"]}},
    "AdaBoost": {"estimator": AdaBoostClassifier(random_state=RANDOM_STATE), "params": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.05, 0.1, 0.5]}},
    "Gradient_Boosting": {"estimator": GradientBoostingClassifier(random_state=RANDOM_STATE), "params": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}}
}
if XGB_AVAILABLE:
    models_and_params["XGBoost"] = {"estimator": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1), "params": {"model__n_estimators": [50, 100], "model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]}}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
results, fitted_models = [], {}
for model_name, config in models_and_params.items():
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", config["estimator"])])
    total = int(np.prod([len(v) for v in config["params"].values()]))
    search = RandomizedSearchCV(pipe, param_distributions=config["params"], n_iter=min(4, total), scoring="f1", cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {"accuracy": accuracy_score(y_test, y_pred), "precision": precision_score(y_test, y_pred, zero_division=0), "recall": recall_score(y_test, y_pred, zero_division=0), "f1": f1_score(y_test, y_pred, zero_division=0), "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan, "best_cv_f1": search.best_score_}
    results.append({"model_name": model_name, **metrics, "best_params": json.dumps(search.best_params_)})
    fitted_models[model_name] = model
results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
best_model_name = results_df.loc[0, "model_name"]
best_model = fitted_models[best_model_name]
joblib.dump(best_model, MODEL_DIR / "best_engine_maintenance_model.joblib")
results_df.to_csv(MODEL_DIR / "model_experiment_results.csv", index=False)
metadata = {"created_at": datetime.utcnow().isoformat() + "Z", "best_model_name": best_model_name, "target_column": TARGET_COL, "feature_columns": feature_cols, "selection_metric": "f1", "best_model_metrics": results_df.iloc[0].drop(labels=["best_params"]).to_dict(), "best_params": json.loads(results_df.loc[0, "best_params"])}
(MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
(MODEL_DIR / "requirements.txt").write_text("pandas\nnumpy\nscikit-learn\njoblib\nxgboost\n", encoding="utf-8")
(MODEL_DIR / "README.md").write_text(f"# Engine Predictive Maintenance Model\n\nBest model: {best_model_name}\n", encoding="utf-8")
api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True, private=False, token=HF_TOKEN)
api.upload_folder(folder_path=str(MODEL_DIR), repo_id=MODEL_REPO_ID, repo_type="model", path_in_repo=".", commit_message="Register best model from GitHub Actions", token=HF_TOKEN)
print(f"Best model registered: {MODEL_REPO_ID}")
