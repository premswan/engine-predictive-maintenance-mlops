# Engine Predictive Maintenance MLOps Project

This repository contains the final MLOps workflow for the predictive maintenance project.

## Main Components

- Data registration on Hugging Face Dataset Hub
- Data preparation and train/test split upload
- Model training with hyperparameter tuning
- Experiment tracking and model artifact generation
- Best model registration on Hugging Face Model Hub
- Hugging Face Space deployment files
- GitHub Actions pipeline for end-to-end automation

## Hugging Face Repositories

- Dataset repository: https://huggingface.co/datasets/premswan/engine-predictive-maintenance-data
- Model repository: https://huggingface.co/premswan/engine-predictive-maintenance-model
- Space repository: https://huggingface.co/spaces/premswan/engine-predictive-maintenance-space

## Required GitHub Secrets

- `HF_TOKEN`: Hugging Face token with write access

The workflow uses GitHub's built-in `GITHUB_TOKEN` to commit generated updates back to the `main` branch.
