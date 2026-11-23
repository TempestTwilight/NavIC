#!/usr/bin/env bash
# ==============================================================================
# Helper script to package and push code & data to Kaggle, run the training
# kernel, and download the resulting artifacts back into the repository.
#
# Usage:
#   ./kaggle/run_kaggle.sh setup     # Package & push datasets
#   ./kaggle/run_kaggle.sh run       # Push kernel and start training run
#   ./kaggle/run_kaggle.sh status    # Check kernel execution status
#   ./kaggle/run_kaggle.sh logs      # Fetch kernel output logs
#   ./kaggle/run_kaggle.sh pull      # Retrieve artifacts into local codebase
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_DIR="${REPO_ROOT}/kaggle"
STAGING_DIR="/tmp/isro_kaggle_staging"

# Slugs (adjust if needed)
KAGGLE_USER="$(kaggle config view | grep username | awk '{print $2}')"
DATA_SLUG="isro-gnss-data"
CODE_SLUG="isro-gnss-code"
KERNEL_SLUG="isro-gnss-train"

case "${1:-help}" in
  setup)
    echo "=== 1. Staging and uploading GNSS data ==="
    mkdir -p "${STAGING_DIR}/data"
    cp "${REPO_ROOT}/results/merged_data/combined_corpus.parquet" "${STAGING_DIR}/data/"
    cat <<EOF > "${STAGING_DIR}/data/dataset-metadata.json"
{
  "title": "ISRO GNSS Training Data",
  "id": "${KAGGLE_USER}/${DATA_SLUG}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    if kaggle datasets status "${KAGGLE_USER}/${DATA_SLUG}" &>/dev/null; then
      echo "Dataset exists. Creating new version..."
      kaggle datasets version -p "${STAGING_DIR}/data" -m "Update training corpus" --dir-mode tar
    else
      echo "Creating new dataset..."
      kaggle datasets create -p "${STAGING_DIR}/data" --dir-mode tar
    fi

    echo "=== 2. Staging and uploading codebase ==="
    mkdir -p "${STAGING_DIR}/code"
    rsync -av --exclude-from="${REPO_ROOT}/.gitignore" \
      --exclude=".git" \
      --exclude=".venv" \
      --exclude="__pycache__" \
      --exclude="*.parquet" \
      --exclude="data" \
      --exclude="gnss-eda" \
      --exclude="results" \
      --exclude="model" \
      "${REPO_ROOT}/" "${STAGING_DIR}/code/"

    cat <<EOF > "${STAGING_DIR}/code/dataset-metadata.json"
{
  "title": "ISRO GNSS Codebase",
  "id": "${KAGGLE_USER}/${CODE_SLUG}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    if kaggle datasets status "${KAGGLE_USER}/${CODE_SLUG}" &>/dev/null; then
      echo "Code dataset exists. Creating new version..."
      kaggle datasets version -p "${STAGING_DIR}/code" -m "Update codebase" --dir-mode tar
    else
      echo "Creating new code dataset..."
      kaggle datasets create -p "${STAGING_DIR}/code" --dir-mode tar
    fi

    echo "=== Setup complete ==="
    ;;

  run)
    echo "=== Pushing and starting Kaggle training kernel ==="
    mkdir -p "${STAGING_DIR}/kernel"
    cp "${KAGGLE_DIR}/kaggle_train.py" "${STAGING_DIR}/kernel/"

    cat <<EOF > "${STAGING_DIR}/kernel/kernel-metadata.json"
{
  "id": "${KAGGLE_USER}/${KERNEL_SLUG}",
  "title": "ISRO GNSS Training Run",
  "code_file": "kaggle_train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_tpu": "false",
  "enable_internet": "true",
  "dataset_sources": [
    "${KAGGLE_USER}/${DATA_SLUG}",
    "${KAGGLE_USER}/${CODE_SLUG}"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF
    kaggle kernels push -p "${STAGING_DIR}/kernel"
    echo "Kernel pushed. Monitor progress with: ./kaggle/run_kaggle.sh status"
    ;;

  status)
    kaggle kernels status "${KAGGLE_USER}/${KERNEL_SLUG}"
    ;;

  logs)
    kaggle kernels output "${KAGGLE_USER}/${KERNEL_SLUG}"
    ;;

  pull)
    echo "=== Pulling output artifacts from Kaggle ==="
    mkdir -p "${STAGING_DIR}/output"
    kaggle kernels output "${KAGGLE_USER}/${KERNEL_SLUG}" -p "${STAGING_DIR}/output"

    echo "Distributing artifacts into repository folders..."
    # 1. Models
    if [ -d "${STAGING_DIR}/output/model/orientation_artifacts" ]; then
      mkdir -p "${REPO_ROOT}/model/orientation_artifacts"
      cp -r "${STAGING_DIR}/output/model/orientation_artifacts/"* "${REPO_ROOT}/model/orientation_artifacts/"
    fi
    if [ -d "${STAGING_DIR}/output/model/temporal_artifacts" ]; then
      mkdir -p "${REPO_ROOT}/model/temporal_artifacts"
      cp -r "${STAGING_DIR}/output/model/temporal_artifacts/"* "${REPO_ROOT}/model/temporal_artifacts/"
    fi

    # 2. Experiments
    if [ -d "${STAGING_DIR}/output/experiments" ]; then
      for exp in baseline hurdle ltr cross_device survival; do
        if [ -d "${STAGING_DIR}/output/experiments/${exp}/results" ]; then
          mkdir -p "${REPO_ROOT}/experiments/${exp}/results"
          cp -r "${STAGING_DIR}/output/experiments/${exp}/results/"* "${REPO_ROOT}/experiments/${exp}/results/"
        fi
      done
    fi

    echo "Artifacts placed successfully. Verifying..."
    python3 -c "
import joblib, json, glob
for path in glob.glob('${REPO_ROOT}/model/*_artifacts/*.joblib'):
    model = joblib.load(path)
    print(f'Verified valid joblib model: {path}')
for path in glob.glob('${REPO_ROOT}/model/*_artifacts/*.json'):
    json.load(open(path))
    print(f'Verified valid json artifact: {path}')
"
    ;;

  *)
    echo "Usage: $0 {setup|run|status|logs|pull}"
    exit 1
    ;;
esac
