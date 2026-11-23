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
DATA_SLUG="isro-gnss-data"
CODE_SLUG="isro-gnss-code"
KERNEL_SLUG="isro-gnss-train"

usage() {
  echo "Usage: $0 {setup|run|status|logs|pull}"
}

# Resolve the Kaggle username: env var -> kaggle.json -> `kaggle config view`.
get_kaggle_user() {
  if [ -n "${KAGGLE_USERNAME:-}" ]; then
    echo "${KAGGLE_USERNAME}"
    return
  fi
  local cfg="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
  if [ -f "$cfg" ]; then
    python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['username'])" "$cfg"
    return
  fi
  # Output looks like "- username: name", so the value is field 3.
  kaggle config view | grep username | awk '{print $3}'
}

# Wait until a dataset finishes processing (needed before a kernel can use it).
wait_for_dataset() {
  local ref="$1" tries=30
  echo "Waiting for dataset ${ref} to be ready..."
  for ((i = 1; i <= tries; i++)); do
    if [ "$(kaggle datasets status "$ref" 2>/dev/null | tr -d '[:space:]')" = "ready" ]; then
      echo "  ${ref}: ready"
      return 0
    fi
    sleep 10
  done
  echo "ERROR: dataset ${ref} not ready after $((tries * 10))s" >&2
  return 1
}

# Copy the contents of $1 into $2 (safe when $1 is empty, unlike `cp dir/*`).
copy_dir_contents() {
  mkdir -p "$2"
  cp -r "$1/." "$2/"
}

COMMAND="${1:-help}"

case "${COMMAND}" in
  help|-h|--help)
    usage
    exit 0
    ;;
  setup|run|status|logs|pull)
    KAGGLE_USER="$(get_kaggle_user)"
    if [ -z "${KAGGLE_USER}" ]; then
      echo "ERROR: could not determine Kaggle username. Set KAGGLE_USERNAME or configure ~/.kaggle/kaggle.json" >&2
      exit 1
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac

case "${COMMAND}" in
  setup)
    echo "=== 1. Staging and uploading GNSS data ==="
    # NOTE: Make sure you're allowed to release this data under CC0,
    # even for a private dataset. Change the license if not.
    rm -rf "${STAGING_DIR}/data"
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
      kaggle datasets version -p "${STAGING_DIR}/data" -m "Update training corpus"
    else
      echo "Creating new dataset..."
      kaggle datasets create -p "${STAGING_DIR}/data"
    fi

    echo "=== 2. Staging and uploading codebase ==="
    # Start from a clean directory so deleted files don't linger.
    rm -rf "${STAGING_DIR}/code"
    mkdir -p "${STAGING_DIR}/code"

    RSYNC_ARGS=(-a --delete)
    if [ -f "${REPO_ROOT}/.gitignore" ]; then
      RSYNC_ARGS+=(--exclude-from="${REPO_ROOT}/.gitignore")
    fi
    # Leading "/" anchors each pattern to the repo root, so nested packages
    # such as src/data/ or src/model/ are NOT excluded.
    RSYNC_ARGS+=(
      --exclude="/.git"
      --exclude="/.venv"
      --exclude="__pycache__"
      --exclude="*.parquet"
      --exclude="/data"
      --exclude="/gnss-eda"
      --exclude="/results"
      --exclude="/model"
    )
    rsync "${RSYNC_ARGS[@]}" "${REPO_ROOT}/" "${STAGING_DIR}/code/"

    echo "Staged files (review before trusting the upload):"
    (cd "${STAGING_DIR}/code" && find . -type f | sort | head -n 50)

    cat <<EOF > "${STAGING_DIR}/code/dataset-metadata.json"
{
  "title": "ISRO GNSS Codebase",
  "id": "${KAGGLE_USER}/${CODE_SLUG}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    # zip: subdirectories are zipped and Kaggle auto-extracts them.
    if kaggle datasets status "${KAGGLE_USER}/${CODE_SLUG}" &>/dev/null; then
      echo "Code dataset exists. Creating new version..."
      kaggle datasets version -p "${STAGING_DIR}/code" -m "Update codebase" --dir-mode zip
    else
      echo "Creating new code dataset..."
      kaggle datasets create -p "${STAGING_DIR}/code" --dir-mode zip
    fi

    echo "=== Setup complete ==="
    ;;

  run)
    wait_for_dataset "${KAGGLE_USER}/${DATA_SLUG}"
    wait_for_dataset "${KAGGLE_USER}/${CODE_SLUG}"

    echo "=== Pushing and starting Kaggle training kernel ==="
    rm -rf "${STAGING_DIR}/kernel"
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
    LOG_DIR="${STAGING_DIR}/logs"
    rm -rf "${LOG_DIR}"
    mkdir -p "${LOG_DIR}"
    kaggle kernels output "${KAGGLE_USER}/${KERNEL_SLUG}" -p "${LOG_DIR}"
    shopt -s nullglob
    log_files=("${LOG_DIR}"/*.log)
    if [ ${#log_files[@]} -eq 0 ]; then
      echo "No .log file found in ${LOG_DIR}. Downloaded files:"
      ls -R "${LOG_DIR}"
    else
      cat "${log_files[@]}"
    fi
    ;;

  pull)
    echo "=== Pulling output artifacts from Kaggle ==="
    # Clear old output so artifacts from a previous run can't sneak in.
    rm -rf "${STAGING_DIR}/output"
    mkdir -p "${STAGING_DIR}/output"
    kaggle kernels output "${KAGGLE_USER}/${KERNEL_SLUG}" -p "${STAGING_DIR}/output"

    echo "Distributing artifacts into repository folders..."
    # 1. Models
    for art in orientation_artifacts temporal_artifacts; do
      if [ -d "${STAGING_DIR}/output/model/${art}" ]; then
        copy_dir_contents "${STAGING_DIR}/output/model/${art}" "${REPO_ROOT}/model/${art}"
      fi
    done

    # 2. Experiments
    if [ -d "${STAGING_DIR}/output/experiments" ]; then
      for exp in baseline hurdle ltr cross_device survival; do
        if [ -d "${STAGING_DIR}/output/experiments/${exp}/results" ]; then
          copy_dir_contents "${STAGING_DIR}/output/experiments/${exp}/results" \
                            "${REPO_ROOT}/experiments/${exp}/results"
        fi
      done
    fi

    echo "Artifacts placed. Verifying..."
    # Only run joblib.load on files you produced yourself: it can execute code.
    REPO_ROOT="${REPO_ROOT}" python3 - <<'PY'
import glob, json, os
import joblib

root = os.environ["REPO_ROOT"]
models = glob.glob(f"{root}/model/*_artifacts/*.joblib")
jsons = glob.glob(f"{root}/model/*_artifacts/*.json")

if not models and not jsons:
    print("WARNING: no model artifacts found to verify.")

for path in models:
    joblib.load(path)
    print(f"Verified valid joblib model: {path}")
for path in jsons:
    with open(path) as f:
        json.load(f)
    print(f"Verified valid json artifact: {path}")
PY
    ;;
esac