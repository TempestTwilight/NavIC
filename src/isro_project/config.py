#!/usr/bin/env python3
"""Centralized config for Kaggle-compatible paths."""

import os
from pathlib import Path

# Kaggle-standard paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Local dev fallback
LOCAL_ROOT = Path(__file__).resolve().parents[2]


def _env_path(key: str, default: Path) -> Path:
    """Get path from env var, fallback to default."""
    val = os.environ.get(key)
    return Path(val) if val else default


# ============================================================
# DATA INPUT PATHS (read-only on Kaggle)
# ============================================================
# Merged corpus (output of 01_merge_sessions.py)
MERGED_CORPUS = _env_path(
    "ISRO_MERGED_CORPUS",
    LOCAL_ROOT / "results" / "merged_data" / "combined_corpus.parquet",
)

# Raw categorized CSV data (for 01_merge_sessions.py CSV path)
CATEGORIZED_DIR = _env_path(
    "ISRO_CATEGORIZED_DIR", LOCAL_ROOT / "data" / "Categorized_Data"
)

# GNSS-EDA parquet (for 01_merge_sessions.py parquet path)
GNSS_PARQUET_DIR = _env_path(
    "ISRO_GNSS_PARQUET_DIR", LOCAL_ROOT / "gnss-eda" / "data" / "parquet"
)

# Raw GNSS logger text files
RAW_LOGS_DIR = _env_path("ISRO_RAW_LOGS_DIR", LOCAL_ROOT / "data" / "Raw_Logs")

# Test data for inference
TEST_DATA_DIR = _env_path("ISRO_TEST_DATA_DIR", LOCAL_ROOT / "data" / "test")

# ============================================================
# OUTPUT PATHS (writable on Kaggle -> /kaggle/working)
# ============================================================
RESULTS_DIR = _env_path("ISRO_RESULTS_DIR", LOCAL_ROOT / "results")

MERGED_OUTPUT_DIR = RESULTS_DIR / "merged_data"

# Model artifacts
MODEL_DIR = _env_path("ISRO_MODEL_DIR", LOCAL_ROOT / "model")

ORIENTATION_ARTIFACTS = MODEL_DIR / "orientation_artifacts"
TEMPORAL_ARTIFACTS = MODEL_DIR / "temporal_artifacts"
BASELINE_ARTIFACTS = MODEL_DIR / "baseline_artifacts"

# ============================================================
# TRAINING CONFIG
# ============================================================
RANDOM_STATE = int(os.environ.get("ISRO_RANDOM_STATE", "42"))
HORIZON_MS = int(os.environ.get("ISRO_HORIZON_MS", "15000"))
TARGET_TOLERANCE_MS = int(os.environ.get("ISRO_TARGET_TOLERANCE_MS", "1000"))

MAX_ORIENTATION_DT_MS = int(os.environ.get("ISRO_MAX_ORIENTATION_DT_MS", "500"))

# Memory limit for DuckDB
DUCKDB_MEMORY_LIMIT = os.environ.get("ISRO_DUCKDB_MEMORY_LIMIT", "2GB")


def ensure_dirs():
    """Create all output directories."""
    for d in [
        MERGED_OUTPUT_DIR,
        ORIENTATION_ARTIFACTS,
        TEMPORAL_ARTIFACTS,
        BASELINE_ARTIFACTS,
    ]:
        d.mkdir(parents=True, exist_ok=True)
