"""
Global configuration for all GNSS experiments.

All paths are absolute to avoid confusion when running scripts from subdirectories.
"""

from pathlib import Path

# =============================================================================
# Paths (Read-Only Data Sources)
# =============================================================================

# Project root (honour ISRO_PROJECT_ROOT for Kaggle or custom environments)
import os
PROJECT_ROOT = Path(os.environ.get("ISRO_PROJECT_ROOT", "/home/tempest/Documents/ISRO_Project"))

# Raw data (never modify)
DATA_DIR = PROJECT_ROOT / "data" / "Raw_Logs"

# Phase 1 typed parquet cache (read-only)
TYPED_DIR = PROJECT_ROOT / "gnss-eda" / "Stats" / "scratch" / "typed"

# Session metadata
META_CSV = PROJECT_ROOT / "gnss-eda" / "Stats" / "scratch" / "sessions_meta.csv"

# Model artifacts (for loading existing models, not training)
MODEL_DIR = PROJECT_ROOT / "model"

# Experiments output
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Scratch for temporary files
SCRATCH_DIR = PROJECT_ROOT / "gnss-eda" / "scratch"

# =============================================================================
# DuckDB Resource Limits (HARD CONSTRAINT: 2 GB system RAM)
# =============================================================================

DUCKDB_MEMORY_LIMIT = "2GB"
DUCKDB_THREADS = 2
DUCKDB_TEMP_DIR = str(SCRATCH_DIR)

# =============================================================================
# Signal Quality Filters
# =============================================================================

# C/N0 range (exclude zero-reported values and extreme outliers)
CN0_MIN = 10.0  # dB-Hz
CN0_MAX = 55.0  # dB-Hz

# Elevation range (negative values are below horizon)
ELEVATION_MIN = 0.0  # degrees
ELEVATION_MAX = 90.0  # degrees

# Minimum track length for lag features
MIN_TRACK_LENGTH = 30  # epochs (seconds)

# =============================================================================
# Target Definition
# =============================================================================

# Prediction horizon
HORIZON_SECONDS = 15

# Fading threshold for binary classification
FADING_THRESHOLD_DB = 1.5  # |Δ C/N0| > 1.5 dB

# Sensitivity floor for dropout regret calculation
SENSITIVITY_FLOOR_DBHZ = 12.0

# =============================================================================
# Cross-Validation
# =============================================================================

# Number of CV folds
N_FOLDS = 5

# CV grouping column
CV_GROUP_COLUMN = "recording_date"

# Minimum sessions per fold
MIN_SESSIONS_PER_FOLD = 3

# =============================================================================
# Feature Engineering
# =============================================================================

# Rolling window for jitter calculation
JITTER_WINDOW_SECONDS = 5

# Elevation bands for categorical feature
ELEVATION_BANDS = {
    "low": (0, 20),
    "mid": (20, 45),
    "high": (45, 90),
}

# =============================================================================
# Evaluation Metrics
# =============================================================================

# Regret tolerance for hit rate
REGRET_TOLERANCE_DB = 0.5

# Top-k for ranking metrics
TOP_K = 4

# =============================================================================
# Device IDs
# =============================================================================

DEVICE_IDS = [
    "OnePlus CPH2401",
    "OnePlus CPH2487",
    "OPPO CPH2325",
    "samsung SM-M146B",
]

# =============================================================================
# Constellation Types (from Android GnssStatus)
# =============================================================================

CONSTELLATION_MAP = {
    1: "GPS",
    2: "SBAS",
    3: "GLONASS",
    4: "QZSS",
    5: "Beidou",
    6: "Galileo",
    7: "NavIC",
}

# =============================================================================
# Code Types (for signal identity key)
# =============================================================================

CODE_TYPE_PRIORITY = {
    # GPS
    "L1CA": 1,
    "L1C": 2,
    "L2C": 3,
    "L5I": 4,
    "L5Q": 5,
    # Galileo
    "E1B": 1,
    "E1C": 2,
    "E5aI": 3,
    "E5aQ": 4,
    # GLONASS
    "C": 1,
    "P": 2,
}
