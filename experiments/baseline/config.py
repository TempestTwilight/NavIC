"""
Experiment A: Ridge Regression Baseline with Proper CV

Goal: Establish whether a simple linear model can beat persistence baseline
Target: Residual Δ C/N0 at 15-second horizon
Features: [s_bz, sin(elevation), σ₅ₛ, device_id, current C/N0]
CV: GroupKFold by calendar day
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.config import EXPERIMENTS_DIR

# =============================================================================
# Experiment-Specific Paths
# =============================================================================

EXPERIMENT_NAME = "baseline_ridge"
EXPERIMENT_DIR = EXPERIMENTS_DIR / "baseline"
RESULTS_DIR = EXPERIMENT_DIR / "results"
FEATURES_DIR = EXPERIMENT_DIR / "features"

# Ensure directories exist
for d in [RESULTS_DIR, FEATURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Feature Configuration
# =============================================================================


@dataclass
class FeatureConfig:
    """Features to extract for baseline model."""

    # Temporal features
    current_cn0: bool = True
    delta_recent_5s: bool = True
    delta_recent_10s: bool = False

    # Orientation features (quaternion-derived body vectors)
    s_bz: bool = True  # Screen-facing axis (directivity)
    s_by: bool = False  # Lateral axis
    off_axis_angle: bool = False

    # Physics features
    sin_elevation: bool = True
    elevation_band: bool = False  # Categorical: low/mid/high

    # Jitter features
    sigma_5s: bool = True
    sigma_10s: bool = False

    # Context features
    device_id: bool = True  # One-hot encoded
    constellation_type: bool = True

    # Target
    delta_cn0_15s: bool = True


FEATURE_CONFIG = FeatureConfig()

# =============================================================================
# Model Configuration
# =============================================================================

# Ridge regression hyperparameters
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Target normalization
TARGET_CENTERING = "epoch"  # Options: "epoch", "device", "none"

# =============================================================================
# Baseline Metrics (from Phase 2 Table 26)
# =============================================================================

# Persistence baseline (continuously tracked epochs only)
PERSISTENCE_MEAN_REGRET_DB = 0.16
PERSISTENCE_MEDIAN_REGRET_DB = 0.0
PERSISTENCE_HIT_RATE_05DB_PCT = 94.8

# Note: This excludes dropout regret; corpus-wide baseline would be higher
