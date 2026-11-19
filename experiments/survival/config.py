"""
Experiment E: Survival-Weighted Selection

Goal: Incorporate dropout probability into selection utility
"""


# Config definitions

# =============================================================================
# Survival Model Configuration
# =============================================================================

# Features for survival probability
SURVIVAL_FEATURES = [
    "ElevationDegrees",
    "sin_elevation",
    "current_cn0",
    "sigma_5s",
    "ConstellationType",
]

# Survival threshold (15-second horizon)
SURVIVAL_HORIZON = 15  # seconds

# Definition of "survive": C/N0 stays above sensitivity floor
SURVIVAL_THRESHOLD_DBHZ = 15.0  # Minimum usable signal

# =============================================================================
# Utility Formulation
# =============================================================================

# Expected value formula:
# E[C/N0] = P(survive) * predicted_C/N0 + (1 - P(survive)) * floor
#
# Where floor = minimum tracking sensitivity (~12 dB-Hz)

FLOOR_DBHZ = 12.0

# =============================================================================
# Training survival model
# =============================================================================

# Binary target: C/N0(t+15s) > SURVIVAL_THRESHOLD_DBHZ
# Model: LogisticRegression or GradientBoostingClassifier
# Features: Same as baseline + current C/N0 relative to floor

# =============================================================================
# Evaluation
# =============================================================================

# Compare selection policies:
# 1. Persistence: max current C/N0
# 2. Elevation-weighted: max(current_C/N0 + elevation_bonus)
# 3. Survival-weighted: max(E[C/N0])
#
# Metric: Mean regret on held-out sessions (including dropout penalties)
