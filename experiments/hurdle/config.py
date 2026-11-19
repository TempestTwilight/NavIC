"""
Experiment B: Two-Stage Hurdle Model Configuration

Stage 1: Binary classifier for fading detection
Stage 2: Specialized regressor for fading magnitude
"""


# Config definitions

# =============================================================================
# Stage 1: Fading Classifier Configuration
# =============================================================================

# Binary target: |Δ C/N0| > 1.5 dB
FADING_THRESHOLD = 1.5  # dB

# Model options
STAGE1_MODEL = "logistic"  # Options: "logistic", "lightgbm", "svm"

# Class weight (fading is ~16% of epochs)
# Use 'balanced' or compute manually:
# n_total / (2 * n_fading) for fading class
# n_total / (2 * n_static) for static class
STAGE1_CLASS_WEIGHT = "balanced"

# Probability threshold for triggering Stage 2
# Lower = more false positives, higher = more false negatives
STAGE1_THRESHOLD = 0.5

# =============================================================================
# Stage 2: Fading Regressor Configuration
# =============================================================================

# Train ONLY on fading epochs (|Δ| > 1.5 dB)
# This prevents the model from learning the trivial "predict 0" solution

# Model options
STAGE2_MODEL = "ridge"  # Options: "ridge", "huber", "lightgbm"

# Robust loss for outlier fading events
STAGE2_USE_ROBUST = True

# =============================================================================
# Combined Inference
# =============================================================================

# For each sample:
# 1. Compute P(fading) from Stage 1
# 2. If P(fading) > threshold: predict Δ from Stage 2
# 3. Else: predict Δ = 0

# =============================================================================
# Evaluation Metrics
# =============================================================================

# Stage 1 metrics:
# - AUC (area under ROC)
# - Precision@k (top-k highest risk)
# - Recall (fraction of fading events detected)
# - Calibration (Brier score)

# Stage 2 metrics:
# - RMSE on fading epochs only
# - MAE on fading epochs only
# - R² on fading epochs only

# Combined metrics:
# - Overall RMSE
# - Mean regret (dB)
# - Hit rate within 0.5 dB
