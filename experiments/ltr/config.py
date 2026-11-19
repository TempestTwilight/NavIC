"""
Experiment C: Learning-to-Rank (LTR) Formulation

Goal: Directly optimize ranking metrics (NDCG@4, MRR) instead of regression MSE
"""


# Config definitions

# =============================================================================
# LTR Configuration
# =============================================================================

# Model: LightGBM LambdaRank
LTR_MODEL = "lambdarank"

# Ranking objective
RANKING_OBJECTIVE = "ndcg"  # Options: "ndcg", "map", "rank_xendcg"

# Top-k for NDCG computation
TOP_K = 4  # Typical number of satellites used in PVT

# =============================================================================
# Grouping
# =============================================================================

# Group by (session, epoch) for ranking
# Each group contains all satellites visible at that epoch
#
# Query: SELECT * FROM features
#        GROUP BY session, sec_epoch
#        ORDER BY future_C/N0 DESC within group

# =============================================================================
# Target for ranking
# =============================================================================

# Option 1: Absolute future C/N0 (raw)
# Rank by: C/N0(t+15s)

# Option 2: Relative future C/N0 (within epoch)
# Rank by: C/N0(t+15s) - epoch_median_future_C/N0

# Option 3: Utility (expected value with survival)
# Rank by: E[C/N0] = P(survive) * C/N0(t+15s) + (1-P) * floor

LTR_TARGET = "absolute"  # Options: "absolute", "relative", "utility"

# =============================================================================
# Training parameters
# =============================================================================

# Number of leaves
NUM_LEAVES = 31

# Learning rate
LEARNING_RATE = 0.05

# Number of estimators
N_ESTIMATORS = 100

# Early stopping rounds
EARLY_STOPPING_ROUNDS = 20

# =============================================================================
# Evaluation
# =============================================================================

# Metrics:
# - NDCG@4 (normalized discounted cumulative gain)
# - MRR (mean reciprocal rank of best satellite)
# - Hit rate (fraction of epochs where top-1 matches ground truth)

# Baseline:
# - Persistence ranking (rank by current C/N0)
# - Random ranking
