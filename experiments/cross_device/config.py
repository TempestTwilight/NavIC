"""
Experiment D: Leave-One-Device-Out Generalization

Goal: Test whether models trained on devices A, B, C can predict on device D
"""


# Config definitions

# =============================================================================
# Experiment Configuration
# =============================================================================

# Test each device as held-out
LODO_DEVICES = [
    (
        "test_on_samsung",
        ["OnePlus CPH2401", "OnePlus CPH2487", "OPPO CPH2325"],
        "samsung SM-M146B",
    ),
    (
        "test_on_oneplus_2401",
        ["OnePlus CPH2487", "OPPO CPH2325", "samsung SM-M146B"],
        "OnePlus CPH2401",
    ),
    (
        "test_on_oneplus_2487",
        ["OnePlus CPH2401", "OPPO CPH2325", "samsung SM-M146B"],
        "OnePlus CPH2487",
    ),
    (
        "test_on_oppo",
        ["OnePlus CPH2401", "OnePlus CPH2487", "samsung SM-M146B"],
        "OPPO CPH2325",
    ),
]

# Feature normalization for cross-device transfer
# Option 1: Per-device bias terms (learned from training devices)
# Option 2: C/N0 relative to per-device baseline (requires calibration)
# Option 3: Epoch-centered targets (device-invariant by construction)

NORMALIZATION_STRATEGIES = [
    "none",  # Raw C/N0, no device features
    "device_bias",  # One-hot device_id + learnable bias per device
    "device_baseline",  # C/N0 - device_median(elevation_band)
    "epoch_centered",  # Target = Δ C/N0 (epoch mean removed)
]
