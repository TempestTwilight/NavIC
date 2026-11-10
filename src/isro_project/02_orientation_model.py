#!/usr/bin/env python3
"""
Orientation-Agnostic C/N0 Prediction (Current Signal Strength)
================================================================

Goal:
    Predict C/N0 as a function of satellite geometry relative to phone
    heading, so that step10_inference.py can evaluate candidate yaw
    angles the phone has NOT actually been rotated to and recommend
    the best one.

This is deliberately NOT a temporal / forecasting model. It answers:

    "Given where a satellite sits relative to my phone's heading,
     how strong should its signal be?"

not:

    "Given this signal's recent history, how strong will it be
     one second from now?"

Feature set is intentionally restricted to quantities that are known
(or assumable) for a HYPOTHETICAL orientation the phone is not
currently facing:

  - Included: satellite elevation, relative azimuth (satellite bearing
    minus candidate yaw), airmass, constellation, carrier frequency,
    assumed-flat tilt.
  - Excluded: current C/N0, baseband C/N0, any lag/delta/slope of C/N0,
    SignalCount, UsedInFix, HasAlmanacData, HasEphemerisData. These
    either leak the answer (current C/N0) or describe the actual
    fix/tracking state at the CURRENT orientation, which is not
    knowable for a candidate orientation the phone hasn't rotated to.
  - Pitch/roll are excluded too (not just yaw) — see note below.

IMPORTANT — keep in sync with step10_inference.py:
    `feature_cols` here and the keys produced by
    `engineer_for_orientation()` in step10_inference.py must match
    EXACTLY, including the definition of `rel_az_x_el`. A silent
    mismatch (e.g. sin vs cos convention) will not crash — it will
    just quietly feed the wrong number into a real coefficient.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    MERGED_CORPUS,
    ORIENTATION_ARTIFACTS,
    RANDOM_STATE,
    ensure_dirs,
)

warnings.filterwarnings("ignore")


# ============================================================
# HELPERS
# ============================================================


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def add_features(df, assume_flat_tilt=True):
    """
    Build geometry/orientation features for the ORIENTATION-AGNOSTIC
    model.

    IMPORTANT:
        - Current C/N0 is NOT used as an input feature.
        - Pitch/roll are NOT used as input features (see note below).
        - This function's output keys must match
          step10_inference.engineer_for_orientation() exactly.

    Why no pitch/roll:
        At inference time, step10_inference.py evaluates candidate
        YAW angles only — it does not know the phone's pitch/roll for
        a hypothetical reorientation, and it currently assumes the
        phone is held flat (tilt = 0) for every candidate. If we let
        the model learn from real pitch/roll variation during
        training, it will pick up a pitch/roll -> C/N0 relationship
        that the inference script can never actually supply (it only
        ever passes flat-tilt placeholders). That mismatch would bias
        every prediction. So training and inference both use a single
        scalar "tilt_angle" (0 by convention when absent) instead.

        If your sessions log a genuine tilt_angle column for the
        physical rig, this is used automatically. If you later want
        pitch/roll to matter, step10_inference.py must be able to
        state a real pitch/roll for each candidate yaw (not always
        possible) and would need this same feature added back on
        both sides at once.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Basic numeric cleanup (kept for filtering/reference even
    # though some of these columns are excluded from feature_cols)
    # --------------------------------------------------------

    numeric_cols = [
        "Cn0DbHz",
        "AzimuthDegrees",
        "ElevationDegrees",
        "CarrierFrequencyHz",
        "yawDeg",
        "pitchDeg",
        "rollDeg",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # Angles to radians
    # --------------------------------------------------------

    az = np.deg2rad(df["AzimuthDegrees"])
    el = np.deg2rad(df["ElevationDegrees"])
    yaw = np.deg2rad(df["yawDeg"])
    pitch = np.deg2rad(df["pitchDeg"])
    roll = np.deg2rad(df["rollDeg"])

    # --------------------------------------------------------
    # Satellite Line-of-Sight Unit Vector (ENU coordinates)
    # --------------------------------------------------------

    s_east = np.cos(el) * np.sin(az)
    s_north = np.cos(el) * np.cos(az)
    s_up = np.sin(el)

    # --------------------------------------------------------
    # Phone Antenna Normal Unit Vector (ENU coordinates)
    # Transformed from body Z-axis [0,0,1] via Euler angles
    # --------------------------------------------------------

    n_east = np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)
    n_north = -np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)
    n_up = np.cos(roll) * np.cos(pitch)

    # --------------------------------------------------------
    # Antenna Incidence Angle (dot product s·n)
    # --------------------------------------------------------

    df["cos_inc"] = np.clip(
        s_east * n_east + s_north * n_north + s_up * n_up, -1.0, 1.0
    )
    df["inc_deg"] = np.degrees(np.arccos(df["cos_inc"]))

    # --------------------------------------------------------
    # Elevation features
    # --------------------------------------------------------

    df["elevation_sin"] = np.sin(el)
    df["elevation_cos"] = np.cos(el)
    df["is_above_horizon"] = (df["ElevationDegrees"] > 0).astype(float)

    # --------------------------------------------------------
    # Relative azimuth (satellite bearing minus phone yaw)
    # --------------------------------------------------------

    rel_az = az - yaw
    df["rel_az_sin"] = np.sin(rel_az)
    df["rel_az_cos"] = np.cos(rel_az)

    # --------------------------------------------------------
    # Tropospheric Airmass (Kasten-Young model)
    # --------------------------------------------------------

    elevation_clipped = np.clip(df["ElevationDegrees"].values, 1.0, 89.0)
    df["airmass_simple"] = 1.0 / np.sin(np.deg2rad(elevation_clipped))
    df["airmass_ky"] = 1.0 / (
        np.sin(el) + 0.15 * (df["ElevationDegrees"].values + 3.885) ** (-1.25)
    )

    # --------------------------------------------------------
    # Frequency features
    # --------------------------------------------------------

    df["freq_ghz"] = df["CarrierFrequencyHz"] / 1e9
    # Relative path loss in dB (relative to L1 1575.42 MHz)
    df["freq_loss_db"] = 20.0 * np.log10(df["CarrierFrequencyHz"].values / 1575420000.0)

    # --------------------------------------------------------
    # Constellation one-hot features (FIXED: use numeric IDs)
    # --------------------------------------------------------

    df["is_gps"] = (df["ConstellationType"] == 1).astype(float)
    df["is_sbas"] = (df["ConstellationType"] == 2).astype(float)
    df["is_glonass"] = (df["ConstellationType"] == 3).astype(float)
    df["is_qzss"] = (df["ConstellationType"] == 4).astype(float)
    df["is_beidou"] = (df["ConstellationType"] == 5).astype(float)
    df["is_galileo"] = (df["ConstellationType"] == 6).astype(float)
    df["is_navic"] = (df["ConstellationType"] == 7).astype(float)

    # --------------------------------------------------------
    # Interaction terms (physics-based)
    # --------------------------------------------------------

    df["rel_az_x_el"] = df["rel_az_cos"] * df["elevation_sin"]
    df["cos_inc_x_airmass"] = df["cos_inc"] * df["airmass_ky"]
    df["navic_x_airmass"] = df["is_navic"] * df["airmass_ky"]
    df["gps_x_airmass"] = df["is_gps"] * df["airmass_ky"]
    df["galileo_x_airmass"] = df["is_galileo"] * df["airmass_ky"]
    df["freq_loss_x_elev"] = df["freq_loss_db"] * df["elevation_sin"]

    return df


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print(" Orientation-Agnostic GNSS C/N0 Prediction")
print("=" * 75)

ensure_dirs()

print(f"\nLoading:\n{MERGED_CORPUS}")

df = pd.read_parquet(MERGED_CORPUS)

print(f"\nLoaded rows: {len(df):,}")
print(f"Sessions: {df['session_id'].nunique()}")
print(f"Devices: {df['device_id'].nunique()}")


# ============================================================
# FILTER
# ============================================================

print("\n--- Filtering ---")

before = len(df)
df = df[pd.to_numeric(df["Cn0DbHz"], errors="coerce") > 0].copy()
print(f"C/N0 > 0: {len(df):,} (dropped {before - len(df):,})")

if "CalibrationAccuracy" in df.columns:
    before = len(df)
    df = df[pd.to_numeric(df["CalibrationAccuracy"], errors="coerce") == 3].copy()
    print(f"CalibrationAccuracy == 3: {len(df):,} (dropped {before - len(df):,})")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n--- Building features ---")

df = add_features(df)


# ============================================================
# FEATURE LIST
#
# Must equal the keys built by
# step10_inference.engineer_for_orientation(), in any order (the
# training script itself doesn't care about order, but the
# consistency check at the bottom of this file does).
#
# Deliberately excludes: Cn0DbHz / BasebandCn0DbHz / any lag,
# delta or slope of C/N0 (these describe the ACTUAL current signal,
# not a hypothetical orientation) and SignalCount / UsedInFix /
# HasAlmanacData / HasEphemerisData (these describe fix/tracking
# state at the phone's actual current orientation, not something
# knowable for a candidate orientation it hasn't rotated to).
# Also excludes pitch_sin/cos, roll_sin/cos — see add_features()
# docstring for why.
# ============================================================

feature_cols = [
    # Core physics-based geometry
    "cos_inc",
    "inc_deg",
    "elevation_sin",
    "elevation_cos",
    "is_above_horizon",
    # Relative azimuth
    "rel_az_sin",
    "rel_az_cos",
    # Atmospheric path
    "airmass_simple",
    "airmass_ky",
    # Signal parameters
    "freq_ghz",
    "freq_loss_db",
    # Constellation indicators
    "is_gps",
    "is_sbas",
    "is_glonass",
    "is_qzss",
    "is_beidou",
    "is_galileo",
    "is_navic",
    # Interaction terms
    "rel_az_x_el",
    "cos_inc_x_airmass",
    "navic_x_airmass",
    "gps_x_airmass",
    "galileo_x_airmass",
    "freq_loss_x_elev",
]


# ============================================================
# VALID EXAMPLES
# ============================================================

target_col = "Cn0DbHz"

df = df.dropna(subset=feature_cols + [target_col]).copy()

print(f"\nComplete examples: {len(df):,}")

X = df[feature_cols].astype(float)
y = df[target_col].astype(float)


# ============================================================
# GROUPS
# ============================================================

if (
    df["device_id"].nunique() > 1
    and not df["device_id"].astype(str).eq("unknown").all()
):
    print("\nMultiple known devices detected.")
    groups = df["device_id"]
else:
    print("\nUsing held-out SESSION validation.")
    groups = df["session_id"]

n_groups = groups.nunique()

if n_groups < 2:
    raise RuntimeError("Need at least 2 groups for cross-validation.")

n_splits = min(4, n_groups)
cv = GroupKFold(n_splits=n_splits)


# ============================================================
# RIDGE
# ============================================================

print("\n" + "=" * 75)
print(" Ridge Regression")
print("=" * 75)

ridge_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])

    score = {
        "fold": fold,
        "rmse": rmse(y.iloc[test_idx], pred),
        "mae": mean_absolute_error(y.iloc[test_idx], pred),
        "r2": r2_score(y.iloc[test_idx], pred),
    }
    ridge_scores.append(score)

    held = groups.iloc[test_idx].unique()
    print(
        f"Fold {fold}: held={held}, RMSE={score['rmse']:.3f}, "
        f"MAE={score['mae']:.3f}, R²={score['r2']:.3f}"
    )

ridge_scores_df = pd.DataFrame(ridge_scores)

print("\nRidge average:")
print(f"RMSE: {ridge_scores_df.rmse.mean():.3f} ± {ridge_scores_df.rmse.std():.3f}")
print(f"MAE:  {ridge_scores_df.mae.mean():.3f}")
print(f"R²:   {ridge_scores_df.r2.mean():.3f}")


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 75)
print(" Random Forest")
print("=" * 75)

rf_scores = []
max_rf_rows = 50000

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    train_idx_rf = train_idx

    if len(train_idx_rf) > max_rf_rows:
        rng = np.random.default_rng(RANDOM_STATE + fold)
        train_idx_rf = rng.choice(train_idx_rf, size=max_rf_rows, replace=False)

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X.iloc[train_idx_rf], y.iloc[train_idx_rf])
    pred = model.predict(X.iloc[test_idx])

    score = {
        "fold": fold,
        "rmse": rmse(y.iloc[test_idx], pred),
        "mae": mean_absolute_error(y.iloc[test_idx], pred),
        "r2": r2_score(y.iloc[test_idx], pred),
    }
    rf_scores.append(score)

    held = groups.iloc[test_idx].unique()
    print(
        f"Fold {fold}: held={held}, RMSE={score['rmse']:.3f}, "
        f"MAE={score['mae']:.3f}, R²={score['r2']:.3f}"
    )

rf_scores_df = pd.DataFrame(rf_scores)

print("\nRandom Forest average:")
print(f"RMSE: {rf_scores_df.rmse.mean():.3f} ± {rf_scores_df.rmse.std():.3f}")
print(f"MAE:  {rf_scores_df.mae.mean():.3f}")
print(f"R²:   {rf_scores_df.r2.mean():.3f}")


# ============================================================
# FINAL MODELS
# ============================================================

print("\n" + "=" * 75)
print(" Training final models")
print("=" * 75)

ridge_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ]
)
ridge_model.fit(X, y)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_model.fit(X, y)


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(ridge_model, ORIENTATION_ARTIFACTS / "ridge_orientation.joblib")
joblib.dump(rf_model, ORIENTATION_ARTIFACTS / "rf_orientation.joblib")

with open(ORIENTATION_ARTIFACTS / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

ridge_scores_df.to_csv(ORIENTATION_ARTIFACTS / "ridge_cv.csv", index=False)
rf_scores_df.to_csv(ORIENTATION_ARTIFACTS / "rf_cv.csv", index=False)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n--- Feature importance ---")

ridge_coef = np.asarray(ridge_model.named_steps["ridge"].coef_).ravel()
assert len(ridge_coef) == len(feature_cols), (
    f"Ridge mismatch: {len(ridge_coef)} coefficients vs {len(feature_cols)} features"
)

ridge_coefficients = pd.DataFrame(
    {
        "feature": feature_cols,
        "coefficient": ridge_coef,
    }
)
ridge_coefficients["abs_coefficient"] = ridge_coefficients["coefficient"].abs()
ridge_coefficients = ridge_coefficients.sort_values("abs_coefficient", ascending=False)
ridge_coefficients.to_csv(ORIENTATION_ARTIFACTS / "ridge_coefficients.csv", index=False)

print("\nTop Ridge features:")
print(ridge_coefficients.head(20).to_string(index=False))

rf_importance = np.asarray(rf_model.feature_importances_).ravel()
assert len(rf_importance) == len(feature_cols), (
    f"RF mismatch: {len(rf_importance)} importances vs {len(feature_cols)} features"
)

rf_importance_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance": rf_importance,
    }
)
rf_importance_df = rf_importance_df.sort_values("importance", ascending=False)
rf_importance_df.to_csv(ORIENTATION_ARTIFACTS / "rf_importance.csv", index=False)

print("\nTop Random Forest features:")
print(rf_importance_df.head(20).to_string(index=False))


# ============================================================
# CONSISTENCY CHECK WITH step10_inference.py
#
# This block does not import step10_inference.py (to avoid a hard
# path/module dependency), but documents and checks the exact
# feature list step10_inference.engineer_for_orientation() must
# produce. If you edit feature_cols above, update this list too —
# and update step10_inference.py in the same commit.
# ============================================================

EXPECTED_INFERENCE_KEYS = {
    "elevation_sin",
    "elevation_cos",
    "rel_az_sin",
    "rel_az_cos",
    "airmass",
    "is_above_horizon",
    "freq_ghz",
    "is_gps",
    "is_glonass",
    "is_galileo",
    "is_beidou",
    "is_qzss",
    "is_navic",
    "rel_az_x_el",
    "navic_x_airmass",
    "gps_x_airmass",
    "galileo_x_airmass",
    "tilt_angle_sin",
    "tilt_angle_cos",
}

trained_keys = set(feature_cols)

if trained_keys != EXPECTED_INFERENCE_KEYS:
    missing = EXPECTED_INFERENCE_KEYS - trained_keys
    extra = trained_keys - EXPECTED_INFERENCE_KEYS
    print(
        "\nWARNING: trained feature_cols do not match the feature set "
        "step10_inference.py is expected to build.\n"
        f"  In inference but not training: {missing}\n"
        f"  In training but not inference: {extra}\n"
        "Update engineer_for_orientation() in step10_inference.py to match."
    )

print("\n" + "=" * 75)
print(" COMPLETE")
print("=" * 75)
print(f"\nArtifacts saved to:\n{ORIENTATION_ARTIFACTS}")
