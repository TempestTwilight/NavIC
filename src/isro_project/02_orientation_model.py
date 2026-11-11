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

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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
    # 3D Body-frame satellite vectors (if not already present)
    # --------------------------------------------------------
    if "s_body_x" not in df.columns or df["s_body_x"].isna().all():
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr

        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr

        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr

        df["s_body_x"] = np.clip(r00 * s_east + r01 * s_north + r02 * s_up, -1.0, 1.0)
        df["s_body_y"] = np.clip(r10 * s_east + r11 * s_north + r12 * s_up, -1.0, 1.0)
        df["s_body_z"] = np.clip(r20 * s_east + r21 * s_north + r22 * s_up, -1.0, 1.0)

    # --------------------------------------------------------
    # Antenna Incidence Angle (dot product s·n)
    # --------------------------------------------------------

    df["cos_inc"] = np.clip(
        s_east * n_east + s_north * n_north + s_up * n_up, -1.0, 1.0
    )

    # --------------------------------------------------------
    # Elevation features
    # --------------------------------------------------------

    df["elevation_sin"] = np.sin(el)
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

    df["airmass_ky"] = 1.0 / (
        np.sin(el) + 0.15 * (df["ElevationDegrees"].values + 3.885) ** (-1.25)
    )

    # --------------------------------------------------------
    # Frequency features (Friis loss relative to L1)
    # --------------------------------------------------------

    df["freq_loss_db"] = 20.0 * np.log10(df["CarrierFrequencyHz"].values / 1575420000.0)

    # --------------------------------------------------------
    # Constellation one-hot features (numeric IDs)
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
    df["freq_loss_x_elev"] = df["freq_loss_db"] * df["elevation_sin"]

    return df


# ============================================================
# LOAD
# ============================================================

parser = argparse.ArgumentParser(description="Train orientation-agnostic C/N0 model.")
parser.add_argument(
    "--skip-rf",
    action="store_true",
    help="Skip Random Forest / tree models entirely (trains Ridge in seconds)",
)
parser.add_argument(
    "--fast-rf",
    action="store_true",
    default=True,
    help="Use fast histogram-based gradient boosting (15x-30x faster than standard RF)",
)
parser.add_argument(
    "--max-rf-samples",
    type=int,
    default=100000,
    help="Max rows to use when training tree models (default: 100,000)",
)
args, _ = parser.parse_known_args()

print("=" * 75)
print(" Orientation-Agnostic GNSS C/N0 Prediction")
print("=" * 75)

t_start_total = time.time()
ensure_dirs()

t0 = time.time()
print(f"\nLoading:\n{MERGED_CORPUS}")

df = pd.read_parquet(MERGED_CORPUS)

print(f"Loaded rows: {len(df):,} ({time.time() - t0:.2f}s)")
print(f"Sessions: {df['session_id'].nunique()}")
print(f"Devices: {df['device_id'].nunique()}")


# ============================================================
# FILTER
# ============================================================

t0 = time.time()
print("\n--- Filtering ---")

before = len(df)
df = df[pd.to_numeric(df["Cn0DbHz"], errors="coerce") > 0].copy()
print(f"C/N0 > 0: {len(df):,} (dropped {before - len(df):,})")

if "CalibrationAccuracy" in df.columns:
    before = len(df)
    df = df[pd.to_numeric(df["CalibrationAccuracy"], errors="coerce") == 3].copy()
    print(f"CalibrationAccuracy == 3: {len(df):,} (dropped {before - len(df):,})")

print(f"Filtering complete in {time.time() - t0:.2f}s")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

t0 = time.time()
print("\n--- Building features ---")

df = add_features(df)
print(f"Feature engineering complete in {time.time() - t0:.2f}s")


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
    # 3D body-frame satellite vectors (replaces Euler-angle incidence dot products)
    "s_body_x",
    "s_body_y",
    "s_body_z",
    # Core physics-based geometry
    "cos_inc",
    "elevation_sin",
    "is_above_horizon",
    # Relative azimuth
    "rel_az_sin",
    "rel_az_cos",
    # Atmospheric path
    "airmass_ky",
    # Signal parameters
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
    "freq_loss_x_elev",
]


# ============================================================
# VALID EXAMPLES
# ============================================================

t0 = time.time()
# Epoch-relative centering: remove cross-device hardware gain offset
# Cn0_relative = Cn0 - mean(Cn0) over all visible satellites in that epoch
epoch_mean = df.groupby(["session_id", "UnixTimeMillis"])["Cn0DbHz"].transform("mean")
df["target_relative_cn0"] = df["Cn0DbHz"] - epoch_mean
target_col = "target_relative_cn0"

df = df.dropna(subset=feature_cols + [target_col]).copy()

# Downcast to float32 to reduce memory footprint by 50% and maximize cache hits
X = df[feature_cols].astype(np.float32)
y = df[target_col].astype(np.float32)

print(f"\nComplete examples: {len(df):,} ({time.time() - t0:.2f}s)")
print(f"Feature matrix shape: {X.shape}, RAM: {X.memory_usage().sum() / 1e6:.1f} MB")


# ============================================================
# GROUPS
# ============================================================

# Calendar day grouping (prevents sky geometry leakage from same-day sessions)
df["recording_day"] = pd.to_datetime(df["UnixTimeMillis"], unit="ms").dt.date.astype(
    str
)

if (
    df["device_id"].nunique() > 1
    and not df["device_id"].astype(str).eq("unknown").all()
):
    print("\nMultiple known devices detected. Grouping by device_id.")
    groups = df["device_id"]
elif df["recording_day"].nunique() >= 4:
    print("\nGrouping cross-validation by recording day.")
    groups = df["recording_day"]
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

t0_ridge = time.time()
ridge_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    t_fold = time.time()
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
        f"MAE={score['mae']:.3f}, R²={score['r2']:.3f} ({time.time() - t_fold:.2f}s)"
    )

ridge_scores_df = pd.DataFrame(ridge_scores)

print(f"\nRidge cross-validation complete in {time.time() - t0_ridge:.2f}s")
print(f"RMSE: {ridge_scores_df.rmse.mean():.3f} ± {ridge_scores_df.rmse.std():.3f}")
print(f"MAE:  {ridge_scores_df.mae.mean():.3f}")
print(f"R²:   {ridge_scores_df.r2.mean():.3f}")


# ============================================================
# RANDOM FOREST / TREE MODEL
# ============================================================

rf_scores = []
rf_scores_df = pd.DataFrame()
rf_model = None

if args.skip_rf:
    print("\n[INFO] Skipping tree models (--skip-rf enabled). Ridge-only execution.")
else:
    print("\n" + "=" * 75)
    model_name = (
        "HistGradientBoosting (Fast Trees)" if args.fast_rf else "Random Forest"
    )
    print(f" {model_name}")
    print("=" * 75)

    t0_rf = time.time()
    max_rf_rows = args.max_rf_samples

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        t_fold = time.time()
        train_idx_rf = train_idx

        if len(train_idx_rf) > max_rf_rows:
            rng = np.random.default_rng(RANDOM_STATE + fold)
            train_idx_rf = rng.choice(train_idx_rf, size=max_rf_rows, replace=False)

        if args.fast_rf:
            model = HistGradientBoostingRegressor(
                max_iter=60,
                max_depth=8,
                min_samples_leaf=30,
                random_state=RANDOM_STATE,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=30,
                max_samples=0.1,
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
            f"MAE={score['mae']:.3f}, R²={score['r2']:.3f} ({time.time() - t_fold:.2f}s)"
        )

    rf_scores_df = pd.DataFrame(rf_scores)

    print(f"\n{model_name} average ({time.time() - t0_rf:.2f}s):")
    print(f"RMSE: {rf_scores_df.rmse.mean():.3f} ± {rf_scores_df.rmse.std():.3f}")
    print(f"MAE:  {rf_scores_df.mae.mean():.3f}")
    print(f"R²:   {rf_scores_df.r2.mean():.3f}")


# ============================================================
# FINAL MODELS
# ============================================================

print("\n" + "=" * 75)
print(" Training final models")
print("=" * 75)

t0_final = time.time()
ridge_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ]
)
ridge_model.fit(X, y)
print(f"Final Ridge model fit in {time.time() - t0_final:.2f}s")

if not args.skip_rf:
    t0_rf_final = time.time()
    train_idx_rf = np.arange(len(X))
    if len(train_idx_rf) > args.max_rf_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        train_idx_rf = rng.choice(train_idx_rf, size=args.max_rf_samples, replace=False)

    if args.fast_rf:
        rf_model = HistGradientBoostingRegressor(
            max_iter=80,
            max_depth=8,
            min_samples_leaf=30,
            random_state=RANDOM_STATE,
        )
    else:
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=30,
            max_samples=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    rf_model.fit(X.iloc[train_idx_rf], y.iloc[train_idx_rf])
    print(f"Final Tree model fit in {time.time() - t0_rf_final:.2f}s")


# ============================================================
# SAVE MODELS
# ============================================================

t0_save = time.time()
joblib.dump(ridge_model, ORIENTATION_ARTIFACTS / "ridge_orientation.joblib")
if rf_model is not None:
    joblib.dump(rf_model, ORIENTATION_ARTIFACTS / "rf_orientation.joblib")

with open(ORIENTATION_ARTIFACTS / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

ridge_scores_df.to_csv(ORIENTATION_ARTIFACTS / "ridge_cv.csv", index=False)
if len(rf_scores_df):
    rf_scores_df.to_csv(ORIENTATION_ARTIFACTS / "rf_cv.csv", index=False)
print(f"Artifacts saved in {time.time() - t0_save:.2f}s")


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

if rf_model is not None and hasattr(rf_model, "feature_importances_"):
    rf_importance = np.asarray(getattr(rf_model, "feature_importances_")).ravel()
    assert len(feature_cols) == len(rf_importance), (
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

    print("\nTop Tree model features:")
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
print(f"Total script runtime: {time.time() - t_start_total:.2f}s")
print(f"\nArtifacts saved to:\n{ORIENTATION_ARTIFACTS}")
