#!/usr/bin/env python3
"""
Temporal GNSS C/N0 Prediction
==============================

Dataset:
    planning/combined_corpus.parquet

Sampling:
    ~1 second per GNSS epoch

Prediction task:

    Information available at time t
                    ↓
             ML prediction
                    ↓
        C/N0 at approximately t+1s


The model predicts future C/N0 for each currently observed signal.

The predicted values can then be used to select the signal expected
to have the strongest C/N0 one second in the future.


Important:
    signal_id is used ONLY to construct temporal history.
    It is NOT used as an ML feature.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    MERGED_CORPUS,
    BASELINE_ARTIFACTS,
    RANDOM_STATE,
    HORIZON_MS,
    ensure_dirs,
)

warnings.filterwarnings("ignore")


# =====================================================================
# CONFIGURATION
# =====================================================================

# Your data shows ~1000 ms epoch spacing.
EPOCH_MS = 1000

# Predict one epoch into the future.
HORIZON_EPOCHS = 1

# Allowed timestamp error when finding lag/target measurements.
#
# Your epoch interval is ~1000 ms, so 150 ms is conservative.
TIME_TOLERANCE_MS = 150


# Cross-validation
N_SPLITS = 4


# RF training sample
RF_SAMPLE_PER_SESSION = 15000


# Random Forest
RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 8,
    "min_samples_leaf": 30,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# =====================================================================
# SIGNAL IDENTITY
# =====================================================================

SIGNAL_KEYS = [
    "device_id",
    "session_id",
    "ConstellationType",
    "Svid",
    "CarrierFrequencyHz",
]


# =====================================================================
# LOAD
# =====================================================================

print("=" * 75)
print(" Temporal GNSS C/N0 Prediction")
print("=" * 75)

ensure_dirs()

print("\nLoading:")

print(MERGED_CORPUS)

df = pd.read_parquet(MERGED_CORPUS)

print(f"\nLoaded rows: {len(df):,}")

print(f"Sessions: {df['session_id'].nunique()}")

print(f"Devices: {df['device_id'].nunique()}")


# =====================================================================
# CHECK DEVICE IDS
# =====================================================================

print("\nDevice IDs:")

print(df["device_id"].value_counts())


if df["device_id"].nunique() == 1 and df["device_id"].iloc[0] == "unknown":
    print("\nWARNING:")

    print("All rows have device_id='unknown'.")

    print("Device-held-out validation will therefore NOT test hardware generalization.")


# =====================================================================
# FILTER
# =====================================================================

print("\n--- Filtering ---")


# C/N0
before = len(df)

df = df[df["Cn0DbHz"] > 0].copy()

print(f"C/N0 > 0: {len(df):,} (dropped {before - len(df):,})")


# Calibration
if "CalibrationAccuracy" in df.columns:
    before = len(df)

    df = df[df["CalibrationAccuracy"] == 3].copy()

    print(f"CalibrationAccuracy == 3: {len(df):,} (dropped {before - len(df):,})")


# Orientation
before = len(df)

df = df[df["yawDeg"].notna() & df["pitchDeg"].notna() & df["rollDeg"].notna()].copy()

print(f"Valid orientation: {len(df):,} (dropped {before - len(df):,})")


# Orientation synchronization
if "orientation_dt_ms" in df.columns:
    before = len(df)

    df = df[df["orientation_dt_ms"].abs() <= 100].copy()

    print(f"Orientation |dt| <= 100 ms: {len(df):,} (dropped {before - len(df):,})")


# =====================================================================
# SORT
# =====================================================================

print("\n--- Sorting signal tracks ---")

df = df.sort_values(SIGNAL_KEYS + ["UnixTimeMillis"]).reset_index(drop=True)


# =====================================================================
# CREATE UNIQUE ROW ID
# =====================================================================

df["_row_id"] = np.arange(len(df))


# =====================================================================
# TEMPORAL FEATURES
# =====================================================================

print("\n--- Building temporal features ---")


# ---------------------------------------------------------------------
# IMPORTANT
#
# We now explicitly use timestamp-based temporal matching.
#
# For example:
#
#     t = 10,000 ms
#
# lag_1 target time = 9,000 ms
#
# lag_5 target time = 5,000 ms
#
# This is safer than blindly assuming shift(1) = one second.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Source table for historical C/N0
# ---------------------------------------------------------------------

history = df[
    SIGNAL_KEYS
    + [
        "UnixTimeMillis",
        "Cn0DbHz",
        "ElevationDegrees",
        "AzimuthDegrees",
    ]
].copy()


history = history.rename(
    columns={
        "UnixTimeMillis": "history_time",
        "Cn0DbHz": "history_cn0",
        "ElevationDegrees": "history_elevation",
        "AzimuthDegrees": "history_azimuth",
    }
)


history = history.sort_values(SIGNAL_KEYS + ["history_time"])


# ---------------------------------------------------------------------
# Helper for time-based historical matching
# ---------------------------------------------------------------------


def add_history_feature(
    base,
    history,
    lag_ms,
    value_column,
    output_column,
):
    """
    Find the measurement of the same signal approximately lag_ms
    milliseconds in the past.
    """

    left = base[
        [
            "_row_id",
            *SIGNAL_KEYS,
            "UnixTimeMillis",
        ]
    ].copy()

    left["query_time"] = left["UnixTimeMillis"] - lag_ms

    left = left.sort_values(["query_time"] + SIGNAL_KEYS)

    right = history[
        [
            *SIGNAL_KEYS,
            "history_time",
            value_column,
        ]
    ].copy()

    right = right.sort_values(["history_time"] + SIGNAL_KEYS)

    result = pd.merge_asof(
        left,
        right,
        left_on="query_time",
        right_on="history_time",
        by=SIGNAL_KEYS,
        tolerance=TIME_TOLERANCE_MS,
        direction="nearest",
    )

    result = result[
        [
            "_row_id",
            value_column,
        ]
    ].rename(columns={value_column: output_column})

    return base.merge(
        result,
        on="_row_id",
        how="left",
    )


# ---------------------------------------------------------------------
# C/N0 history
# ---------------------------------------------------------------------

for seconds in [1, 2, 5, 10]:
    print(f"  Adding C/N0 lag {seconds}s...")

    df = add_history_feature(
        df,
        history,
        lag_ms=seconds * EPOCH_MS,
        value_column="history_cn0",
        output_column=f"cn0_lag_{seconds}s",
    )


# ---------------------------------------------------------------------
# Elevation history
# ---------------------------------------------------------------------

for seconds in [1, 5]:
    print(f"  Adding elevation lag {seconds}s...")

    df = add_history_feature(
        df,
        history,
        lag_ms=seconds * EPOCH_MS,
        value_column="history_elevation",
        output_column=f"elevation_lag_{seconds}s",
    )


# ---------------------------------------------------------------------
# Azimuth history
# ---------------------------------------------------------------------

df = add_history_feature(
    df,
    history,
    lag_ms=EPOCH_MS,
    value_column="history_azimuth",
    output_column="azimuth_lag_1s",
)


# =====================================================================
# TEMPORAL DERIVATIVES
# =====================================================================

print("\n--- Building temporal derivatives ---")


# C/N0 change

df["cn0_delta_1s"] = df["Cn0DbHz"] - df["cn0_lag_1s"]


df["cn0_delta_5s"] = df["Cn0DbHz"] - df["cn0_lag_5s"]


df["cn0_delta_10s"] = df["Cn0DbHz"] - df["cn0_lag_10s"]


# C/N0 slopes

df["cn0_slope_1s"] = df["cn0_delta_1s"]


df["cn0_slope_5s"] = df["cn0_delta_5s"] / 5.0


df["cn0_slope_10s"] = df["cn0_delta_10s"] / 10.0


# Elevation changes

df["elevation_delta_1s"] = df["ElevationDegrees"] - df["elevation_lag_1s"]


df["elevation_delta_5s"] = df["ElevationDegrees"] - df["elevation_lag_5s"]


# Circular azimuth change

df["azimuth_delta_1s"] = (df["AzimuthDegrees"] - df["azimuth_lag_1s"] + 180) % 360 - 180


# =====================================================================
# FUTURE TARGET
# =====================================================================

print("\n--- Constructing t -> t+1 second target ---")


# ---------------------------------------------------------------------
# Future C/N0 source
# ---------------------------------------------------------------------

future = df[
    SIGNAL_KEYS
    + [
        "UnixTimeMillis",
        "Cn0DbHz",
    ]
].copy()


future = future.rename(
    columns={
        "UnixTimeMillis": "future_time",
        "Cn0DbHz": "target_cn0",
    }
)


future = future.sort_values(["future_time"] + SIGNAL_KEYS)


# ---------------------------------------------------------------------
# Query exactly one second ahead
# ---------------------------------------------------------------------

target_left = df[
    [
        "_row_id",
        *SIGNAL_KEYS,
        "UnixTimeMillis",
    ]
].copy()


target_left["desired_future_time"] = target_left["UnixTimeMillis"] + HORIZON_MS


target_left = target_left.sort_values(["desired_future_time"] + SIGNAL_KEYS)


# ---------------------------------------------------------------------
# Match future observation
#
# direction="nearest" is safe here because the desired target time is
# already t+1 second and tolerance is only 150 ms.
# ---------------------------------------------------------------------

target_match = pd.merge_asof(
    target_left,
    future,
    left_on="desired_future_time",
    right_on="future_time",
    by=SIGNAL_KEYS,
    tolerance=TIME_TOLERANCE_MS,
    direction="nearest",
)


target_match = target_match[
    [
        "_row_id",
        "future_time",
        "target_cn0",
    ]
]


df = df.merge(
    target_match,
    on="_row_id",
    how="left",
)


# =====================================================================
# TARGET QUALITY
# =====================================================================

print("\n--- Target quality ---")


valid_target = df["target_cn0"].notna()


print(f"Rows with t+1s target: {valid_target.sum():,}")

print(f"Rows without target: {(~valid_target).sum():,}")


# Target must be positive
df = df[df["target_cn0"] > 0].copy()


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================

print("\n--- Building model features ---")


features = pd.DataFrame(index=df.index)


# ---------------------------------------------------------------------
# Satellite geometry
# ---------------------------------------------------------------------

elev_rad = np.radians(df["ElevationDegrees"].values)

az_rad = np.radians(df["AzimuthDegrees"].values)

yaw_rad = np.radians(df["yawDeg"].values)


# ---------------------------------------------------------------------
# Relative azimuth
# ---------------------------------------------------------------------

relative_azimuth = (df["AzimuthDegrees"].values - df["yawDeg"].values) % 360


relative_azimuth_rad = np.radians(relative_azimuth)


features["elevation_sin"] = np.sin(elev_rad)

features["elevation_cos"] = np.cos(elev_rad)


features["rel_az_sin"] = np.sin(relative_azimuth_rad)

features["rel_az_cos"] = np.cos(relative_azimuth_rad)


# ---------------------------------------------------------------------
# Airmass
# ---------------------------------------------------------------------

safe_elevation = np.maximum(elev_rad, np.radians(1))


features["airmass"] = 1.0 / np.sin(safe_elevation)


# ---------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------

features["is_above_horizon"] = (df["ElevationDegrees"].values > 5).astype(int)


# ---------------------------------------------------------------------
# Constellation
# ---------------------------------------------------------------------

constellations = {
    1: "gps",
    3: "glonass",
    4: "qzss",
    5: "beidou",
    6: "galileo",
    7: "navic",
}


for constellation, name in constellations.items():
    features[f"is_{name}"] = (df["ConstellationType"].values == constellation).astype(
        int
    )


# ---------------------------------------------------------------------
# Frequency
# ---------------------------------------------------------------------

features["freq_ghz"] = df["CarrierFrequencyHz"].values / 1e9


# ---------------------------------------------------------------------
# Phone orientation
#
# Absolute yaw is NOT used.
#
# Relative azimuth already captures satellite direction relative
# to the phone.
# ---------------------------------------------------------------------

pitch_rad = np.radians(df["pitchDeg"].values)

roll_rad = np.radians(df["rollDeg"].values)


features["pitch_sin"] = np.sin(pitch_rad)

features["pitch_cos"] = np.cos(pitch_rad)

features["roll_sin"] = np.sin(roll_rad)

features["roll_cos"] = np.cos(roll_rad)


# ---------------------------------------------------------------------
# Experimental session tilt
# ---------------------------------------------------------------------

if "tilt_angle" in df.columns:
    tilt_rad = np.radians(df["tilt_angle"].values)

    features["tilt_angle_sin"] = np.sin(tilt_rad)

    features["tilt_angle_cos"] = np.cos(tilt_rad)


# ---------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------

features["rel_az_x_el"] = features["rel_az_sin"] * features["elevation_sin"]


features["navic_x_airmass"] = features["is_navic"] * features["airmass"]


# =====================================================================
# CURRENT GNSS STATE
# =====================================================================

features["cn0_current"] = df["Cn0DbHz"].values


# Baseband C/N0, if available

if "BasebandCn0DbHz" in df.columns:
    features["baseband_cn0"] = df["BasebandCn0DbHz"].values


# Signal state

features["used_in_fix"] = df["UsedInFix"].fillna(0).values

features["has_almanac"] = df["HasAlmanacData"].fillna(0).values

features["has_ephemeris"] = df["HasEphemerisData"].fillna(0).values


# Number of signals visible at epoch

features["signal_count"] = df["SignalCount"].values


# =====================================================================
# TEMPORAL FEATURES
# =====================================================================

features["cn0_lag_1s"] = df["cn0_lag_1s"].values

features["cn0_lag_2s"] = df["cn0_lag_2s"].values

features["cn0_lag_5s"] = df["cn0_lag_5s"].values

features["cn0_lag_10s"] = df["cn0_lag_10s"].values


features["cn0_delta_1s"] = df["cn0_delta_1s"].values

features["cn0_delta_5s"] = df["cn0_delta_5s"].values

features["cn0_delta_10s"] = df["cn0_delta_10s"].values


features["cn0_slope_1s"] = df["cn0_slope_1s"].values

features["cn0_slope_5s"] = df["cn0_slope_5s"].values

features["cn0_slope_10s"] = df["cn0_slope_10s"].values


# Geometry dynamics

features["elevation_delta_1s"] = df["elevation_delta_1s"].values

features["elevation_delta_5s"] = df["elevation_delta_5s"].values

features["azimuth_delta_1s"] = df["azimuth_delta_1s"].values


# =====================================================================
# COMBINE
# =====================================================================

feature_cols = list(features.columns)


model_df = pd.concat(
    [
        df.reset_index(drop=True),
        features.reset_index(drop=True),
    ],
    axis=1,
)


model_df = model_df.replace([np.inf, -np.inf], np.nan)


# Drop rows without temporal history.

before = len(model_df)


model_df = model_df.dropna(subset=feature_cols + ["target_cn0"]).copy()


print(f"\nComplete temporal examples: {len(model_df):,}")

print(f"Dropped: {before - len(model_df):,}")


# =====================================================================
# MODEL ARRAYS
# =====================================================================

X = model_df[feature_cols].values.astype(np.float64)


y = model_df["target_cn0"].values.astype(np.float64)


groups = model_df["device_id"].astype(str).values


# =====================================================================
# METRIC HELPER
# =====================================================================


def calculate_metrics(actual, predicted):

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    mae = mean_absolute_error(actual, predicted)

    r2 = r2_score(actual, predicted)

    return rmse, mae, r2


# =====================================================================
# CROSS VALIDATION
# =====================================================================

print("\n" + "=" * 75)
print(" Cross Validation")
print("=" * 75)


# =====================================================================
# IMPORTANT:
#
# If only one device exists, device GroupKFold is impossible.
#
# In that case we use session-held-out CV instead.
# =====================================================================

if model_df["device_id"].nunique() >= 2:
    cv_groups = model_df["device_id"].astype(str).values

    cv_group_name = "device"

else:
    print("\nOnly one device detected.")

    print("Using held-out SESSION validation.")

    cv_groups = model_df["session_id"].astype(str).values

    cv_group_name = "session"


n_groups = len(np.unique(cv_groups))


n_splits = min(N_SPLITS, n_groups)


gkf = GroupKFold(n_splits=n_splits)


# =====================================================================
# RIDGE
# =====================================================================

print("\n--- Ridge Regression ---")


ridge_scores = []


for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, cv_groups), start=1):
    X_train = X[train_idx]

    X_val = X[val_idx]

    y_train = y[train_idx]

    y_val = y[val_idx]

    held_out = np.unique(cv_groups[val_idx])

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )

    ridge.fit(X_train, y_train)

    prediction = ridge.predict(X_val)

    rmse, mae, r2 = calculate_metrics(y_val, prediction)

    ridge_scores.append((rmse, mae, r2))

    print(f"Fold {fold}: held={held_out}, RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")


print("\nRidge average:")

print(
    f"RMSE: "
    f"{np.mean([x[0] for x in ridge_scores]):.3f}"
    f" ± "
    f"{np.std([x[0] for x in ridge_scores]):.3f}"
)

print(f"MAE:  {np.mean([x[1] for x in ridge_scores]):.3f}")

print(f"R²:   {np.mean([x[2] for x in ridge_scores]):.3f}")


# =====================================================================
# RANDOM FOREST
# =====================================================================

print("\n--- Random Forest ---")


np.random.seed(42)


sample_indices = []


for session in model_df["session_id"].unique():
    idx = np.where(model_df["session_id"].values == session)[0]

    n = min(RF_SAMPLE_PER_SESSION, len(idx))

    chosen = np.random.choice(idx, size=n, replace=False)

    sample_indices.extend(chosen)


sample_indices = np.array(sample_indices)


X_rf = X[sample_indices]

y_rf = y[sample_indices]

groups_rf = model_df.iloc[sample_indices]["device_id"].astype(str).values


# If one device, use session groups.

if len(np.unique(groups_rf)) < 2:
    groups_rf = model_df.iloc[sample_indices]["session_id"].astype(str).values


print(f"RF training rows: {len(X_rf):,}")


rf_scores = []


rf_gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups_rf))))


for fold, (train_idx, val_idx) in enumerate(
    rf_gkf.split(X_rf, y_rf, groups_rf), start=1
):
    model = RandomForestRegressor(**RF_PARAMS)

    model.fit(X_rf[train_idx], y_rf[train_idx])

    prediction = model.predict(X_rf[val_idx])

    rmse, mae, r2 = calculate_metrics(y_rf[val_idx], prediction)

    rf_scores.append((rmse, mae, r2))

    held_out = np.unique(groups_rf[val_idx])

    print(f"Fold {fold}: held={held_out}, RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")


print("\nRandom Forest average:")

print(
    f"RMSE: "
    f"{np.mean([x[0] for x in rf_scores]):.3f}"
    f" ± "
    f"{np.std([x[0] for x in rf_scores]):.3f}"
)

print(f"MAE:  {np.mean([x[1] for x in rf_scores]):.3f}")

print(f"R²:   {np.mean([x[2] for x in rf_scores]):.3f}")


# =====================================================================
# TRAIN FINAL MODELS
# =====================================================================

print("\n--- Training final models ---")


ridge_final = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ]
)


ridge_final.fit(X, y)


rf_final = RandomForestRegressor(**RF_PARAMS)


rf_final.fit(X_rf, y_rf)


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================
#
# FIX (was broken):
#   - `ridge_model` / `rf_model` were never defined anywhere in the
#     original script. The fitted objects are `ridge_final` (a
#     Pipeline) and `rf_final` (a RandomForestRegressor).
#   - `ARTIFACT_DIR` was never defined. The correct output location
#     is `BASELINE_ARTIFACTS`.
#   - Ridge coefficients live on the inner "ridge" step of the
#     pipeline, not on the pipeline object itself.
# =====================================================================

print("\n--- Feature importance ---")

# ============================================================
# RIDGE COEFFICIENTS
# ============================================================

ridge_coef = np.asarray(ridge_final.named_steps["ridge"].coef_).ravel()

# Get the actual number of input features used by Ridge
n_ridge_features = len(ridge_coef)

# First try to use the feature columns directly
if len(feature_cols) == n_ridge_features:
    ridge_feature_names = list(feature_cols)
else:
    print(
        f"WARNING: feature_cols has {len(feature_cols)} names "
        f"but Ridge has {n_ridge_features} coefficients."
    )

    # If the model is a Pipeline, inspect the preprocessing step
    ridge_feature_names = None

    if hasattr(ridge_final, "named_steps"):
        print("Ridge pipeline steps:", list(ridge_final.named_steps.keys()))

        # If there is a ColumnTransformer, recover transformed names
        for step_name, step in ridge_final.named_steps.items():
            if hasattr(step, "get_feature_names_out"):
                try:
                    ridge_feature_names = list(step.get_feature_names_out(feature_cols))
                    print(
                        f"Recovered {len(ridge_feature_names)} "
                        f"transformed feature names from '{step_name}'."
                    )
                    break
                except Exception:
                    pass

    # Final fallback: guarantee equal lengths
    if ridge_feature_names is None:
        ridge_feature_names = [f"feature_{i}" for i in range(n_ridge_features)]

# Absolute coefficient magnitude
ridge_coefficients = pd.DataFrame(
    {
        "feature": ridge_feature_names,
        "coefficient": ridge_coef,
    }
)

ridge_coefficients["abs_coefficient"] = ridge_coefficients["coefficient"].abs()

ridge_coefficients = ridge_coefficients.sort_values("abs_coefficient", ascending=False)

ridge_coefficients.to_csv(BASELINE_ARTIFACTS / "ridge_coefficients.csv", index=False)

print("\nTop Ridge coefficients:")
print(ridge_coefficients.head(20).to_string(index=False))


# ============================================================
# RANDOM FOREST IMPORTANCE
# ============================================================

rf_importance_values = np.asarray(rf_final.feature_importances_).ravel()

n_rf_features = len(rf_importance_values)

if len(feature_cols) == n_rf_features:
    rf_feature_names = list(feature_cols)
else:
    print(
        f"WARNING: feature_cols has {len(feature_cols)} names "
        f"but RF has {n_rf_features} importances."
    )

    rf_feature_names = [f"feature_{i}" for i in range(n_rf_features)]

rf_importance_df = pd.DataFrame(
    {
        "feature": rf_feature_names,
        "importance": rf_importance_values,
    }
)

rf_importance_df = rf_importance_df.sort_values("importance", ascending=False)

rf_importance_df.to_csv(BASELINE_ARTIFACTS / "rf_importance.csv", index=False)

print("\nTop Random Forest features:")
print(rf_importance_df.head(20).to_string(index=False))

# =====================================================================
# SIGNAL SELECTION
# =====================================================================

print("\n" + "=" * 75)

print(" Signal Selection Evaluation")

print("=" * 75)


# ---------------------------------------------------------------------
# Generate predictions for all examples.
# ---------------------------------------------------------------------

model_df["predicted_cn0"] = ridge_final.predict(X)


# ---------------------------------------------------------------------
# Current strongest signal baseline
#
# This is the strongest signal RIGHT NOW.
#
# We ask:
#
#     How strong is that same signal one second later?
#
# This is our baseline.
# ---------------------------------------------------------------------

model_df["baseline_score"] = model_df["Cn0DbHz"]


# ---------------------------------------------------------------------
# Epoch grouping
#
# We evaluate among signals available at the SAME current epoch.
# ---------------------------------------------------------------------

selection_groups = model_df.groupby(
    [
        "device_id",
        "session_id",
        "UnixTimeMillis",
    ],
    sort=False,
)


selection_results = []


for (device, session, epoch), group in selection_groups:
    if len(group) < 2:
        continue

    # ================================================================
    # ACTUAL BEST FUTURE SIGNAL
    # ================================================================

    actual_best_idx = group["target_cn0"].idxmax()

    actual_best_cn0 = group.loc[actual_best_idx, "target_cn0"]

    # ================================================================
    # ML CHOICE
    # ================================================================

    ml_idx = group["predicted_cn0"].idxmax()

    ml_actual_cn0 = group.loc[ml_idx, "target_cn0"]

    # ================================================================
    # BASELINE CHOICE
    # ================================================================

    baseline_idx = group["baseline_score"].idxmax()

    baseline_actual_cn0 = group.loc[baseline_idx, "target_cn0"]

    # ================================================================
    # REGRET
    # ================================================================

    ml_regret = actual_best_cn0 - ml_actual_cn0

    baseline_regret = actual_best_cn0 - baseline_actual_cn0

    # ================================================================
    # TOP-1
    # ================================================================

    ml_top1 = ml_idx == actual_best_idx

    baseline_top1 = baseline_idx == actual_best_idx

    # ================================================================
    # TOP-3
    # ================================================================

    top3_indices = group.sort_values("predicted_cn0", ascending=False).head(3).index

    ml_top3 = actual_best_idx in top3_indices

    selection_results.append(
        {
            "device_id": device,
            "session_id": session,
            "UnixTimeMillis": epoch,
            "actual_best_cn0": actual_best_cn0,
            "ml_selected_cn0": ml_actual_cn0,
            "baseline_selected_cn0": baseline_actual_cn0,
            "ml_top1": ml_top1,
            "ml_top3": ml_top3,
            "baseline_top1": baseline_top1,
            "ml_regret": ml_regret,
            "baseline_regret": baseline_regret,
        }
    )


selection_df = pd.DataFrame(selection_results)


# =====================================================================
# SELECTION METRICS
# =====================================================================

if len(selection_df) > 0:
    ml_top1 = selection_df["ml_top1"].mean()

    ml_top3 = selection_df["ml_top3"].mean()

    baseline_top1 = selection_df["baseline_top1"].mean()

    ml_regret = selection_df["ml_regret"].mean()

    baseline_regret = selection_df["baseline_regret"].mean()

    print(f"\nEpochs evaluated: {len(selection_df):,}")

    print("\nCurrent-best baseline:")

    print(f"  Top-1 accuracy: {baseline_top1 * 100:.2f}%")

    print(f"  Mean regret: {baseline_regret:.3f} dB-Hz")

    print("\nTemporal Ridge:")

    print(f"  Top-1 accuracy: {ml_top1 * 100:.2f}%")

    print(f"  Top-3 accuracy: {ml_top3 * 100:.2f}%")

    print(f"  Mean regret: {ml_regret:.3f} dB-Hz")


# =====================================================================
# SAVE MODELS
# =====================================================================
#
# NOTE: ridge_coefficients.csv and rf_importance.csv were already
# saved above in the "FEATURE IMPORTANCE" section — no need to write
# them again here (the original script both duplicated the save and,
# for rf_importance, tried to call .to_csv() on a raw numpy array
# instead of the DataFrame, which would have crashed).
# =====================================================================

print("\n--- Saving artifacts ---")


joblib.dump(ridge_final, BASELINE_ARTIFACTS / "ridge_temporal_1s.joblib")


joblib.dump(rf_final, BASELINE_ARTIFACTS / "rf_temporal_1s.joblib")


# ---------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------

with open(BASELINE_ARTIFACTS / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)


# ---------------------------------------------------------------------
# CV results
# ---------------------------------------------------------------------

cv_rows = []


for fold, (rmse, mae, r2) in enumerate(ridge_scores, start=1):
    cv_rows.append(
        {
            "model": "Ridge",
            "fold": fold,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }
    )


for fold, (rmse, mae, r2) in enumerate(rf_scores, start=1):
    cv_rows.append(
        {
            "model": "RandomForest",
            "fold": fold,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }
    )


pd.DataFrame(cv_rows).to_csv(BASELINE_ARTIFACTS / "cv_results.csv", index=False)


# ---------------------------------------------------------------------
# Signal selection
# ---------------------------------------------------------------------

if len(selection_df) > 0:
    selection_df.to_parquet(
        BASELINE_ARTIFACTS / "selection_results.parquet", index=False
    )


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 75)

print(" FINAL SUMMARY")

print("=" * 75)


print(f"\nTraining examples: {len(model_df):,}")


print(f"Features: {len(feature_cols)}")


print(f"Prediction horizon: {HORIZON_MS} ms")


print(f"CV type: held-out-{cv_group_name}")


print("\nRidge:")

print(f"  RMSE = {np.mean([x[0] for x in ridge_scores]):.3f}")

print(f"  MAE  = {np.mean([x[1] for x in ridge_scores]):.3f}")

print(f"  R²   = {np.mean([x[2] for x in ridge_scores]):.3f}")


print("\nRandom Forest:")

print(f"  RMSE = {np.mean([x[0] for x in rf_scores]):.3f}")

print(f"  MAE  = {np.mean([x[1] for x in rf_scores]):.3f}")

print(f"  R²   = {np.mean([x[2] for x in rf_scores]):.3f}")


if len(selection_df) > 0:
    print("\nSignal selection:")

    print(f"  Baseline Top-1 = {baseline_top1 * 100:.2f}%")

    print(f"  Ridge Top-1    = {ml_top1 * 100:.2f}%")

    print(f"  Ridge Top-3    = {ml_top3 * 100:.2f}%")

    print(f"  Baseline regret = {baseline_regret:.3f} dB-Hz")

    print(f"  Ridge regret    = {ml_regret:.3f} dB-Hz")


print("\nArtifacts:")

print(BASELINE_ARTIFACTS)

print("\nDone.")
