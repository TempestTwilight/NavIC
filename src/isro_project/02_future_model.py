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
    HORIZON_MS,
    MERGED_CORPUS,
    RANDOM_STATE,
    TARGET_TOLERANCE_MS,
    TEMPORAL_ARTIFACTS,
    ensure_dirs,
)

SIGNAL_KEYS = [
    "device_id",
    "session_id",
    "ConstellationType",
    "Svid",
    "CarrierFrequencyHz",
]

warnings.filterwarnings("ignore")


# ============================================================
# HELPERS
# ============================================================


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def circular_delta_deg(a, b):
    """Smallest signed angular difference a-b, in [-180, 180]."""
    return ((a - b + 180) % 360) - 180


def signal_tuple(row, keys=SIGNAL_KEYS):
    # FIX: was redefined on every inner-loop iteration; now a
    # single top-level function.
    return tuple(row[k] for k in keys)


def add_static_features(df):

    df = df.copy()

    numeric_cols = [
        "Cn0DbHz",
        "BasebandCn0DbHz",
        "AzimuthDegrees",
        "ElevationDegrees",
        "CarrierFrequencyHz",
        "yawDeg",
        "pitchDeg",
        "rollDeg",
        "SignalCount",
        "UsedInFix",
        "HasAlmanacData",
        "HasEphemerisData",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    az = np.deg2rad(df["AzimuthDegrees"])
    el = np.deg2rad(df["ElevationDegrees"])
    yaw = np.deg2rad(df["yawDeg"])
    pitch = np.deg2rad(df["pitchDeg"])
    roll = np.deg2rad(df["rollDeg"])

    df["elevation_sin"] = np.sin(el)
    df["elevation_cos"] = np.cos(el)
    df["is_above_horizon"] = (df["ElevationDegrees"] > 0).astype(float)

    rel_az = az - yaw
    df["rel_az_sin"] = np.sin(rel_az)
    df["rel_az_cos"] = np.cos(rel_az)

    el_clipped = np.clip(df["ElevationDegrees"].values, 1.0, 89.0)
    df["airmass"] = 1.0 / np.sin(np.deg2rad(el_clipped))
    df["airmass_ky"] = 1.0 / (
        np.sin(el)
        + 0.15
        * np.power(np.clip(df["ElevationDegrees"].values, 0.0, 90.0) + 3.885, -1.25)
    )

    df["pitch_sin"] = np.sin(pitch)
    df["pitch_cos"] = np.cos(pitch)
    df["roll_sin"] = np.sin(roll)
    df["roll_cos"] = np.cos(roll)

    df["freq_ghz"] = df["CarrierFrequencyHz"] / 1e9
    df["freq_loss_db"] = 20.0 * np.log10(
        np.clip(df["CarrierFrequencyHz"].values, 1e8, 1e11) / 1575420000.0
    )

    ctype = (
        pd.to_numeric(df["ConstellationType"], errors="coerce").fillna(-1).astype(int)
    )
    df["is_gps"] = (ctype == 1).astype(float)
    df["is_sbas"] = (ctype == 2).astype(float)
    df["is_glonass"] = (ctype == 3).astype(float)
    df["is_qzss"] = (ctype == 4).astype(float)
    df["is_beidou"] = (ctype == 5).astype(float)
    df["is_galileo"] = (ctype == 6).astype(float)
    df["is_navic"] = (ctype == 7).astype(float)

    df["rel_az_x_el"] = df["rel_az_cos"] * df["elevation_sin"]
    df["freq_loss_x_elev"] = df["freq_loss_db"] * df["elevation_sin"]

    if "tilt_angle" in df.columns:
        tilt = pd.to_numeric(df["tilt_angle"], errors="coerce")
        tilt_rad = np.deg2rad(tilt)
        df["tilt_angle_sin"] = np.sin(tilt_rad)
        df["tilt_angle_cos"] = np.cos(tilt_rad)

    return df


def add_lag_feature(frame, value_col, lag_ms, output_col):
    """
    Attach the value of `value_col` observed ~lag_ms earlier,
    for the same physical signal (matched on SIGNAL_KEYS).
    """

    key_cols = SIGNAL_KEYS + ["UnixTimeMillis"]
    if frame[key_cols].isna().any().any():
        raise ValueError(
            f"NaNs found in {key_cols} before building '{output_col}'. "
            "Drop or fill them first."
        )

    left = frame[SIGNAL_KEYS + ["UnixTimeMillis"]].copy()
    left["_orig_index"] = frame.index.values
    left["query_time"] = left["UnixTimeMillis"] - lag_ms
    left = left.sort_values(["query_time"] + SIGNAL_KEYS)

    right = frame[SIGNAL_KEYS + ["UnixTimeMillis", value_col]].copy()
    right = right.rename(
        columns={"UnixTimeMillis": "history_time", value_col: output_col}
    )
    right = right.sort_values(["history_time"] + SIGNAL_KEYS)

    merged = pd.merge_asof(
        left,
        right,
        left_on="query_time",
        right_on="history_time",
        by=SIGNAL_KEYS,
        direction="backward",
        tolerance=TARGET_TOLERANCE_MS,
    )

    result = pd.Series(index=frame.index, dtype=float)
    result.loc[merged["_orig_index"].values] = merged[output_col].values

    frame = frame.copy()
    frame[output_col] = result
    return frame


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print(" Temporal GNSS C/N0 Prediction (t+1s)")
print("=" * 75)

ensure_dirs()

print(f"\nLoading:\n{MERGED_CORPUS}")

if not MERGED_CORPUS.exists():
    raise FileNotFoundError(
        f"Data file not found at {MERGED_CORPUS}. "
        "Set ISRO_MERGED_CORPUS to override the default location."
    )

df = pd.read_parquet(MERGED_CORPUS)

print(f"\nLoaded rows: {len(df):,}")
print(f"Sessions: {df['session_id'].nunique()}")
print(f"Devices: {df['device_id'].nunique()}")
print("\nDevice IDs:")
print(df["device_id"].value_counts())


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
# TIME
# ============================================================

df["UnixTimeMillis"] = pd.to_numeric(df["UnixTimeMillis"], errors="coerce")
df = df.dropna(subset=["UnixTimeMillis"]).copy()

# FIX: SIGNAL_KEYS with NaNs break merge_asof's `by` matching
# silently (see add_lag_feature docstring). Drop them here so
# every downstream temporal-feature step is safe.
before = len(df)
df = df.dropna(subset=SIGNAL_KEYS).copy()
print(f"\nDropped {before - len(df):,} rows with missing signal-id fields")


# ============================================================
# STATIC FEATURES
# ============================================================

print("\n--- Building static features ---")
df = add_static_features(df)


# ============================================================
# SORT
# ============================================================

print("\n--- Sorting signal tracks ---")
df = df.sort_values(SIGNAL_KEYS + ["UnixTimeMillis"]).reset_index(drop=True)


# ============================================================
# TEMPORAL FEATURES
# ============================================================

print("\n--- Building temporal features ---")

lags = [
    (1000, "cn0_lag_1s"),
    (2000, "cn0_lag_2s"),
    (5000, "cn0_lag_5s"),
    (10000, "cn0_lag_10s"),
]

for lag_ms, output_col in lags:
    print(f"  Adding C/N0 lag {lag_ms / 1000:g}s...")
    df = add_lag_feature(df, "Cn0DbHz", lag_ms, output_col)

for lag_ms, output_col in [
    (1000, "elevation_lag_1s"),
    (5000, "elevation_lag_5s"),
    (1000, "azimuth_lag_1s"),
]:
    print(f"  Adding {output_col}...")
    source_col = "ElevationDegrees" if "elevation" in output_col else "AzimuthDegrees"
    df = add_lag_feature(df, source_col, lag_ms, output_col)


# ============================================================
# TEMPORAL DERIVATIVES
# ============================================================

print("\n--- Building temporal derivatives ---")

df["cn0_delta_1s"] = df["Cn0DbHz"] - df["cn0_lag_1s"]
df["cn0_delta_2s"] = df["Cn0DbHz"] - df["cn0_lag_2s"]
df["cn0_delta_5s"] = df["Cn0DbHz"] - df["cn0_lag_5s"]

df["cn0_slope_1s"] = df["cn0_delta_1s"] / 1.0
df["cn0_slope_5s"] = (df["Cn0DbHz"] - df["cn0_lag_5s"]) / 5.0

df["elevation_delta_1s"] = df["ElevationDegrees"] - df["elevation_lag_1s"]
df["elevation_delta_5s"] = df["ElevationDegrees"] - df["elevation_lag_5s"]
df["azimuth_delta_1s"] = circular_delta_deg(df["AzimuthDegrees"], df["azimuth_lag_1s"])


# ============================================================
# FUTURE TARGET
# ============================================================

print("\n--- Constructing t -> t+1 second target ---")

# FIX: same row-duplication risk as add_lag_feature, fixed the
# same way -- assign back by original index, not by a
# key-based merge.

target_left = df[SIGNAL_KEYS + ["UnixTimeMillis"]].copy()
target_left["_orig_index"] = df.index.values
target_left["target_time"] = target_left["UnixTimeMillis"] + HORIZON_MS
target_left = target_left.sort_values(["target_time"] + SIGNAL_KEYS)

target_right = df[SIGNAL_KEYS + ["UnixTimeMillis", "Cn0DbHz"]].copy()
target_right = target_right.rename(
    columns={"UnixTimeMillis": "future_time", "Cn0DbHz": "target_cn0_raw"}
)
target_right = target_right.sort_values(["future_time"] + SIGNAL_KEYS)

future = pd.merge_asof(
    target_left,
    target_right,
    left_on="target_time",
    right_on="future_time",
    by=SIGNAL_KEYS,
    direction="forward",  # strictly future relative to t
    tolerance=TARGET_TOLERANCE_MS,
)

target_raw = pd.Series(index=df.index, dtype=float)
target_raw.loc[future["_orig_index"].values] = future["target_cn0_raw"].values

# Residual target: delta Cn0 = future_cn0 - current_cn0
df["target_delta_cn0"] = target_raw - df["Cn0DbHz"]
df["target_cn0"] = df["target_delta_cn0"]


# ============================================================
# TARGET QUALITY
# ============================================================

print("\n--- Target quality ---")
valid_target = df["target_cn0"].notna()
print(f"Rows with t+1s target: {valid_target.sum():,}")
print(f"Rows without target: {(~valid_target).sum():,}")


# ============================================================
# FEATURES
# ============================================================

print("\n--- Building model features ---")

feature_cols = [
    "elevation_sin",
    "rel_az_sin",
    "rel_az_cos",
    "airmass_ky",
    "is_above_horizon",
    "freq_loss_db",
    "pitch_sin",
    "pitch_cos",
    "roll_sin",
    "roll_cos",
    "is_gps",
    "is_sbas",
    "is_glonass",
    "is_galileo",
    "is_beidou",
    "is_qzss",
    "is_navic",
    "rel_az_x_el",
    "freq_loss_x_elev",
    "Cn0DbHz",
    "BasebandCn0DbHz",
    "cn0_lag_1s",
    "cn0_lag_2s",
    "cn0_lag_5s",
    "cn0_lag_10s",
    "cn0_delta_1s",
    "cn0_delta_2s",
    "cn0_delta_5s",
    "cn0_slope_1s",
    "cn0_slope_5s",
    "elevation_lag_1s",
    "elevation_lag_5s",
    "azimuth_lag_1s",
    "elevation_delta_1s",
    "elevation_delta_5s",
    "azimuth_delta_1s",
]

if "tilt_angle_sin" in df.columns:
    feature_cols += ["tilt_angle_sin", "tilt_angle_cos"]


# ============================================================
# COMPLETE DATASET
# ============================================================

required_cols = feature_cols + ["target_cn0"]
complete = df.dropna(subset=required_cols).copy()

print(f"\nComplete temporal examples: {len(complete):,}")
print(f"Dropped: {len(df) - len(complete):,}")

X = complete[feature_cols].astype(float)
y = complete["target_cn0"].astype(float)


# ============================================================
# GROUPS
# ============================================================

# Calendar day grouping (prevents sky geometry leakage from same-day sessions)
complete["recording_day"] = pd.to_datetime(complete["UnixTimeMillis"], unit="ms").dt.date.astype(str)

if (
    complete["device_id"].nunique() > 1
    and not complete["device_id"].astype(str).eq("unknown").all()
):
    print("\nMultiple known devices detected. Grouping by device_id.")
    groups = complete["device_id"]
elif complete["recording_day"].nunique() >= 4:
    print("\nGrouping cross-validation by recording day.")
    groups = complete["recording_day"]
else:
    print("\nOnly one device detected.")
    print("Using held-out SESSION validation.")
    groups = complete["session_id"]

n_groups = groups.nunique()

# FIX: GroupKFold requires at least 2 groups; with 1 it fails
# with a confusing sklearn error. Fail with a clear message.
if n_groups < 2:
    raise ValueError(
        f"Only {n_groups} group(s) available for cross-validation "
        f"(from '{groups.name}'). Need at least 2 distinct "
        "devices/sessions to run GroupKFold."
    )

n_splits = min(4, n_groups)
cv = GroupKFold(n_splits=n_splits)


# ============================================================
# RIDGE CV
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
# RANDOM FOREST CV
# ============================================================

print("\n" + "=" * 75)
print(" Random Forest")
print("=" * 75)

rf_scores = []
max_rf_rows = 50000

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    if len(train_idx) > max_rf_rows:
        rng = np.random.default_rng(RANDOM_STATE + fold)
        train_idx_rf = rng.choice(train_idx, size=max_rf_rows, replace=False)
    else:
        train_idx_rf = train_idx

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
# FEATURE IMPORTANCE
# ============================================================

print("\n--- Feature importance ---")

ridge_coef = np.asarray(ridge_model.named_steps["ridge"].coef_).ravel()

print("Feature count:", len(feature_cols))
print("Ridge coefficient count:", len(ridge_coef))
assert len(feature_cols) == len(ridge_coef), "Feature/coefficient mismatch"

ridge_coefficients = pd.DataFrame(
    {
        "feature": feature_cols,
        "coefficient": ridge_coef,
    }
)
ridge_coefficients["abs_coefficient"] = ridge_coefficients["coefficient"].abs()
ridge_coefficients = ridge_coefficients.sort_values("abs_coefficient", ascending=False)
ridge_coefficients.to_csv(TEMPORAL_ARTIFACTS / "ridge_coefficients.csv", index=False)

print("\nTop Ridge features:")
print(ridge_coefficients.head(20).to_string(index=False))

rf_importance = np.asarray(rf_model.feature_importances_).ravel()
print("\nRF importance count:", len(rf_importance))
assert len(feature_cols) == len(rf_importance), "Feature/RF importance mismatch"

rf_importance_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance": rf_importance,
    }
)
rf_importance_df = rf_importance_df.sort_values("importance", ascending=False)
rf_importance_df.to_csv(TEMPORAL_ARTIFACTS / "rf_importance.csv", index=False)

print("\nTop Random Forest features:")
print(rf_importance_df.head(20).to_string(index=False))


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(ridge_model, TEMPORAL_ARTIFACTS / "ridge_temporal_1s.joblib")
joblib.dump(rf_model, TEMPORAL_ARTIFACTS / "rf_temporal_1s.joblib")

with open(TEMPORAL_ARTIFACTS / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

ridge_scores_df.to_csv(TEMPORAL_ARTIFACTS / "ridge_cv.csv", index=False)
rf_scores_df.to_csv(TEMPORAL_ARTIFACTS / "rf_cv.csv", index=False)


# ============================================================
# SIGNAL SELECTION EVALUATION
# ============================================================

print("\n" + "=" * 75)
print(" Signal Selection Evaluation")
print("=" * 75)

# We evaluate selection on CV test folds rather than on the
# final model's training predictions.

selection_records = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    print(f"\nSelection fold {fold}")

    ridge_cv = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    ridge_cv.fit(X.iloc[train_idx], y.iloc[train_idx])
    ridge_pred = ridge_cv.predict(X.iloc[test_idx])

    if len(train_idx) > max_rf_rows:
        rng = np.random.default_rng(RANDOM_STATE + 100 + fold)
        rf_train_idx = rng.choice(train_idx, size=max_rf_rows, replace=False)
    else:
        rf_train_idx = train_idx

    rf_cv = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_cv.fit(X.iloc[rf_train_idx], y.iloc[rf_train_idx])
    rf_pred = rf_cv.predict(X.iloc[test_idx])

    test = complete.iloc[test_idx].copy()
    test["ridge_pred"] = ridge_pred
    test["rf_pred"] = rf_pred

    # Persistence baseline: current C/N0 predicts future C/N0
    test["persistence_pred"] = test["Cn0DbHz"]

    group_cols = ["device_id", "session_id", "UnixTimeMillis"]

    for group_key, g in test.groupby(group_cols, sort=False):
        if len(g) < 2:
            continue

        actual_idx = g["target_cn0"].idxmax()
        actual_best = g.loc[actual_idx, "target_cn0"]
        actual_best_signal = g.loc[actual_idx, SIGNAL_KEYS + ["target_cn0"]].to_dict()
        actual_signal_tuple = tuple(actual_best_signal[k] for k in SIGNAL_KEYS)

        persistence_idx = g["persistence_pred"].idxmax()
        persistence_actual = g.loc[persistence_idx, "target_cn0"]
        persistence_regret = actual_best - persistence_actual

        ridge_idx = g["ridge_pred"].idxmax()
        ridge_actual = g.loc[ridge_idx, "target_cn0"]
        ridge_regret = actual_best - ridge_actual

        rf_idx = g["rf_pred"].idxmax()
        rf_actual = g.loc[rf_idx, "target_cn0"]
        rf_regret = actual_best - rf_actual

        top3_ridge = g.sort_values("ridge_pred", ascending=False).head(3)
        top3_rf = g.sort_values("rf_pred", ascending=False).head(3)
        top3_persistence = g.sort_values("persistence_pred", ascending=False).head(3)

        ridge_top3 = {signal_tuple(row) for _, row in top3_ridge.iterrows()}
        rf_top3 = {signal_tuple(row) for _, row in top3_rf.iterrows()}
        persistence_top3 = {signal_tuple(row) for _, row in top3_persistence.iterrows()}

        selection_records.append(
            {
                "fold": fold,
                "device_id": group_key[0],
                "session_id": group_key[1],
                "UnixTimeMillis": group_key[2],
                "num_candidates": len(g),
                "actual_best_cn0": actual_best,
                "persistence_selected_cn0": persistence_actual,
                "ridge_selected_cn0": ridge_actual,
                "rf_selected_cn0": rf_actual,
                "persistence_regret": persistence_regret,
                "ridge_regret": ridge_regret,
                "rf_regret": rf_regret,
                "persistence_top1": int(
                    signal_tuple(g.loc[persistence_idx]) == actual_signal_tuple
                ),
                "ridge_top1": int(
                    signal_tuple(g.loc[ridge_idx]) == actual_signal_tuple
                ),
                "rf_top1": int(signal_tuple(g.loc[rf_idx]) == actual_signal_tuple),
                "persistence_top3": int(actual_signal_tuple in persistence_top3),
                "ridge_top3": int(actual_signal_tuple in ridge_top3),
                "rf_top3": int(actual_signal_tuple in rf_top3),
            }
        )

selection_df = pd.DataFrame(selection_records)


# ============================================================
# SELECTION RESULTS
# ============================================================

if len(selection_df) > 0:
    print("\n--- Selection results ---")

    for name in ["persistence", "ridge", "rf"]:
        print(f"\n{name.upper()}")
        print(f"Top-1 accuracy: {100 * selection_df[name + '_top1'].mean():.2f}%")
        print(f"Top-3 accuracy: {100 * selection_df[name + '_top3'].mean():.2f}%")
        print(f"Mean C/N0 regret: {selection_df[name + '_regret'].mean():.3f} dB-Hz")
        print(
            f"Median C/N0 regret: {selection_df[name + '_regret'].median():.3f} dB-Hz"
        )

    selection_df.to_parquet(
        TEMPORAL_ARTIFACTS / "selection_results.parquet", index=False
    )

    selection_summary = pd.DataFrame(
        [
            {
                "model": name,
                "top1_accuracy": selection_df[name + "_top1"].mean(),
                "top3_accuracy": selection_df[name + "_top3"].mean(),
                "mean_regret_dbhz": selection_df[name + "_regret"].mean(),
                "median_regret_dbhz": selection_df[name + "_regret"].median(),
            }
            for name in ["persistence", "ridge", "rf"]
        ]
    )
    selection_summary.to_csv(TEMPORAL_ARTIFACTS / "selection_summary.csv", index=False)

    print("\nSelection summary:")
    print(selection_summary.to_string(index=False))

else:
    print("\nWARNING: No valid selection groups.")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print(" COMPLETE")
print("=" * 75)
print(f"\nArtifacts saved to:\n{TEMPORAL_ARTIFACTS}")
