"""
Run all GNSS experiments (A through E) in a single script.
Reuses the data pipeline from src/isro_project/02_future_model.py.
Reports real results for:
  - Experiment A: Baseline Ridge + HistGB (absolute future C/N0 regret vs persistence)
  - Experiment B: Hurdle model (fading classifier + specialized regressor)
  - Experiment C: Learning-to-Rank via epoch-relative target
  - Experiment D: Leave-one-device-out zero-shot regret
  - Experiment E: Survival-weighted expected-value selection
"""

import importlib
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/home/tempest/Documents/ISRO_Project")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isro_project.config import (  # noqa: E402
    HORIZON_MS,
    MERGED_CORPUS,
    RANDOM_STATE,
    TARGET_TOLERANCE_MS,
)

future_mod = importlib.import_module("isro_project.02_future_model")
SIGNAL_KEYS = future_mod.SIGNAL_KEYS
add_static_features = future_mod.add_static_features
add_lag_feature = future_mod.add_lag_feature
circular_delta_deg = future_mod.circular_delta_deg
signal_tuple = future_mod.signal_tuple

print("=" * 75)
print(" GNSS Signal Optimization Experiments A-E")
print("=" * 75)

t0_total = time.time()

# -------------------------------------------------------------
# 1. LOAD & PREPARE DATA
# -------------------------------------------------------------
print("\n[1/6] Loading combined corpus...")
t0 = time.time()
df = pd.read_parquet(MERGED_CORPUS)
print(f"Loaded {len(df):,} rows ({time.time() - t0:.2f}s)")

# Filter
df = df[pd.to_numeric(df["Cn0DbHz"], errors="coerce") > 0].copy()
if "CalibrationAccuracy" in df.columns:
    df = df[pd.to_numeric(df["CalibrationAccuracy"], errors="coerce") == 3].copy()

df["UnixTimeMillis"] = pd.to_numeric(df["UnixTimeMillis"], errors="coerce")
df = df.dropna(subset=["UnixTimeMillis"] + SIGNAL_KEYS).copy()

# Add static features
df = add_static_features(df)
df = df.sort_values(SIGNAL_KEYS + ["UnixTimeMillis"]).reset_index(drop=True)

# Build lags
print("\n[2/6] Building lag & temporal features...")
t0 = time.time()
for lag_ms, col in [
    (1000, "cn0_lag_1s"),
    (2000, "cn0_lag_2s"),
    (5000, "cn0_lag_5s"),
    (10000, "cn0_lag_10s"),
]:
    df = add_lag_feature(df, "Cn0DbHz", lag_ms, col)

for lag_ms, col in [
    (1000, "elevation_lag_1s"),
    (5000, "elevation_lag_5s"),
    (1000, "azimuth_lag_1s"),
]:
    source = "ElevationDegrees" if "elevation" in col else "AzimuthDegrees"
    df = add_lag_feature(df, source, lag_ms, col)

df["cn0_delta_1s"] = df["Cn0DbHz"] - df["cn0_lag_1s"]
df["cn0_delta_2s"] = df["Cn0DbHz"] - df["cn0_lag_2s"]
df["cn0_delta_5s"] = df["Cn0DbHz"] - df["cn0_lag_5s"]
df["cn0_slope_1s"] = df["cn0_delta_1s"] / 1.0
df["cn0_slope_5s"] = (df["Cn0DbHz"] - df["cn0_lag_5s"]) / 5.0
df["elevation_delta_1s"] = df["ElevationDegrees"] - df["elevation_lag_1s"]
df["elevation_delta_5s"] = df["ElevationDegrees"] - df["elevation_lag_5s"]
df["azimuth_delta_1s"] = circular_delta_deg(df["AzimuthDegrees"], df["azimuth_lag_1s"])

# Future target: t+15s
print("\n[3/6] Building t+15s target...")
t0 = time.time()
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
    direction="forward",
    tolerance=TARGET_TOLERANCE_MS,
)

matched_idx = future["_orig_index"].to_numpy()
target_raw = np.full(len(df), np.nan, dtype=float)
target_raw[matched_idx] = future["target_cn0_raw"].to_numpy()

df["target_cn0_raw"] = target_raw
df["target_delta_cn0"] = target_raw - df["Cn0DbHz"]
df["is_fading"] = (df["target_delta_cn0"].abs() > 1.5).astype(float)

# Filter complete cases
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

data = df.dropna(subset=feature_cols + ["target_delta_cn0", "target_cn0_raw"]).copy()
print(f"Complete cases: {len(data):,} ({time.time() - t0:.2f}s)")

# Downsample for tractable multi-model run if large
SAMPLE_LIMIT = 200000
if len(data) > SAMPLE_LIMIT:
    print(f"Downsampling from {len(data):,} to {SAMPLE_LIMIT:,} for cross-validation")
    rng = np.random.default_rng(RANDOM_STATE)
    data = data.sample(n=SAMPLE_LIMIT, random_state=RANDOM_STATE).copy()

X = data[feature_cols].astype(np.float32)
y_delta = data["target_delta_cn0"].astype(np.float32)
y_raw = data["target_cn0_raw"].astype(np.float32)
y_fade = data["is_fading"].astype(np.float32)
devices = data["device_id"]

print("\n[4/6] Running Experiments...")

# -------------------------------------------------------------
# EXPERIMENT A: Baseline Ridge + Proper Selection Regret
# -------------------------------------------------------------
print("\n--- EXPERIMENT A: Baseline Ridge with Absolute-Regret Evaluation ---")
exp_a_results = []
models = {
    "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]),
    "HistGB": HistGradientBoostingRegressor(
        max_iter=60, max_depth=8, min_samples_leaf=30, random_state=RANDOM_STATE
    ),
}

# Cross-device LODO splits
unique_devices = devices.unique()
for dev in unique_devices:
    train_mask = devices != dev
    test_mask = devices == dev

    X_train, y_train = X[train_mask], y_delta[train_mask]
    X_test, y_test = X[test_mask], y_delta[test_mask]
    test_data = data[test_mask].copy()

    # Train Ridge
    models["Ridge"].fit(X_train, y_train)
    pred_delta_ridge = models["Ridge"].predict(X_test)

    # Absolute future C/N0 prediction = current C/N0 + delta_pred
    test_data["pred_raw_ridge"] = test_data["Cn0DbHz"] + pred_delta_ridge
    test_data["pred_raw_persistence"] = test_data["Cn0DbHz"]

    # Evaluate selection per (session, epoch)
    group_cols = ["session_id", "UnixTimeMillis"]
    regrets_ridge = []
    regrets_persistence = []

    for _, g in test_data.groupby(group_cols):
        if len(g) < 2:
            continue
        best_actual = g["target_cn0_raw"].max()

        # Persistence choice
        p_choice = g.loc[g["pred_raw_persistence"].idxmax(), "target_cn0_raw"]
        regrets_persistence.append(best_actual - p_choice)

        # Ridge choice
        r_choice = g.loc[g["pred_raw_ridge"].idxmax(), "target_cn0_raw"]
        regrets_ridge.append(best_actual - r_choice)

    exp_a_results.append(
        {
            "held_out_device": dev,
            "n_epochs": len(regrets_ridge),
            "persistence_mean_regret": np.mean(regrets_persistence),
            "ridge_mean_regret": np.mean(regrets_ridge),
            "regret_diff": np.mean(regrets_ridge) - np.mean(regrets_persistence),
        }
    )

exp_a_df = pd.DataFrame(exp_a_results)
print(exp_a_df.to_string(index=False))
exp_a_df.to_csv(
    PROJECT_ROOT / "experiments/baseline/results/baseline_regret.csv", index=False
)

# -------------------------------------------------------------
# EXPERIMENT B: Two-Stage Hurdle Model
# -------------------------------------------------------------
print("\n--- EXPERIMENT B: Two-Stage Hurdle Model ---")
# Stage 1: Classifier for fading (|delta| > 1.5 dB)
# Stage 2: Regressor trained only on fading instances
exp_b_results = []

for dev in unique_devices:
    train_mask = devices != dev
    test_mask = devices == dev

    X_train, y_train_fade, y_train_delta = (
        X[train_mask],
        y_fade[train_mask],
        y_delta[train_mask],
    )
    X_test, y_test_fade, y_test_delta = (
        X[test_mask],
        y_fade[test_mask],
        y_delta[test_mask],
    )

    # Stage 1
    clf = HistGradientBoostingClassifier(max_iter=50, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train_fade)
    pred_fade_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_fade, pred_fade_prob)

    # Stage 2 (trained on fading instances only)
    fade_train_mask = y_train_fade == 1
    reg_fading = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    reg_fading.fit(X_train[fade_train_mask], y_train_delta[fade_train_mask])

    # Combined prediction: if P(fade) > 0.5 use fading regressor, else 0 (pure persistence)
    hurdle_delta = np.where(pred_fade_prob > 0.5, reg_fading.predict(X_test), 0.0)

    # Metrics on fading subset
    fading_test_mask = y_test_fade == 1
    rmse_hurdle_fading = np.sqrt(
        mean_squared_error(
            y_test_delta[fading_test_mask], hurdle_delta[fading_test_mask]
        )
    )
    rmse_persist_fading = np.sqrt(
        mean_squared_error(
            y_test_delta[fading_test_mask],
            np.zeros_like(y_test_delta[fading_test_mask]),
        )
    )

    exp_b_results.append(
        {
            "held_out_device": dev,
            "fading_auc": auc,
            "persistence_fading_rmse": rmse_persist_fading,
            "hurdle_fading_rmse": rmse_hurdle_fading,
            "hurdle_beats_persistence": rmse_hurdle_fading < rmse_persist_fading,
        }
    )

exp_b_df = pd.DataFrame(exp_b_results)
print(exp_b_df.to_string(index=False))
exp_b_df.to_csv(
    PROJECT_ROOT / "experiments/hurdle/results/hurdle_results.csv", index=False
)

# -------------------------------------------------------------
# EXPERIMENT C: Learning-to-Rank (Epoch-Relative Target)
# -------------------------------------------------------------
print("\n--- EXPERIMENT C: Learning-to-Rank Formulation ---")
# Compute epoch-centered future C/N0: y_rel = target_cn0_raw - mean(target_cn0_raw | epoch)
data["epoch_mean_future"] = data.groupby(["session_id", "UnixTimeMillis"])[
    "target_cn0_raw"
].transform("mean")
data["target_relative"] = data["target_cn0_raw"] - data["epoch_mean_future"]

y_rel = data["target_relative"].astype(np.float32)

exp_c_results = []
for dev in unique_devices:
    train_mask = devices != dev
    test_mask = devices == dev

    X_train, y_train_r = X[train_mask], y_rel[train_mask]
    X_test = X[test_mask]
    test_data = data[test_mask].copy()

    # Train HistGB on relative target
    ltr_model = HistGradientBoostingRegressor(
        max_iter=60, max_depth=8, min_samples_leaf=30, random_state=RANDOM_STATE
    )
    ltr_model.fit(X_train, y_train_r)
    test_data["pred_rel"] = ltr_model.predict(X_test)
    test_data["pred_persist"] = test_data["Cn0DbHz"]

    regrets_ltr = []
    regrets_p = []
    top1_match_ltr = []
    top1_match_p = []

    for _, g in test_data.groupby(["session_id", "UnixTimeMillis"]):
        if len(g) < 2:
            continue
        best_idx = g["target_cn0_raw"].idxmax()
        best_val = g.loc[best_idx, "target_cn0_raw"]

        ltr_choice_idx = g["pred_rel"].idxmax()
        p_choice_idx = g["pred_persist"].idxmax()

        regrets_ltr.append(best_val - g.loc[ltr_choice_idx, "target_cn0_raw"])
        regrets_p.append(best_val - g.loc[p_choice_idx, "target_cn0_raw"])
        top1_match_ltr.append(int(ltr_choice_idx == best_idx))
        top1_match_p.append(int(p_choice_idx == best_idx))

    exp_c_results.append(
        {
            "held_out_device": dev,
            "ltr_mean_regret": np.mean(regrets_ltr),
            "persistence_mean_regret": np.mean(regrets_p),
            "ltr_top1_acc": np.mean(top1_match_ltr),
            "persistence_top1_acc": np.mean(top1_match_p),
        }
    )

exp_c_df = pd.DataFrame(exp_c_results)
print(exp_c_df.to_string(index=False))
exp_c_df.to_csv(PROJECT_ROOT / "experiments/ltr/results/ltr_results.csv", index=False)

# -------------------------------------------------------------
# EXPERIMENT D: Cross-Device Generalization & Zero-Shot Regret
# -------------------------------------------------------------
print("\n--- EXPERIMENT D: Leave-One-Device-Out Zero-Shot Regret Bounds ---")
exp_d_results = []
for dev in unique_devices:
    train_mask = devices != dev
    test_mask = devices == dev

    # 1. Raw target (no normalization)
    m_raw = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    m_raw.fit(X[train_mask], y_raw[train_mask])
    pred_raw = m_raw.predict(X[test_mask])

    # 2. Residual delta target (our recommended strategy)
    m_delta = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    m_delta.fit(X[train_mask], y_delta[train_mask])
    pred_delta = data.loc[test_mask, "Cn0DbHz"] + m_delta.predict(X[test_mask])

    test_data = data[test_mask].copy()
    test_data["pred_raw_strategy"] = pred_raw
    test_data["pred_delta_strategy"] = pred_delta

    reg_raw = []
    reg_delta = []
    for _, g in test_data.groupby(["session_id", "UnixTimeMillis"]):
        if len(g) < 2:
            continue
        best_val = g["target_cn0_raw"].max()
        reg_raw.append(
            best_val - g.loc[g["pred_raw_strategy"].idxmax(), "target_cn0_raw"]
        )
        reg_delta.append(
            best_val - g.loc[g["pred_delta_strategy"].idxmax(), "target_cn0_raw"]
        )

    exp_d_results.append(
        {
            "held_out_device": dev,
            "raw_strategy_mean_regret": np.mean(reg_raw),
            "delta_strategy_mean_regret": np.mean(reg_delta),
            "delta_improvement": np.mean(reg_raw) - np.mean(reg_delta),
        }
    )

exp_d_df = pd.DataFrame(exp_d_results)
print(exp_d_df.to_string(index=False))
exp_d_df.to_csv(
    PROJECT_ROOT / "experiments/cross_device/results/cross_device_results.csv",
    index=False,
)

# -------------------------------------------------------------
# EXPERIMENT E: Survival-Weighted Selection
# -------------------------------------------------------------
print("\n--- EXPERIMENT E: Survival-Weighted Utility Selection ---")
# Model dropout / survival: does signal survive at t+15s?
# In our filtered corpus with complete targets, all rows survived.
# But low elevation satellites have higher probability of dropping out.
# Model P(survive) as logistic function of elevation and current C/N0:
# P(survive) = 1 / (1 + exp(-(0.05 * el + 0.1 * (cn0 - 15))))
SENSITIVITY_FLOOR = 12.0  # dB-Hz
data["p_survive_synthetic"] = 1.0 / (
    1.0 + np.exp(-(0.05 * data["ElevationDegrees"] + 0.1 * (data["Cn0DbHz"] - 15.0)))
)

# Utility = P(survive) * predicted_C/N0 + (1 - P(survive)) * floor
data["expected_utility_persistence"] = (
    data["p_survive_synthetic"] * data["Cn0DbHz"]
    + (1.0 - data["p_survive_synthetic"]) * SENSITIVITY_FLOOR
)

exp_e_results = []
for dev in unique_devices:
    test_data = data[devices == dev]

    reg_pure_persist = []
    reg_survival_weighted = []

    for _, g in test_data.groupby(["session_id", "UnixTimeMillis"]):
        if len(g) < 2:
            continue
        best_val = g["target_cn0_raw"].max()

        # Pure persistence choice
        pure_choice = g.loc[g["Cn0DbHz"].idxmax(), "target_cn0_raw"]
        reg_pure_persist.append(best_val - pure_choice)

        # Survival-weighted choice
        surv_choice = g.loc[
            g["expected_utility_persistence"].idxmax(), "target_cn0_raw"
        ]
        reg_survival_weighted.append(best_val - surv_choice)

    exp_e_results.append(
        {
            "held_out_device": dev,
            "pure_persistence_mean_regret": np.mean(reg_pure_persist),
            "survival_weighted_mean_regret": np.mean(reg_survival_weighted),
            "difference": np.mean(reg_survival_weighted) - np.mean(reg_pure_persist),
        }
    )

exp_e_df = pd.DataFrame(exp_e_results)
print(exp_e_df.to_string(index=False))
exp_e_df.to_csv(
    PROJECT_ROOT / "experiments/survival/results/survival_results.csv", index=False
)

print("\n" + "=" * 75)
print(f" ALL EXPERIMENTS FINISHED in {time.time() - t0_total:.2f}s")
print("=" * 75)
