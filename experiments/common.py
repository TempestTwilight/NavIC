"""Shared data loading and feature building for GNSS experiments.
Reuses tested pipeline logic from src/isro_project/02_future_model.py.
"""

import importlib
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("ISRO_PROJECT_ROOT", "/home/tempest/Documents/ISRO_Project"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isro_project.config import HORIZON_MS, MERGED_CORPUS, TARGET_TOLERANCE_MS  # noqa: E402

future_mod = importlib.import_module("isro_project.02_future_model")
SIGNAL_KEYS = future_mod.SIGNAL_KEYS
add_static_features = future_mod.add_static_features
add_lag_feature = future_mod.add_lag_feature
circular_delta_deg = future_mod.circular_delta_deg

FEATURE_COLS = [
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


def load_and_preprocess_data(
    corpus_path: Path = MERGED_CORPUS, sample_n: int | None = None
) -> pd.DataFrame:
    """Load combined corpus, attach lags, derivatives, and t+15s target."""
    df = pd.read_parquet(corpus_path)
    df = df[pd.to_numeric(df["Cn0DbHz"], errors="coerce") > 0].copy()
    if "CalibrationAccuracy" in df.columns:
        df = df[pd.to_numeric(df["CalibrationAccuracy"], errors="coerce") == 3].copy()

    df["UnixTimeMillis"] = pd.to_numeric(df["UnixTimeMillis"], errors="coerce")
    df = df.dropna(subset=["UnixTimeMillis"] + SIGNAL_KEYS).copy()

    df = add_static_features(df)
    df = df.sort_values(SIGNAL_KEYS + ["UnixTimeMillis"]).reset_index(drop=True)

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
        src = "ElevationDegrees" if "elevation" in col else "AzimuthDegrees"
        df = add_lag_feature(df, src, lag_ms, col)

    df["cn0_delta_1s"] = df["Cn0DbHz"] - df["cn0_lag_1s"]
    df["cn0_delta_2s"] = df["Cn0DbHz"] - df["cn0_lag_2s"]
    df["cn0_delta_5s"] = df["Cn0DbHz"] - df["cn0_lag_5s"]
    df["cn0_slope_1s"] = df["cn0_delta_1s"] / 1.0
    df["cn0_slope_5s"] = (df["Cn0DbHz"] - df["cn0_lag_5s"]) / 5.0
    df["elevation_delta_1s"] = df["ElevationDegrees"] - df["elevation_lag_1s"]
    df["elevation_delta_5s"] = df["ElevationDegrees"] - df["elevation_lag_5s"]
    df["azimuth_delta_1s"] = circular_delta_deg(
        df["AzimuthDegrees"], df["azimuth_lag_1s"]
    )

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

    data = df.dropna(
        subset=FEATURE_COLS + ["target_delta_cn0", "target_cn0_raw"]
    ).copy()
    if sample_n and len(data) > sample_n:
        data = data.sample(n=sample_n, random_state=42).copy()
    return data
