#!/usr/bin/env python3
"""
Step 10: Inference Pipeline — predict_best_orientation()

Usage:
  .venv/bin/python scripts/step10_inference.py
  .venv/bin/python scripts/step10_inference.py --session 15_12-15-53
  .venv/bin/python scripts/step10_inference.py --csv data/test/15_12-15-53

Takes satellite Status data from a session or folder, predicts C/N0 for
each candidate orientation, and recommends the best one with dB improvement.

IMPORTANT — kept in sync with train_orientation_model.py:
    The keys built in engineer_for_orientation() must exactly match
    `feature_cols` saved by train_orientation_model.py (loaded below
    from feature_cols.json, and cross-checked at import time). Point
    ARTIFACTS at orientation_artifacts/ — NOT at a temporal/tracking
    model's artifacts directory, since those models were trained on a
    different feature set (they include current/lagged C/N0, which
    this script cannot supply for a hypothetical orientation) and a
    different target (future C/N0 of the same signal, not current
    C/N0 as a function of geometry).
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(PROJECT_ROOT)
ARTIFACTS = PROJECT_ROOT / "model" / "orientation_artifacts"
print(ARTIFACTS)
DATA_DIR = PROJECT_ROOT / "data" / "test"
print(DATA_DIR)

CANDIDATE_ANGLES = [0, 45, 90, 135]

# Load model + feature columns
ridge = joblib.load(ARTIFACTS / "ridge_orientation.joblib")
rf = joblib.load(ARTIFACTS / "rf_orientation.joblib")
with open(ARTIFACTS / "feature_cols.json") as f:
    FEATURE_COLS = json.load(f)


def find_sessions():
    """Any subfolder of DATA_DIR containing a Status.csv is a valid session."""
    sessions = sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and (d / "Status.csv").exists()
    )
    print(sessions)
    return sessions


def load_sky_from_folder(session_path: Path) -> pd.DataFrame:
    """Load Status.csv + OrientationDeg.csv from a session folder as a
    'current sky' snapshot.

    This is the single shared loader used by BOTH the --session and --csv
    code paths. It must be used for --csv too: yawDeg/pitchDeg/rollDeg/
    CalibrationAccuracy only exist in OrientationDeg.csv, not in
    Status.csv, so reading a bare Status.csv (as the old --csv branch
    did) can never produce a usable current_yaw for the recommendation.
    """
    status_path = session_path / "Status.csv"
    if not status_path.exists():
        raise FileNotFoundError(f"Status.csv not found in {session_path}")

    status = pd.read_csv(status_path, header=0)
    status.columns = [
        "_type", "UnixTimeMillis", "SignalCount", "SignalIndex",
        "ConstellationType", "Svid", "CarrierFrequencyHz", "Cn0DbHz",
        "AzimuthDegrees", "ElevationDegrees", "UsedInFix",
        "HasAlmanacData", "HasEphemerisData", "BasebandCn0DbHz",
    ]
    status = status.drop(columns=["_type"])

    for col in ["UnixTimeMillis", "ConstellationType", "Cn0DbHz", "AzimuthDegrees", "ElevationDegrees", "CarrierFrequencyHz"]:
        status[col] = pd.to_numeric(status[col], errors="coerce")

    status = status[status["Cn0DbHz"] > 0].copy()
    status = status.dropna(subset=["UnixTimeMillis"]).copy()
    status["UnixTimeMillis"] = status["UnixTimeMillis"].astype("int64")
    status = status.sort_values("UnixTimeMillis")

    # Fetch actual orientation from OrientationDeg.csv
    orient_path = session_path / "OrientationDeg.csv"
    if orient_path.exists():
        orient = pd.read_csv(orient_path, header=0)
        orient.columns = [
            "_type", "utcTimeMillis", "elapsedRealtimeNanos",
            "yawDeg", "rollDeg", "pitchDeg", "CalibrationAccuracy",
        ]
        orient = orient.drop(columns=["_type"])
        orient["utcTimeMillis"] = pd.to_numeric(orient["utcTimeMillis"], errors="coerce")
        orient = orient.dropna(subset=["utcTimeMillis"]).copy()
        orient["utcTimeMillis"] = orient["utcTimeMillis"].astype(int)
        orient = orient.sort_values("utcTimeMillis")

        status = status.sort_values("UnixTimeMillis")
        status = pd.merge_asof(
            status, orient,
            left_on="UnixTimeMillis", right_on="utcTimeMillis",
            tolerance=100, direction="nearest",
        )
    else:
        raise FileNotFoundError(f"OrientationDeg.csv not found in {session_path}")

    # Filter to high-quality orientation
    status = status[status["CalibrationAccuracy"] == 3].copy()

    if len(status) == 0:
        raise ValueError(
            "No rows remain after filtering (Cn0 > 0, valid orientation, "
            "CalibrationAccuracy == 3). Check the session's data quality "
            "before running inference."
        )

    return status


def load_session_sky(session_name: str) -> pd.DataFrame:
    """Resolve a session NAME (folder under DATA_DIR, or Categorized_Data
    fallback) to a folder path, then load it via load_sky_from_folder."""
    session_path = DATA_DIR / session_name
    if not session_path.exists():
        # Try with Categorized_Data prefix
        session_path = PROJECT_ROOT / "data" / "Categorized_Data" / session_name

    if not session_path.exists():
        raise FileNotFoundError(
            f"Session '{session_name}' not found under {DATA_DIR} "
            f"or {PROJECT_ROOT / 'data' / 'Categorized_Data'}."
        )

    return load_sky_from_folder(session_path)


def engineer_for_orientation(df: pd.DataFrame, target_yaw: float) -> np.ndarray:
    """Engineer features for a specific phone orientation (yaw angle).

    [Strategy 1 — Yaw-Agnostic]
    `yaw_sin` / `yaw_cos` are intentionally excluded. Only `rel_az_sin` /
    `rel_az_cos` (satellite azimuth relative to phone heading) are used
    for directional signal, making the feature set device-agnostic.

    Feature keys here must exactly match `feature_cols` saved by
    train_orientation_model.py. In particular:
      - No current/lagged/delta/slope C/N0 (not knowable for a
        hypothetical orientation).
      - No SignalCount / UsedInFix / HasAlmanacData / HasEphemerisData
        (describe fix state at the phone's ACTUAL current orientation,
        not a candidate one).
      - No pitch_sin/cos or roll_sin/cos — the model was trained with a
        single assumed-flat tilt (tilt_angle = 0) to match the
        assumption made here, so pitch/roll are never used by either
        side.
      - `rel_az_x_el` = rel_az_cos * elevation_sin (canonical
        definition — keep in sync with train_orientation_model.py).
    """
    elev_rad = np.radians(df["ElevationDegrees"].values)
    az_rad = np.radians(df["AzimuthDegrees"].values)
    rel_az = (df["AzimuthDegrees"].values - target_yaw) % 360
    rel_az_rad = np.radians(rel_az)

    feat = {}
    feat["elevation_sin"] = np.sin(elev_rad)
    feat["elevation_cos"] = np.cos(elev_rad)
    feat["rel_az_sin"] = np.sin(rel_az_rad)
    feat["rel_az_cos"] = np.cos(rel_az_rad)

    safe_elev = np.maximum(elev_rad, np.radians(1))
    feat["airmass"] = 1.0 / np.sin(safe_elev)

    feat["is_above_horizon"] = (df["ElevationDegrees"].values > 0).astype(float)

    for ctype, name in [(1, "gps"), (3, "glonass"), (4, "qzss"), (5, "beidou"), (6, "galileo"), (7, "navic")]:
        feat[f"is_{name}"] = (df["ConstellationType"].values == ctype).astype(float)

    # Canonical definition — must match train_orientation_model.add_features()
    feat["rel_az_x_el"] = feat["rel_az_cos"] * feat["elevation_sin"]
    feat["navic_x_airmass"] = feat["is_navic"] * feat["airmass"]
    feat["gps_x_airmass"] = feat["is_gps"] * feat["airmass"]
    feat["galileo_x_airmass"] = feat["is_galileo"] * feat["airmass"]

    feat["freq_ghz"] = df["CarrierFrequencyHz"].values / 1e9

    # Scalar features, broadcast to full array size.
    # Phone assumed held flat (tilt = 0) for every candidate orientation,
    # matching training's flat-tilt assumption: sin(0)=0, cos(0)=1.
    n = len(df)
    feat["tilt_angle_sin"] = np.full(n, 0.0)
    feat["tilt_angle_cos"] = np.full(n, 1.0)

    # Fail loudly rather than silently mispredicting on a feature-set
    # drift between this function and the trained model.
    expected = set(FEATURE_COLS)
    built = set(feat.keys())
    missing = expected - built
    extra = built - expected
    if missing or extra:
        raise ValueError(
            "Feature mismatch between engineer_for_orientation() and "
            f"the loaded model's feature_cols.json.\n"
            f"  Model expects but this function doesn't build: {missing}\n"
            f"  This function builds but model doesn't expect: {extra}\n"
            "Update engineer_for_orientation() (or retrain) so both sides "
            "agree exactly."
        )

    X = np.column_stack([feat[col] for col in FEATURE_COLS])
    return X


def predict_best_orientation(sky_df: pd.DataFrame, model=None, use="ridge"):
    """Run inference: predict signal quality for each candidate orientation."""
    if model is None:
        model = rf if use == "rf" else ridge

    results = []
    for angle in CANDIDATE_ANGLES:
        X = engineer_for_orientation(sky_df, float(angle))
        preds = model.predict(X)
        preds = np.clip(preds, 0, 55)  # physically plausible range
        weights = np.maximum(np.sin(np.radians(sky_df["ElevationDegrees"].values)), 0.1)
        score = np.average(preds, weights=weights)
        results.append((angle, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def format_recommendation(results, current_yaw, session_name="", model_name="Ridge"):
    """Format human-readable recommendation from prediction results."""
    best_angle = results[0][0]
    best_score = results[0][1]

    current_angle = min(CANDIDATE_ANGLES, key=lambda a: abs(a - current_yaw))
    current_score = next(s for a, s in results if a == current_angle)
    improvement = best_score - current_score

    lines = []
    lines.append("=" * 55)
    lines.append(f"  NavIC Orientation Recommendation{session_name}")
    lines.append("=" * 55)
    lines.append("")
    lines.append("  Predicted signal quality per orientation:")
    for angle, score in results:
        marker = " ★ BEST" if angle == best_angle else ""
        lines.append(f"    {angle:>3}°: {score:5.2f} dB-Hz{marker}")
    lines.append("")
    lines.append(f"  Current orientation (yaw ≈ {current_yaw:.0f}°):  {current_angle:>3}° — {current_score:.2f} dB-Hz")
    lines.append(f"  Recommended orientation:              {best_angle:>3}° — {best_score:.2f} dB-Hz")
    lines.append(f"  Estimated improvement:                {improvement:+.2f} dB-Hz")
    if improvement > 0:
        direction = "clockwise" if (best_angle - current_angle) % 360 < 180 else "counter-clockwise"
        turn = (best_angle - current_angle) % 360
        if turn > 180:
            turn = 360 - turn
        lines.append(f"  → Rotate ~{turn:.0f}° {direction} (yaw)")
    lines.append("")
    lines.append(f"  Model: {model_name}")
    lines.append("=" * 55)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Predict best phone orientation for GNSS signal")
    parser.add_argument("--session", default=None, help="Session folder name under data/test (or data/Categorized_Data)")
    parser.add_argument("--model", choices=["ridge", "rf"], default="ridge", help="Which model to use")
    parser.add_argument("--csv", default=None, help="Path to a session FOLDER containing Status.csv + OrientationDeg.csv")
    args = parser.parse_args()

    if args.csv:
        # NOTE: --csv expects a session FOLDER, not a bare Status.csv file.
        # yawDeg/pitchDeg/rollDeg/CalibrationAccuracy live in
        # OrientationDeg.csv, which must sit alongside Status.csv in the
        # same folder — a lone Status.csv has no orientation data at all,
        # so reading it directly (the old behavior) left `sky["yawDeg"]`
        # undefined further down.
        csv_path = Path(args.csv)
        print(f"Loading folder: {csv_path}")
        sky = load_sky_from_folder(csv_path)
    elif args.session:
        print(f"Loading session: {args.session}")
        sky = load_session_sky(args.session)
    else:
        sessions = find_sessions()
        if not sessions:
            raise FileNotFoundError(f"No sessions with Status.csv found in {DATA_DIR}")
        default = sessions[-1]
        print(f"No session specified. Using last session: {default}")
        sky = load_session_sky(default)  # NOTE: previously `sky` was never assigned here

    print(f"Loaded {len(sky):,} satellite readings from {sky['UnixTimeMillis'].nunique()} epochs")
    print(f"Constellations: {sorted(sky['ConstellationType'].unique())}")

    model_obj = rf if args.model == "rf" else ridge
    model_name = "Random Forest" if args.model == "rf" else "Ridge"

    results = predict_best_orientation(sky, model=model_obj)
    current_yaw = sky["yawDeg"].iloc[0]

    print(f"\nModel: {model_name}")
    rec = format_recommendation(results, current_yaw, f" [{args.session or args.csv or 'auto'}]", model_name)
    print(rec)

    best_angle = results[0][0]
    X_best = engineer_for_orientation(sky, float(best_angle))
    preds = model_obj.predict(X_best)
    preds = np.clip(preds, 0, 55)
    actual = sky["Cn0DbHz"].values
    print(f"\n  Per-satellite stats at {best_angle}°: "
          f"predicted mean={preds.mean():.1f} dB-Hz, "
          f"actual mean={actual.mean():.1f} dB-Hz")


if __name__ == "__main__":
    main()