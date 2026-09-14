"""
Merge Status.csv with OrientationDeg.csv per session.

Produces:
    planning/combined_corpus.parquet

The merged dataset contains:
    - Per-signal GNSS measurements
    - Nearest phone orientation
    - Orientation/GNSS synchronization error
    - Session metadata
    - Device ID
    - Signal identity

This dataset is used by the temporal C/N0 prediction pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path().resolve()

DATA_DIR = PROJECT_ROOT / "data" / "Categorized_Data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "merged_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# CONFIGURATION
# =====================================================================

ORIENTATION_TOLERANCE_MS = 100


# =====================================================================
# SESSION DISCOVERY
# =====================================================================

FOLDERS = sorted(
    [d.name for d in DATA_DIR.iterdir() if d.is_dir() and "degree" in d.name.lower()]
)


print("=" * 70)
print(" GNSS + Orientation Merge")
print("=" * 70)

print(f"\nFound {len(FOLDERS)} session folders.")


# =====================================================================
# SESSION NAME PARSER
# =====================================================================


def parse_session(name):
    """
    Extract experimental metadata from session folder name.

    Expected examples:

        45_degree_front_horizontal
        90_degree_back_vertical
        0_degree_up_facing

    Returns:
        tilt_angle, facing, phone_roll
    """

    try:
        parts = name.replace("_degree_", "_").replace("_facing", "").split("_")

        angle = int(parts[0])

        facing = parts[1] if len(parts) > 1 else "unknown"

        phone_roll = parts[2] if len(parts) > 2 else "unknown"

        return angle, facing, phone_roll

    except Exception:
        return (np.nan, "unknown", "unknown")


# =====================================================================
# DEVICE IDENTIFICATION
# =====================================================================


def get_device_id(folder_name):
    """
    Infer device ID from session folder name.

    IMPORTANT:
    Update this function if your actual folder names use a
    different naming convention.
    """

    name = folder_name.lower()

    if "oneplus" in name:
        return "oneplus_cph2401"

    if "pixel4xlmodded" in name:
        return "Pixel4XLModded"

    if "pixel4xl" in name:
        return "Pixel4XL"

    if "pixel4" in name:
        return "Pixel4"

    if "mi8" in name:
        return "Mi8"

    return "unknown"


# =====================================================================
# MERGE ALL SESSIONS
# =====================================================================

all_merged = []


for folder_name in FOLDERS:
    print("\n" + "-" * 70)
    print(f"Processing: {folder_name}")
    print("-" * 70)

    session_path = DATA_DIR / folder_name

    status_path = session_path / "Status.csv"

    orient_path = session_path / "OrientationDeg.csv"

    # =================================================================
    # CHECK FILES
    # =================================================================

    if not status_path.exists():
        print(f"  WARNING: Missing {status_path}")

        continue

    if not orient_path.exists():
        print(f"  WARNING: Missing {orient_path}")

        continue

    # =================================================================
    # READ STATUS
    # =================================================================

    status = pd.read_csv(status_path)

    # -----------------------------------------------------------------
    # Validate column count
    # -----------------------------------------------------------------

    if len(status.columns) != 14:
        raise ValueError(
            f"{status_path} has {len(status.columns)} columns; expected 14."
        )

    status.columns = [
        "_type",
        "UnixTimeMillis",
        "SignalCount",
        "SignalIndex",
        "ConstellationType",
        "Svid",
        "CarrierFrequencyHz",
        "Cn0DbHz",
        "AzimuthDegrees",
        "ElevationDegrees",
        "UsedInFix",
        "HasAlmanacData",
        "HasEphemerisData",
        "BasebandCn0DbHz",
    ]

    status = status.drop(columns=["_type"])

    # -----------------------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------------------

    numeric_status = [
        "UnixTimeMillis",
        "SignalCount",
        "SignalIndex",
        "ConstellationType",
        "Svid",
        "CarrierFrequencyHz",
        "Cn0DbHz",
        "AzimuthDegrees",
        "ElevationDegrees",
        "UsedInFix",
        "HasAlmanacData",
        "HasEphemerisData",
        "BasebandCn0DbHz",
    ]

    for col in numeric_status:
        status[col] = pd.to_numeric(status[col], errors="coerce")

    # =================================================================
    # READ ORIENTATION
    # =================================================================

    orient = pd.read_csv(orient_path)

    if len(orient.columns) != 7:
        raise ValueError(
            f"{orient_path} has {len(orient.columns)} columns; expected 7."
        )

    orient.columns = [
        "_type",
        "utcTimeMillis",
        "elapsedRealtimeNanos",
        "yawDeg",
        "rollDeg",
        "pitchDeg",
        "CalibrationAccuracy",
    ]

    orient = orient.drop(columns=["_type"])

    numeric_orientation = [
        "utcTimeMillis",
        "elapsedRealtimeNanos",
        "yawDeg",
        "rollDeg",
        "pitchDeg",
        "CalibrationAccuracy",
    ]

    for col in numeric_orientation:
        orient[col] = pd.to_numeric(orient[col], errors="coerce")

    # =================================================================
    # REMOVE INVALID TIMESTAMPS
    # =================================================================

    status = status.dropna(subset=["UnixTimeMillis"]).copy()

    orient = orient.dropna(subset=["utcTimeMillis"]).copy()

    status["UnixTimeMillis"] = status["UnixTimeMillis"].astype(np.int64)

    orient["utcTimeMillis"] = orient["utcTimeMillis"].astype(np.int64)

    # =================================================================
    # SORT ORIENTATION
    # =================================================================

    orient = orient.sort_values("utcTimeMillis").reset_index(drop=True)

    # -----------------------------------------------------------------
    # Remove exact duplicate orientation timestamps.
    #
    # We keep the LAST measurement at an identical timestamp.
    # -----------------------------------------------------------------

    orient = orient.drop_duplicates(subset=["utcTimeMillis"], keep="last")

    # =================================================================
    # SORT STATUS / ORIENTATION
    # =================================================================

    status = status.sort_values("UnixTimeMillis").reset_index(drop=True)

    orient = orient.sort_values("utcTimeMillis").reset_index(drop=True)

    # =================================================================
    # MERGE
    # =================================================================

    merged = pd.merge_asof(
        status,
        orient,
        left_on="UnixTimeMillis",
        right_on="utcTimeMillis",
        tolerance=ORIENTATION_TOLERANCE_MS,
        direction="nearest",
    )

    # =================================================================
    # ORIENTATION SYNCHRONIZATION ERROR
    # =================================================================

    merged["orientation_dt_ms"] = merged["UnixTimeMillis"] - merged["utcTimeMillis"]

    # =================================================================
    # SESSION METADATA
    # =================================================================

    merged["session_id"] = folder_name

    angle, facing, phone_roll = parse_session(folder_name)

    merged["tilt_angle"] = angle

    merged["facing"] = facing

    merged["phone_roll"] = phone_roll

    # =================================================================
    # DEVICE ID
    # =================================================================

    merged["device_id"] = get_device_id(folder_name)

    # =================================================================
    # SIGNAL IDENTITY
    # =================================================================
    #
    # A signal is identified by:
    #
    #   device
    #   session
    #   constellation
    #   SVID
    #   carrier frequency
    #
    # This is NOT used as an ML feature.
    #
    # It is only used to track the same satellite signal through time.
    # =================================================================

    merged["signal_id"] = (
        merged["device_id"].astype(str)
        + "_"
        + merged["session_id"].astype(str)
        + "_"
        + merged["ConstellationType"].fillna(-1).astype(int).astype(str)
        + "_"
        + merged["Svid"].fillna(-1).astype(int).astype(str)
        + "_"
        + merged["CarrierFrequencyHz"].fillna(-1).astype(np.int64).astype(str)
    )

    # =================================================================
    # SESSION STATISTICS
    # =================================================================

    matched = merged["utcTimeMillis"].notna().sum()

    total = len(merged)

    match_percent = 100.0 * matched / total if total > 0 else 0.0

    print(f"  Status rows:       {len(status):,}")

    print(f"  Merged rows:       {len(merged):,}")

    print(f"  Orientation match: {match_percent:.2f}%")

    if matched > 0:
        abs_dt = merged["orientation_dt_ms"].abs()

        print(f"  |dt| median:       {abs_dt.median():.1f} ms")

        print(f"  |dt| 95th pct:     {abs_dt.quantile(0.95):.1f} ms")

        print(f"  |dt| max:          {abs_dt.max():.1f} ms")

    # =================================================================
    # SAVE SESSION
    # =================================================================

    all_merged.append(merged)


# =====================================================================
# CHECK
# =====================================================================

if not all_merged:
    raise RuntimeError("No sessions were successfully merged.")


# =====================================================================
# CONCATENATE
# =====================================================================

full = pd.concat(all_merged, ignore_index=True)


# =====================================================================
# FINAL SORT
# =====================================================================

full = full.sort_values(
    [
        "device_id",
        "session_id",
        "UnixTimeMillis",
        "ConstellationType",
        "Svid",
        "CarrierFrequencyHz",
    ]
).reset_index(drop=True)


# =====================================================================
# SAVE
# =====================================================================

output_path = OUTPUT_DIR / "combined_corpus.parquet"


full.to_parquet(output_path, index=False, compression="snappy")


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print(" MERGE COMPLETE")
print("=" * 70)


print(f"\nTotal rows: {len(full):,}")

print(f"Sessions: {full['session_id'].nunique()}")

print(f"Devices: {full['device_id'].nunique()}")

print(f"Unique signals: {full['signal_id'].nunique():,}")

print(f"Output: {output_path}")

print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")


# =====================================================================
# DEVICE COUNTS
# =====================================================================

print("\nPer-device counts:")

print(full.groupby("device_id").size().sort_values(ascending=False))


# =====================================================================
# SESSION COUNTS
# =====================================================================

print("\nPer-session counts:")

print(full.groupby("session_id").size().sort_values(ascending=False))


# =====================================================================
# ORIENTATION SYNCHRONIZATION
# =====================================================================

print("\nOrientation synchronization |dt|:")

print(full["orientation_dt_ms"].abs().describe())


# =====================================================================
# TIME INTERVAL ANALYSIS
# =====================================================================
#
# IMPORTANT:
# Status.csv contains multiple satellites at each epoch.
#
# Therefore, we first remove duplicate timestamps before calculating
# epoch intervals.
# =====================================================================

epoch_times = (
    full[
        [
            "device_id",
            "session_id",
            "UnixTimeMillis",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "device_id",
            "session_id",
            "UnixTimeMillis",
        ]
    )
)


epoch_times["epoch_dt_ms"] = epoch_times.groupby(
    [
        "device_id",
        "session_id",
    ]
)["UnixTimeMillis"].diff()


print("\nEpoch interval statistics:")

print(epoch_times["epoch_dt_ms"].dropna().describe())


print("\nMedian epoch interval:")

print(f"{epoch_times['epoch_dt_ms'].median():.1f} ms")


# =====================================================================
# FINAL COLUMN LIST
# =====================================================================

print("\nColumns:")

for col in full.columns:
    print(f"  {col}")


print("\nDone.")
