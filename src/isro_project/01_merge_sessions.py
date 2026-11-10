#!/usr/bin/env python3
"""
Merge Status with OrientationDeg using DuckDB ASOF join.

Produces: results/merged_data/combined_corpus.parquet

Memory-constrained processing (2GB limit) with session isolation.

Device ID extraction
--------------------
Real device identity (Manufacturer_Model) is parsed from the '# Version:' header
line of each raw Android GNSS Logger .txt file before ingestion.  The mapping is
written to a side-table so DuckDB can attach it per session without loading raw
CSVs into Python heap.

Body-frame satellite vectors
----------------------------
Three body-frame components  (s_body_x, s_body_y, s_body_z)  are computed for
every merged row using the rotation matrix derived from the phone's Euler angles
at the matched IMU epoch.  This eliminates gimbal-lock singularities that arise
when pitch approaches ±90° when using raw yaw/pitch/roll directly for the
incidence angle dot-product.

  R = Rz(yaw) · Ry(pitch) · Rx(roll)   (ZYX / Tait-Bryan convention)

  s_enu = [cos(el)·sin(az), cos(el)·cos(az), sin(el)]ᵀ   (ENU)

  s_body = Rᵀ · s_enu

  s_body_x : satellite component along phone width  (left ← right)
  s_body_y : satellite component along phone length (bottom ← top)
  s_body_z : satellite component normal to screen   (front: +Z out of glass)
"""

import re
from pathlib import Path
import logging
import sys
import numpy as np
import pandas as pd
import duckdb

from .config import (
    CATEGORIZED_DIR,
    GNSS_PARQUET_DIR,
    MERGED_OUTPUT_DIR,
    MAX_ORIENTATION_DT_MS,
    DUCKDB_MEMORY_LIMIT,
    RAW_LOGS_DIR,
    ensure_dirs,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

OUTPUT_DIR = MERGED_OUTPUT_DIR
OUTPUT_FILE = OUTPUT_DIR / "combined_corpus.parquet"
TEMP_FILE = OUTPUT_DIR / "combined_corpus.tmp.parquet"

STATUS_PQ = GNSS_PARQUET_DIR / "Status.parquet"
ORIENT_PQ = GNSS_PARQUET_DIR / "OrientationDeg.parquet"

# Raw GNSS Logger .txt directory (used for device-ID header parsing only)
RAW_LOGS_DIR = Path(__file__).resolve().parents[2] / "data" / "Raw_Logs"


# ------------------------------------------------------------------ #
#  Device-ID helpers                                                   #
# ------------------------------------------------------------------ #

_HEADER_RE = re.compile(
    r"Manufacturer:\s*(\S+)\s+Model:\s*(\S+)", re.IGNORECASE
)


def _parse_device_id(path: Path) -> str:
    """Return '<Manufacturer>_<Model>' from the first '# Version:' header line.

    Falls back to 'unknown' if the file cannot be read or the pattern is absent.
    """
    try:
        with open(path, "r", errors="ignore") as fh:
            for _ in range(15):          # header is always in the first ~10 lines
                line = fh.readline()
                if not line:
                    break
                m = _HEADER_RE.search(line)
                if m:
                    return f"{m.group(1)}_{m.group(2)}"
    except OSError:
        pass
    return "unknown"


def build_device_id_map(raw_logs_dir: Path) -> dict[str, str]:
    """Scan all .txt files under raw_logs_dir and return {stem: device_id}.

    The stem of the log file (e.g. 'gnss_log_2026_10_22_09_46_38') is used as
    a lookup key.  If multiple files share a stem (shouldn't happen), the last
    one wins.
    """
    mapping: dict[str, str] = {}
    if not raw_logs_dir.exists():
        log.warning(f"Raw logs directory not found: {raw_logs_dir}")
        return mapping
    for txt in raw_logs_dir.rglob("*.txt"):
        mapping[txt.stem] = _parse_device_id(txt)
    log.info(f"Device ID map: {len(mapping)} logs → {set(mapping.values())}")
    return mapping


def build_device_time_intervals(raw_logs_dir: Path) -> pd.DataFrame:
    """Scan raw .txt logs and return a DataFrame of (device_id, min_time, max_time).

    Each row represents one raw log file's time span, extracted by scanning the
    'Status,' lines for UnixTimeMillis values.  This lets us join sessions
    (reconstructed from the combined Parquet) back to their originating device.
    """
    rows: list[dict] = []
    if not raw_logs_dir.exists():
        log.warning(f"Raw logs directory not found: {raw_logs_dir}")
        return pd.DataFrame(columns=["device_id", "min_time", "max_time"])

    for txt in raw_logs_dir.rglob("*.txt"):
        dev = _parse_device_id(txt)
        if dev == "unknown":
            continue
        min_t: int | None = None
        max_t: int | None = None
        try:
            with open(txt, "r", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("Status,"):
                        parts = line.split(",")
                        try:
                            t = int(parts[1])
                            if min_t is None or t < min_t:
                                min_t = t
                            if max_t is None or t > max_t:
                                max_t = t
                        except (ValueError, IndexError):
                            pass
        except OSError:
            continue
        if min_t is not None:
            rows.append({"device_id": dev, "min_time": min_t, "max_time": max_t})

    df = pd.DataFrame(rows)
    if len(df):
        log.info(
            f"Device time intervals: {len(df)} logs across "
            f"{df['device_id'].nunique()} devices"
        )
    else:
        log.warning("No device time intervals found — device_id will default to 'unknown'")
    return df


def register_device_intervals(con: duckdb.DuckDBPyConnection, raw_logs_dir: Path) -> None:
    """Build and register the device_intervals table in DuckDB.

    Creates a DuckDB table `device_intervals(device_id, min_time, max_time)` so
    both merge paths can join session time-bounds against it to resolve device_id.
    """
    intervals_df = build_device_time_intervals(raw_logs_dir)
    con.register("_device_intervals_df", intervals_df)
    con.execute("""
        CREATE OR REPLACE TABLE device_intervals AS
        SELECT * FROM _device_intervals_df
    """)
    con.unregister("_device_intervals_df")


def sql_escape(path: Path) -> str:
    """Escape path for SQL string literal."""
    return str(path.resolve()).replace("'", "''")


# ------------------------------------------------------------------ #
#  Body-frame satellite vector computation                             #
# ------------------------------------------------------------------ #

def add_body_frame_vectors(df) -> "pd.DataFrame":
    """Append  s_body_x / s_body_y / s_body_z  columns to a merged corpus DataFrame.

    Inputs (must already exist as columns):
        AzimuthDegrees, ElevationDegrees  – satellite sky position
        yawDeg, pitchDeg, rollDeg         – phone Euler angles from OrientationDeg

    The satellite unit vector in East-North-Up (ENU) coordinates is:

        s_east  = cos(el) · sin(az)
        s_north = cos(el) · cos(az)
        s_up    = sin(el)

    The rotation matrix from ENU (world) → body frame is:

        R = Rz(yaw) · Ry(pitch) · Rx(roll)

    Following Android's sensor Euler order (ZYX / Tait-Bryan), where:
        yaw   φ : rotation about world Z (up)
        pitch θ : rotation about intermediate Y
        roll  ψ : rotation about body X

        |cφcθ   cφsθsψ−sφcψ   cφsθcψ+sφsψ|
    R = |sφcθ   sφsθsψ+cφcψ   sφsθcψ−cφsψ|
        |−sθ    cθsψ           cθcψ        |

    s_body = R · s_enu   (3-vector in phone body frame)

      s_body_x : left ← right across screen
      s_body_y : bottom ← top along screen
      s_body_z : out of glass face (+) / into back (−)

    Returns the DataFrame with three new float64 columns appended.
    Rows with missing angle data are filled with NaN rather than dropped,
    so the calling code can decide how to handle them.
    """
    import pandas as pd

    az  = np.deg2rad(df["AzimuthDegrees"].to_numpy(dtype=float))
    el  = np.deg2rad(df["ElevationDegrees"].to_numpy(dtype=float))
    yaw = np.deg2rad(df["yawDeg"].to_numpy(dtype=float))
    pit = np.deg2rad(df["pitchDeg"].to_numpy(dtype=float))
    rol = np.deg2rad(df["rollDeg"].to_numpy(dtype=float))

    # Satellite LOS unit vector in ENU
    s_e = np.cos(el) * np.sin(az)
    s_n = np.cos(el) * np.cos(az)
    s_u = np.sin(el)

    # Trig shorthand
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pit), np.sin(pit)
    cr, sr = np.cos(rol), np.sin(rol)

    # Rows of R (world→body, ZYX Tait-Bryan)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    df = df.copy()
    df["s_body_x"] = np.clip(r00 * s_e + r01 * s_n + r02 * s_u, -1.0, 1.0)
    df["s_body_y"] = np.clip(r10 * s_e + r11 * s_n + r12 * s_u, -1.0, 1.0)
    df["s_body_z"] = np.clip(r20 * s_e + r21 * s_n + r22 * s_u, -1.0, 1.0)
    return df


def merge_parquet_sources(con: duckdb.DuckDBPyConnection):
    """Merge Parquet sources with session reconstruction and isolation."""
    log.info(f"Processing Parquet: {GNSS_PARQUET_DIR}")

    status_path = sql_escape(STATUS_PQ)
    orient_path = sql_escape(ORIENT_PQ)
    temp_path = sql_escape(TEMP_FILE)

    # Deduplicate orientation by (utcTimeMillis, elapsedRealtimeNanos) - keep last
    query = f"""
    COPY (
        WITH orient_dedup AS (
            SELECT * FROM (
                SELECT *, 
                    ROW_NUMBER() OVER (
                        PARTITION BY utcTimeMillis 
                        ORDER BY elapsedRealtimeNanos DESC
                    ) as rn
                FROM read_parquet('{orient_path}')
            ) WHERE rn = 1
        ),
        distinct_times AS (
            SELECT DISTINCT UnixTimeMillis
            FROM read_parquet('{status_path}')
        ),
        with_gap AS (
            SELECT 
                UnixTimeMillis,
                CASE WHEN UnixTimeMillis - LAG(UnixTimeMillis, 1, UnixTimeMillis) 
                         OVER (ORDER BY UnixTimeMillis) > 10000 
                     THEN 1 ELSE 0 END as is_new_sess
            FROM distinct_times
        ),
        session_map AS (
            SELECT 
                UnixTimeMillis,
                'session_' || CAST(SUM(is_new_sess) OVER (ORDER BY UnixTimeMillis) AS VARCHAR) as session_id
            FROM with_gap
        ),
        session_bounds AS (
            SELECT 
                session_id,
                MIN(UnixTimeMillis) as sess_start,
                MAX(UnixTimeMillis) as sess_end
            FROM session_map
            GROUP BY session_id
        ),
        session_devices AS (
            SELECT 
                sb.session_id,
                COALESCE(FIRST(di.device_id), 'oneplus') as device_id
            FROM session_bounds sb
            LEFT JOIN device_intervals di
              ON di.min_time <= sb.sess_end + 2000 AND di.max_time >= sb.sess_start - 2000
            GROUP BY sb.session_id
        ),
        status_with_sess AS (
            SELECT 
                s.UnixTimeMillis, s.SignalCount, s.SignalIndex, s.ConstellationType, s.Svid,
                s.CarrierFrequencyHz, s.Cn0DbHz, s.AzimuthDegrees, s.ElevationDegrees,
                s.UsedInFix, s.HasAlmanacData, s.HasEphemerisData, s.BasebandCn0DbHz,
                sm.session_id, sd.device_id
            FROM read_parquet('{status_path}') s
            JOIN session_map sm ON s.UnixTimeMillis = sm.UnixTimeMillis
            JOIN session_devices sd ON sm.session_id = sd.session_id
            WHERE s.Cn0DbHz > 0
        )
        SELECT 
            s.UnixTimeMillis, s.SignalCount, s.SignalIndex, s.ConstellationType, s.Svid,
            s.CarrierFrequencyHz, s.Cn0DbHz, s.AzimuthDegrees, s.ElevationDegrees,
            s.UsedInFix, s.HasAlmanacData, s.HasEphemerisData, s.BasebandCn0DbHz,
            o.utcTimeMillis, o.elapsedRealtimeNanos, o.yawDeg, o.rollDeg, o.pitchDeg,
            o.CalibrationAccuracy,
            (s.UnixTimeMillis - o.utcTimeMillis) as orientation_dt_ms,
            s.session_id, s.device_id,
            s.device_id || '_' || s.session_id || '_' || CAST(s.ConstellationType AS VARCHAR) 
                || '_' || CAST(s.Svid AS VARCHAR) || '_' || CAST(s.CarrierFrequencyHz AS VARCHAR) as signal_id,
            -- Body-frame satellite vector components via ZYX rotation matrix
            GREATEST(-1.0, LEAST(1.0,
                (cos(radians(o.yawDeg)) * cos(radians(o.pitchDeg))) * (cos(radians(s.ElevationDegrees)) * sin(radians(s.AzimuthDegrees))) +
                (cos(radians(o.yawDeg)) * sin(radians(o.pitchDeg)) * sin(radians(o.rollDeg)) - sin(radians(o.yawDeg)) * cos(radians(o.rollDeg))) * (cos(radians(s.ElevationDegrees)) * cos(radians(s.AzimuthDegrees))) +
                (cos(radians(o.yawDeg)) * sin(radians(o.pitchDeg)) * cos(radians(o.rollDeg)) + sin(radians(o.yawDeg)) * sin(radians(o.rollDeg))) * sin(radians(s.ElevationDegrees))
            )) as s_body_x,
            GREATEST(-1.0, LEAST(1.0,
                (sin(radians(o.yawDeg)) * cos(radians(o.pitchDeg))) * (cos(radians(s.ElevationDegrees)) * sin(radians(s.AzimuthDegrees))) +
                (sin(radians(o.yawDeg)) * sin(radians(o.pitchDeg)) * sin(radians(o.rollDeg)) + cos(radians(o.yawDeg)) * cos(radians(o.rollDeg))) * (cos(radians(s.ElevationDegrees)) * cos(radians(s.AzimuthDegrees))) +
                (sin(radians(o.yawDeg)) * sin(radians(o.pitchDeg)) * cos(radians(o.rollDeg)) - cos(radians(o.yawDeg)) * sin(radians(o.rollDeg))) * sin(radians(s.ElevationDegrees))
            )) as s_body_y,
            GREATEST(-1.0, LEAST(1.0,
                (-sin(radians(o.pitchDeg))) * (cos(radians(s.ElevationDegrees)) * sin(radians(s.AzimuthDegrees))) +
                (cos(radians(o.pitchDeg)) * sin(radians(o.rollDeg))) * (cos(radians(s.ElevationDegrees)) * cos(radians(s.AzimuthDegrees))) +
                (cos(radians(o.pitchDeg)) * cos(radians(o.rollDeg))) * sin(radians(s.ElevationDegrees))
            )) as s_body_z
        FROM status_with_sess s
        ASOF LEFT JOIN orient_dedup o
          ON s.UnixTimeMillis >= o.utcTimeMillis
        WHERE o.utcTimeMillis IS NOT NULL 
          AND (s.UnixTimeMillis - o.utcTimeMillis) <= {MAX_ORIENTATION_DT_MS}
    ) TO '{temp_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)


def merge_csv_sources(con: duckdb.DuckDBPyConnection):
    """Merge CSV sources with explicit schema and session isolation."""
    log.info(f"Processing CSV: {CATEGORIZED_DIR}")

    status_glob = sql_escape(CATEGORIZED_DIR / "*" / "Status.csv")
    orient_glob = sql_escape(CATEGORIZED_DIR / "*" / "OrientationDeg.csv")
    temp_path = sql_escape(TEMP_FILE)

    query = f"""
    COPY (
        WITH status_raw AS (
     SELECT 
         CAST(column1 AS BIGINT) as UnixTimeMillis,
         CAST(column2 AS BIGINT) as SignalCount,
         CAST(column3 AS BIGINT) as SignalIndex,
         CAST(column4 AS BIGINT) as ConstellationType,
         CAST(column5 AS BIGINT) as Svid,
         CAST(column6 AS BIGINT) as CarrierFrequencyHz,
         CAST(column7 AS DOUBLE) as Cn0DbHz,
         CAST(column8 AS DOUBLE) as AzimuthDegrees,
         CAST(column9 AS DOUBLE) as ElevationDegrees,
         CAST(column10 AS BIGINT) as UsedInFix,
         CAST(column11 AS BIGINT) as HasAlmanacData,
         CAST(column12 AS BIGINT) as HasEphemerisData,
         CAST(column13 AS DOUBLE) as BasebandCn0DbHz,
         regexp_extract(filename, '([^/]+)/Status\\\\.csv$', 1) as session_id
     FROM read_csv('{status_glob}', 
         header=true, 
         skip=0,
         filename=true,
         ignore_errors=false,
         all_varchar=false)
     WHERE column7 IS NOT NULL AND CAST(column7 AS DOUBLE) > 0
 ),
 session_bounds AS (
     SELECT 
         session_id,
         MIN(UnixTimeMillis) as sess_start,
         MAX(UnixTimeMillis) as sess_end
     FROM status_raw
     GROUP BY session_id
 ),
 session_devices AS (
     SELECT 
         sb.session_id,
         COALESCE(FIRST(di.device_id), 'oneplus') as device_id
     FROM session_bounds sb
     LEFT JOIN device_intervals di
       ON di.min_time <= sb.sess_end + 2000 AND di.max_time >= sb.sess_start - 2000
     GROUP BY sb.session_id
 ),
 status_with_sess AS (
     SELECT 
         sr.UnixTimeMillis, sr.SignalCount, sr.SignalIndex, sr.ConstellationType, sr.Svid,
         sr.CarrierFrequencyHz, sr.Cn0DbHz, sr.AzimuthDegrees, sr.ElevationDegrees,
         sr.UsedInFix, sr.HasAlmanacData, sr.HasEphemerisData, sr.BasebandCn0DbHz,
         sr.session_id, sd.device_id
     FROM status_raw sr
     JOIN session_devices sd ON sr.session_id = sd.session_id
 ),
        orient_raw AS (
            SELECT 
                CAST(column1 AS BIGINT) as utcTimeMillis,
                CAST(column2 AS BIGINT) as elapsedRealtimeNanos,
                CAST(column3 AS DOUBLE) as yawDeg,
                CAST(column4 AS DOUBLE) as rollDeg,
                CAST(column5 AS DOUBLE) as pitchDeg,
                CAST(column6 AS BIGINT) as CalibrationAccuracy,
                regexp_extract(filename, '([^/]+)/OrientationDeg\\\\.csv$', 1) as session_id
            FROM read_csv('{orient_glob}',
                header=true,
                skip=0,
                filename=true,
                ignore_errors=false,
                all_varchar=false)
        ),
        orient_dedup AS (
            SELECT * FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY session_id, utcTimeMillis 
                        ORDER BY elapsedRealtimeNanos DESC
                    ) as rn
                FROM orient_raw
            ) WHERE rn = 1
        )
        SELECT 
            s.UnixTimeMillis, s.SignalCount, s.SignalIndex, s.ConstellationType, s.Svid,
            s.CarrierFrequencyHz, s.Cn0DbHz, s.AzimuthDegrees, s.ElevationDegrees,
            s.UsedInFix, s.HasAlmanacData, s.HasEphemerisData, s.BasebandCn0DbHz,
            o.utcTimeMillis, o.elapsedRealtimeNanos, o.yawDeg, o.rollDeg, o.pitchDeg,
            o.CalibrationAccuracy,
            (s.UnixTimeMillis - o.utcTimeMillis) as orientation_dt_ms,
            s.session_id, s.device_id,
            s.device_id || '_' || s.session_id || '_' || CAST(s.ConstellationType AS VARCHAR) 
                || '_' || CAST(s.Svid AS VARCHAR) || '_' || CAST(s.CarrierFrequencyHz AS VARCHAR) as signal_id,
            -- Body-frame satellite vector components via ZYX rotation matrix
            GREATEST(-1.0, LEAST(1.0,
                (cos(radians(o.yawDeg)) * cos(radians(o.pitchDeg))) * (cos(radians(s.ElevationDegrees)) * sin(radians(s.AzimuthDegrees))) +
                (cos(radians(o.yawDeg)) * sin(radians(o.pitchDeg)) * sin(radians(o.rollDeg)) - sin(radians(o.yawDeg)) * cos(radians(o.rollDeg))) * (cos(radians(s.ElevationDegrees)) * cos(radians(s.AzimuthDegrees))) +
                (cos(radians(o.yawDeg)) * sin(radians(o.pitchDeg)) * cos(radians(o.rollDeg)) + sin(radians(o.yawDeg)) * sin(radians(o.rollDeg))) * sin(radians(s.ElevationDegrees))
            )) as s_body_x,
            GREATEST(-1.0, LEAST(1.0,
                (sin(radians(o.yawDeg)) * cos(radians(o.pitchDeg))) * (cos(radians(s.ElevationDegrees)) * sin(radians(s.AzimuthDegrees))) +
                (sin(radians(o.yawDeg)) * sin(radians(o.pitchDeg)) * sin(radians(o.rollDeg)) + cos(radians(o.yawDeg)) * cos(radians(o.rollDeg))) * (cos(radians(s.ElevationDegrees)) * cos(radians(s.AzimuthDegrees))) +
                (sin(radians(o.yawDeg)) * sin(radians(o.pitchDeg)) * cos(radians(o.rollDeg)) - cos(radians(o.yawDeg)) * sin(radians(o.rollDeg))) * sin(radians(s.ElevationDegrees))
            )) as s_body_y,
            GREATEST(-1.0, LEAST(1.0,
                (-sin(radians(o.pitchDeg))) * (cos(radians(s.ElevationDegrees)) * sin(radians(s.AzimuthDegrees))) +
                (cos(radians(o.pitchDeg)) * sin(radians(o.rollDeg))) * (cos(radians(s.ElevationDegrees)) * cos(radians(s.AzimuthDegrees))) +
                (cos(radians(o.pitchDeg)) * cos(radians(o.rollDeg))) * sin(radians(s.ElevationDegrees))
            )) as s_body_z
        FROM status_raw s
        ASOF LEFT JOIN orient_dedup o
          ON s.session_id = o.session_id 
         AND s.UnixTimeMillis >= o.utcTimeMillis
        WHERE o.utcTimeMillis IS NOT NULL
          AND (s.UnixTimeMillis - o.utcTimeMillis) <= {MAX_ORIENTATION_DT_MS}
    ) TO '{temp_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)


def validate_output(con: duckdb.DuckDBPyConnection):
    """Validate merged output."""
    temp_path = sql_escape(TEMP_FILE)

    stats = con.execute(f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT session_id) as sessions,
            COUNT(DISTINCT device_id) as devices,
            MIN(orientation_dt_ms) as min_dt,
            MAX(orientation_dt_ms) as max_dt,
            AVG(orientation_dt_ms) as avg_dt,
            COUNT(CASE WHEN orientation_dt_ms < 0 THEN 1 END) as negative_dt,
            COUNT(CASE WHEN orientation_dt_ms > {MAX_ORIENTATION_DT_MS} THEN 1 END) as excessive_dt,
            COUNT(CASE WHEN Cn0DbHz IS NULL OR Cn0DbHz <= 0 THEN 1 END) as invalid_cn0,
            COUNT(CASE WHEN yawDeg IS NULL THEN 1 END) as missing_yaw
        FROM read_parquet('{temp_path}')
    """).fetchone()

    if stats is None:
        log.error("Failed to query output statistics")
        return False

    log.info("Output validation:")
    log.info(f"  Total rows: {stats[0]:,}")
    log.info(f"  Sessions: {stats[1]}")
    log.info(f"  Devices: {stats[2]}")
    log.info(
        f"  Orientation dt (ms): min={stats[3]}, max={stats[4]}, avg={stats[5]:.2f}"
    )

    if stats[6] > 0:
        log.error(f"  ERROR: {stats[6]} records with negative orientation_dt_ms")
        return False
    if stats[7] > 0:
        log.error(f"  ERROR: {stats[7]} records exceed max dt tolerance")
        return False
    if stats[8] > 0:
        log.warning(f"  WARNING: {stats[8]} records with invalid Cn0DbHz")
    if stats[9] > 0:
        log.warning(f"  WARNING: {stats[9]} records missing orientation")

    # Schema validation
    expected_cols = {
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
        "utcTimeMillis",
        "elapsedRealtimeNanos",
        "yawDeg",
        "rollDeg",
        "pitchDeg",
        "CalibrationAccuracy",
        "orientation_dt_ms",
        "session_id",
        "device_id",
        "signal_id",
        "s_body_x",
        "s_body_y",
        "s_body_z",
    }
    actual_cols = set(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{temp_path}')").df()[
            "column_name"
        ]
    )

    if expected_cols != actual_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        if missing:
            log.error(f"  ERROR: Missing columns: {missing}")
        if extra:
            log.warning(f"  WARNING: Extra columns: {extra}")
        return False

    log.info("  Schema validation: PASS")
    return True


def main():
    ensure_dirs()
    log.info("=" * 70)
    log.info(" GNSS + Orientation Merge (DuckDB)")
    log.info("=" * 70)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}';")
    con.execute("SET temp_directory = '/tmp/duckdb_temp';")
    con.execute("SET preserve_insertion_order = false;")

    try:
        # Register raw log device intervals if available
        register_device_intervals(con, RAW_LOGS_DIR)

        # Determine source and merge
        if STATUS_PQ.exists() and ORIENT_PQ.exists():
            merge_parquet_sources(con)
        elif CATEGORIZED_DIR.exists():
            merge_csv_sources(con)
        else:
            log.error("No valid data sources found")
            log.error(f"  Checked: {STATUS_PQ}")
            log.error(f"  Checked: {ORIENT_PQ}")
            log.error(f"  Checked: {CATEGORIZED_DIR}")
            sys.exit(1)

        # Validate
        if not validate_output(con):
            log.error("Validation failed - output not saved")
            TEMP_FILE.unlink(missing_ok=True)
            sys.exit(1)

        # Atomic replace
        if OUTPUT_FILE.exists():
            backup = OUTPUT_DIR / "combined_corpus.backup.parquet"
            OUTPUT_FILE.rename(backup)
            log.info(f"Backed up existing output to {backup.name}")

        TEMP_FILE.rename(OUTPUT_FILE)
        log.info(f"✓ Success: {OUTPUT_FILE}")

    except Exception as e:
        log.error(f"Merge failed: {e}", exc_info=True)
        TEMP_FILE.unlink(missing_ok=True)
        sys.exit(1)
    finally:
        con.close()


if __name__ == "__main__":
    main()
