"""
Extract features for baseline experiment using DuckDB.

This script processes each session independently to stay within memory limits.
All heavy computation is done in SQL; only aggregated results are exported.
"""

from pathlib import Path
import sys
import duckdb
import pandas as pd

def tqdm(it, *args, **kwargs):
    return it

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.config import (  # noqa: E402
    CN0_MAX,
    CN0_MIN,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    ELEVATION_MAX,
    ELEVATION_MIN,
    META_CSV,
    TYPED_DIR,
)
from baseline.config import FEATURES_DIR  # noqa: E402


def get_session_list():
    """Get list of sessions with recording dates."""
    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    query = f"""
    SELECT 
        session_id,
        device_name,
        CASE 
            WHEN session_id REGEXP 'gnss_log_[0-9]{{4}}_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+' THEN
                CAST(
                    regexp_extract(session_id, 'gnss_log_([0-9]{{4}})_([0-9]+)_([0-9]+)', 1) || '-' ||
                    LPAD(regexp_extract(session_id, 'gnss_log_([0-9]{{4}})_([0-9]+)_([0-9]+)', 2), 2, '0') || '-' ||
                    LPAD(regexp_extract(session_id, 'gnss_log_([0-9]{{4}})_([0-9]+)_([0-9]+)', 3), 2, '0')
                    AS DATE
                )
            ELSE NULL
        END AS recording_date
    FROM read_csv_auto('{META_CSV}')
    ORDER BY recording_date, device_name
    """

    sessions = con.execute(query).fetchdf()
    con.close()
    return sessions


def extract_features_for_session(session_id: str, device_name: str) -> pd.DataFrame:
    """Extract features for a single session."""

    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{DUCKDB_TEMP_DIR}'")

    try:
        # Step 1: Get Raw and Status for this session
        con.execute(f"""
            CREATE TABLE raw_session AS
            SELECT 
                session,
                utcTimeMillis,
                CAST(ROUND(utcTimeMillis / 1000.0) AS BIGINT) AS sec_epoch,
                ConstellationType,
                Svid,
                CarrierFrequencyHz,
                CodeType,
                Cn0DbHz,
                BasebandCn0DbHz
            FROM read_parquet('{TYPED_DIR}/Raw.parquet')
            WHERE session = '{session_id}'
              AND Cn0DbHz > 0
              AND Cn0DbHz >= {CN0_MIN}
              AND Cn0DbHz <= {CN0_MAX}
        """)

        # Step 2: Get Status (elevation, azimuth)
        con.execute(f"""
            CREATE TABLE status_session AS
            SELECT 
                session,
                CAST(ROUND(utcTimeMillis / 1000.0) AS BIGINT) AS sec_epoch,
                ConstellationType,
                Svid,
                CarrierFrequencyHz,
                ElevationDegrees,
                AzimuthDegrees
            FROM read_parquet('{TYPED_DIR}/Status.parquet')
            WHERE session = '{session_id}'
              AND ElevationDegrees >= {ELEVATION_MIN}
              AND ElevationDegrees <= {ELEVATION_MAX}
        """)

        # Step 3: Join Raw and Status
        con.execute("""
            CREATE TABLE joined AS
            SELECT 
                r.*,
                s.ElevationDegrees,
                s.AzimuthDegrees
            FROM raw_session r
            JOIN status_session s
                ON r.sec_epoch = s.sec_epoch
                AND r.ConstellationType = s.ConstellationType
                AND r.Svid = s.Svid
                AND r.CarrierFrequencyHz = s.CarrierFrequencyHz
        """)

        # Step 4: Add lag/lead features (window functions)
        con.execute("""
            CREATE TABLE features AS
            SELECT 
                *,
                -- Future C/N0 at +15s (target)
                LEAD(Cn0DbHz, 15) OVER (
                    PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
                    ORDER BY sec_epoch
                ) AS cn0_future_15s,
                -- Past C/N0 values for jitter
                LAG(Cn0DbHz, 1) OVER (
                    PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
                    ORDER BY sec_epoch
                ) AS cn0_lag_1,
                LAG(Cn0DbHz, 2) OVER (
                    PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
                    ORDER BY sec_epoch
                ) AS cn0_lag_2,
                LAG(Cn0DbHz, 3) OVER (
                    PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
                    ORDER BY sec_epoch
                ) AS cn0_lag_3,
                LAG(Cn0DbHz, 4) OVER (
                    PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
                    ORDER BY sec_epoch
                ) AS cn0_lag_4,
                LAG(Cn0DbHz, 5) OVER (
                    PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
                    ORDER BY sec_epoch
                ) AS cn0_lag_5,
                -- Row number for track length check
                ROW_NUMBER() OVER (
                    PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
                    ORDER BY sec_epoch
                ) AS track_row_num
            FROM joined
        """)

        # Step 5: Compute jitter and target, filter valid rows
        result = con.execute("""
            SELECT 
                session,
                sec_epoch,
                ConstellationType,
                Svid,
                CarrierFrequencyHz,
                CodeType,
                Cn0DbHz AS current_cn0,
                ElevationDegrees,
                SIN(RADIANS(ElevationDegrees)) AS sin_elevation,
                AzimuthDegrees,
                -- Jitter: std of past 5 values
                CASE 
                    WHEN cn0_lag_1 IS NOT NULL 
                         AND cn0_lag_2 IS NOT NULL 
                         AND cn0_lag_3 IS NOT NULL 
                         AND cn0_lag_4 IS NOT NULL 
                         AND cn0_lag_5 IS NOT NULL 
                    THEN STDDEV_SAMP(Cn0DbHz) OVER (
                        PARTITION BY ConstellationType, Svid, CarrierFrequencyHz, CodeType
                        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                    )
                    ELSE NULL
                END AS sigma_5s,
                -- Recent delta
                Cn0DbHz - cn0_lag_5 AS delta_recent_5s,
                -- Target
                cn0_future_15s,
                cn0_future_15s - Cn0DbHz AS delta_cn0_15s,
                -- Fading flag
                CASE WHEN ABS(cn0_future_15s - Cn0DbHz) > 1.5 THEN 1 ELSE 0 END AS is_fading
            FROM features
            WHERE cn0_future_15s IS NOT NULL
              AND track_row_num >= 5
        """).fetchdf()

        # Add device name
        result["device_name"] = device_name

        return result

    finally:
        con.close()


def main():
    """Extract features for all sessions."""
    print("=" * 60)
    print("EXTRACTING FEATURES FOR BASELINE EXPERIMENT")
    print("=" * 60)

    # Get session list
    sessions = get_session_list()
    print(f"\nTotal sessions: {len(sessions)}")
    print(f"Devices: {sessions['device_name'].unique().tolist()}")
    print(
        f"Date range: {sessions['recording_date'].min()} to {sessions['recording_date'].max()}"
    )

    # Process each session
    all_features = []

    print("\nProcessing sessions:")
    for _, row in tqdm(sessions.iterrows(), total=len(sessions), desc="Sessions"):
        session_id = row["session_id"]
        device_name = row["device_name"]
        recording_date = row["recording_date"]

        try:
            features = extract_features_for_session(session_id, device_name)
            features["recording_date"] = recording_date

            if len(features) > 0:
                all_features.append(features)

        except Exception as e:
            print(f"\n  Warning: Session {session_id} failed: {e}")
            continue

    # Concatenate all features
    if all_features:
        combined = pd.concat(all_features, ignore_index=True)

        # Save to parquet
        output_path = FEATURES_DIR / "features_combined.parquet"
        combined.to_parquet(output_path, index=False)

        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"  Total rows: {len(combined):,}")
        print(f"  Total sessions: {combined['session'].nunique()}")
        print(
            f"  Date range: {combined['recording_date'].min()} to {combined['recording_date'].max()}"
        )
        print("\n  Features per device:")
        print(combined.groupby("device_name").size().to_string())
        print(f"\n  Output saved to: {output_path}")

        # Quick stats
        print("\n  Target (delta_cn0_15s) statistics:")
        print(f"    Mean: {combined['delta_cn0_15s'].mean():.3f} dB")
        print(f"    Std:  {combined['delta_cn0_15s'].std():.3f} dB")
        print(f"    Min:  {combined['delta_cn0_15s'].min():.3f} dB")
        print(f"    Max:  {combined['delta_cn0_15s'].max():.3f} dB")

        # Fading share
        fading_pct = combined["is_fading"].mean() * 100
        print(f"\n  Fading events (|Δ| > 1.5 dB): {fading_pct:.1f}%")

    else:
        print("\n  ERROR: No features extracted!")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
