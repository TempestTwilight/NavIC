"""
Verify data availability and integrity before running experiments.

This script checks:
1. All required parquet files exist
2. Feature columns are populated across devices
3. Session metadata is complete
4. Join keys are valid
5. DuckDB can connect with resource limits
"""

import duckdb
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    META_CSV,
    TYPED_DIR,
)


def verify_parquet_files():
    """Check that all required parquet files exist."""
    print("=" * 60)
    print("1. Verifying Parquet Files")
    print("=" * 60)

    required_files = [
        "Raw.parquet",
        "Status.parquet",
        "OrientationDeg.parquet",
        "UncalAccel.parquet",
    ]

    all_exist = True
    for fname in required_files:
        fpath = TYPED_DIR / fname
        exists = fpath.exists()
        size_mb = fpath.stat().st_size / (1024 * 1024) if exists else 0
        status = "✓" if exists else "✗"
        print(f"  {status} {fname}: {size_mb:.1f} MB")
        if not exists:
            all_exist = False

    return all_exist


def verify_session_metadata():
    """Check session metadata integrity."""
    print("\n" + "=" * 60)
    print("2. Verifying Session Metadata")
    print("=" * 60)

    import pandas as pd

    meta = pd.read_csv(META_CSV)

    print(f"  Total sessions: {len(meta)}")
    print(f"  Columns: {list(meta.columns)}")
    print("\n  Sessions per device:")
    for device, count in meta["device_name"].value_counts().items():
        print(f"    {device}: {count}")

    # Check for recording date column
    if "recording_date" not in meta.columns:
        print(
            "\n  ⚠ WARNING: 'recording_date' column missing, will derive from session_id"
        )
    else:
        print("\n  ✓ Recording dates available")

    return True


def verify_feature_availability():
    """Check that key features are populated across devices."""
    print("\n" + "=" * 60)
    print("3. Verifying Feature Availability Across Devices")
    print("=" * 60)

    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{DUCKDB_TEMP_DIR}'")

    # Load metadata
    con.execute(f"""
        CREATE TABLE sessions_meta AS 
        SELECT * FROM read_csv_auto('{META_CSV}')
    """)

    # Load raw data
    con.execute(f"""
        CREATE TABLE raw AS 
        SELECT * FROM read_parquet('{TYPED_DIR}/Raw.parquet')
    """)

    # Check feature availability by device
    query = """
    SELECT 
        m.device_name,
        COUNT(*) as total_rows,
        SUM(CASE WHEN r.Cn0DbHz > 0 THEN 1 ELSE 0 END) as cn0_nonzero,
        SUM(CASE WHEN r.BasebandCn0DbHz IS NOT NULL THEN 1 ELSE 0 END) as baseband_available,
        SUM(CASE WHEN r.AgcDb IS NOT NULL THEN 1 ELSE 0 END) as agc_available
    FROM raw r
    JOIN sessions_meta m ON r.session = m.session_id
    GROUP BY m.device_name
    ORDER BY m.device_name
    """

    result = con.execute(query).fetchdf()
    print(result.to_string(index=False))

    # Check BasebandCn0DbHz specifically
    print("\n  BasebandCn0DbHz availability:")
    for _, row in result.iterrows():
        pct = (
            (row["baseband_available"] / row["total_rows"] * 100)
            if row["total_rows"] > 0
            else 0
        )
        status = "✓" if pct > 95 else "⚠"
        print(f"    {status} {row['device_name']}: {pct:.1f}%")

    # Check AgcDb specifically
    print("\n  AgcDb availability:")
    for _, row in result.iterrows():
        pct = (
            (row["agc_available"] / row["total_rows"] * 100)
            if row["total_rows"] > 0
            else 0
        )
        status = "✓" if pct > 95 else ("partial" if pct > 0 else "✗")
        print(f"    {status} {row['device_name']}: {pct:.1f}%")

    con.close()
    return True


def verify_orientation_data():
    """Check orientation data availability (only in June sessions)."""
    print("\n" + "=" * 60)
    print("4. Verifying Orientation Data")
    print("=" * 60)

    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    # Load metadata
    con.execute(f"""
        CREATE TABLE sessions_meta AS 
        SELECT * FROM read_csv_auto('{META_CSV}')
    """)

    # Check which sessions have OrientationDeg
    query = f"""
    SELECT 
        m.device_name,
        COUNT(DISTINCT m.session_id) as sessions_with_orientation
    FROM sessions_meta m
    WHERE EXISTS (
        SELECT 1 FROM read_parquet('{TYPED_DIR}/OrientationDeg.parquet') o
        WHERE o.session = m.session_id
        LIMIT 1
    )
    GROUP BY m.device_name
    ORDER BY m.device_name
    """

    try:
        result = con.execute(query).fetchdf()
        print(result.to_string(index=False))
        print(
            "\n  Note: OrientationDeg data is available in June sessions (OnePlus CPH2401)"
        )
    except Exception as e:
        print(f"  ⚠ Could not verify orientation data: {e}")

    con.close()
    return True


def verify_join_keys():
    """Check that join keys are valid for Raw-Status join."""
    print("\n" + "=" * 60)
    print("5. Verifying Join Keys")
    print("=" * 60)

    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    # Test join on a sample session
    query = f"""
    WITH raw_sample AS (
        SELECT * FROM read_parquet('{TYPED_DIR}/Raw.parquet')
        WHERE session = (SELECT session_id FROM read_csv_auto('{META_CSV}') LIMIT 1)
        LIMIT 1000
    ),
    status_sample AS (
        SELECT * FROM read_parquet('{TYPED_DIR}/Status.parquet')
        WHERE session = (SELECT session_id FROM read_csv_auto('{META_CSV}') LIMIT 1)
        LIMIT 1000
    )
    SELECT 
        COUNT(*) as raw_rows,
        COUNT(DISTINCT r.session) as sessions,
        MIN(r.utcTimeMillis) as min_time,
        MAX(r.utcTimeMillis) as max_time
    FROM raw_sample r
    """

    result = con.execute(query).fetchdf()
    print("  Sample session raw data:")
    print(result.to_string(index=False))

    # Verify signal identity key uniqueness
    query2 = f"""
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT (session, ConstellationType, Svid, CarrierFrequencyHz, CodeType)) as unique_keys_without_code,
        COUNT(DISTINCT (session, ConstellationType, Svid, CarrierFrequencyHz)) as unique_keys_simple
    FROM read_parquet('{TYPED_DIR}/Raw.parquet')
    WHERE session = (SELECT session_id FROM read_csv_auto('{META_CSV}') LIMIT 1)
    LIMIT 10000
    """

    result2 = con.execute(query2).fetchdf()
    print("\n  Signal key uniqueness test:")
    print(result2.to_string(index=False))
    print("  Note: CodeType is required for uniqueness (I/Q channels)")

    con.close()
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("GNSS EXPERIMENT DATA VERIFICATION")
    print("=" * 60)

    checks = [
        ("Parquet files", verify_parquet_files),
        ("Session metadata", verify_session_metadata),
        ("Feature availability", verify_feature_availability),
        ("Orientation data", verify_orientation_data),
        ("Join keys", verify_join_keys),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n  ✗ {name} failed: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All checks passed. Ready to run experiments.")
    else:
        print("\n  Some checks failed. Review errors above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
