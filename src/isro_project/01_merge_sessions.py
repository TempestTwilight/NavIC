#!/usr/bin/env python3
"""
Merge Status with OrientationDeg using DuckDB ASOF join.

Produces: results/merged_data/combined_corpus.parquet

Memory-constrained processing (2GB limit) with session isolation.
"""

from pathlib import Path
import logging
import sys
import duckdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "merged_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "combined_corpus.parquet"
TEMP_FILE = OUTPUT_DIR / "combined_corpus.tmp.parquet"

GNSS_PARQUET_DIR = PROJECT_ROOT / "gnss-eda" / "data" / "parquet"
STATUS_PQ = GNSS_PARQUET_DIR / "Status.parquet"
ORIENT_PQ = GNSS_PARQUET_DIR / "OrientationDeg.parquet"

CATEGORIZED_DIR = PROJECT_ROOT / "data" / "Categorized_Data"

MAX_ORIENTATION_DT_MS = 500  # Maximum timestamp difference tolerance


def sql_escape(path: Path) -> str:
    """Escape path for SQL string literal."""
    return str(path.resolve()).replace("'", "''")


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
        status_with_sess AS (
            SELECT 
                s.UnixTimeMillis, s.SignalCount, s.SignalIndex, s.ConstellationType, s.Svid,
                s.CarrierFrequencyHz, s.Cn0DbHz, s.AzimuthDegrees, s.ElevationDegrees,
                s.UsedInFix, s.HasAlmanacData, s.HasEphemerisData, s.BasebandCn0DbHz,
                sm.session_id, 'oneplus' as device_id
            FROM read_parquet('{status_path}') s
            JOIN session_map sm ON s.UnixTimeMillis = sm.UnixTimeMillis
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
                || '_' || CAST(s.Svid AS VARCHAR) || '_' || CAST(s.CarrierFrequencyHz AS VARCHAR) as signal_id
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
                regexp_extract(filename, '([^/]+)/Status\\\\.csv$', 1) as session_id,
                'oneplus' as device_id
            FROM read_csv('{status_glob}', 
                header=true, 
                skip=0,
                filename=true,
                ignore_errors=false,
                all_varchar=false)
            WHERE column7 IS NOT NULL AND CAST(column7 AS DOUBLE) > 0
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
                || '_' || CAST(s.Svid AS VARCHAR) || '_' || CAST(s.CarrierFrequencyHz AS VARCHAR) as signal_id
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
    log.info(f"  Orientation dt (ms): min={stats[3]}, max={stats[4]}, avg={stats[5]:.2f}")
    
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
        'UnixTimeMillis', 'SignalCount', 'SignalIndex', 'ConstellationType', 'Svid',
        'CarrierFrequencyHz', 'Cn0DbHz', 'AzimuthDegrees', 'ElevationDegrees',
        'UsedInFix', 'HasAlmanacData', 'HasEphemerisData', 'BasebandCn0DbHz',
        'utcTimeMillis', 'elapsedRealtimeNanos', 'yawDeg', 'rollDeg', 'pitchDeg',
        'CalibrationAccuracy', 'orientation_dt_ms', 'session_id', 'device_id', 'signal_id'
    }
    actual_cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{temp_path}')").df()['column_name'])
    
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
    log.info("=" * 70)
    log.info(" GNSS + Orientation Merge (DuckDB)")
    log.info("=" * 70)
    
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB';")
    con.execute("SET temp_directory = '/tmp/duckdb_temp';")
    con.execute("SET preserve_insertion_order = false;")
    
    try:
        # Determine source and merge
        if STATUS_PQ.exists() and ORIENT_PQ.exists():
            merge_parquet_sources(con)
        elif CATEGORIZED_DIR.exists():
            merge_csv_sources(con)
        else:
            log.error("No valid data sources found")
            log.error(f"  Checked: {GNSS_PARQUET_DIR}")
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
