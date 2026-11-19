-- =============================================================================
-- Experiment A: Baseline Ridge Feature Extraction
-- =============================================================================
-- 
-- This query extracts features for the 15-second residual Δ C/N0 prediction task.
-- All aggregation is done in SQL; only final results are exported to Python.
--
-- Resource constraints:
--   - Memory limit: 1GB
--   - Threads: 2
--   - Process per session for lag/lead operations
--
-- Output: One row per (session, epoch, satellite, CodeType) with features and target
-- =============================================================================

-- Configuration (substitute via Python)
-- SET memory_limit='1GB';
-- SET threads=2;
-- SET temp_directory='/home/tempest/Documents/ISRO_Project/gnss-eda/scratch';

-- =============================================================================
-- Step 1: Create session metadata table with derived recording date
-- =============================================================================

CREATE OR REPLACE TABLE sessions_with_dates AS
SELECT 
    session_id,
    device_name,
    -- Derive recording date from session_id (format: gnss_log_YYYY_M_D_H_M_S)
    -- Some sessions have single-digit month/day, handle both formats
    CASE 
        WHEN session_id REGEXP 'gnss_log_[0-9]{4}_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+' THEN
            CAST(
                regexp_extract(session_id, 'gnss_log_([0-9]{4})_([0-9]+)_([0-9]+)', 1) || '-' ||
                LPAD(regexp_extract(session_id, 'gnss_log_([0-9]{4})_([0-9]+)_([0-9]+)', 2), 2, '0') || '-' ||
                LPAD(regexp_extract(session_id, 'gnss_log_([0-9]{4})_([0-9]+)_([0-9]+)', 3), 2, '0')
                AS DATE
            )
        ELSE NULL
    END AS recording_date
FROM read_csv_auto('${META_CSV}');

-- =============================================================================
-- Step 2: Filter Raw signals and join with Status for elevation/azimuth
-- =============================================================================

CREATE OR REPLACE TABLE raw_with_status AS
SELECT 
    r.session,
    r.utcTimeMillis,
    -- Round to second for join
    CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) AS sec_epoch,
    r.ConstellationType,
    r.Svid,
    r.CarrierFrequencyHz,
    r.CodeType,
    r.Cn0DbHz,
    r.BasebandCn0DbHz,
    r.AgcDb,
    -- Status fields (elevation, azimuth)
    s.ElevationDegrees,
    s.AzimuthDegrees,
    s.UsedInFix
FROM read_parquet('${TYPED_DIR}/Raw.parquet') r
LEFT JOIN read_parquet('${TYPED_DIR}/Status.parquet') s
    ON r.session = s.session
    AND CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = CAST(ROUND(s.utcTimeMillis / 1000.0) AS BIGINT)
    AND r.ConstellationType = s.ConstellationType
    AND r.Svid = s.Svid
    AND r.CarrierFrequencyHz = s.CarrierFrequencyHz
WHERE 
    -- Filter out zero C/N0 (reporting artifact)
    r.Cn0DbHz > 0
    -- Filter valid elevation
    AND s.ElevationDegrees >= 0
    AND s.ElevationDegrees <= 90
    -- Filter reasonable C/N0 range
    AND r.Cn0DbHz >= 10.0
    AND r.Cn0DbHz <= 55.0;

-- =============================================================================
-- Step 3: Add device info and recording date
-- =============================================================================

CREATE OR REPLACE TABLE signals_with_device AS
SELECT 
    r.*,
    m.device_name,
    m.recording_date
FROM raw_with_status r
JOIN sessions_with_dates m ON r.session = m.session_id;

-- =============================================================================
-- Step 4: Compute epoch-level common-mode signal (for target centering)
-- =============================================================================

CREATE OR REPLACE TABLE epoch_stats AS
SELECT 
    session,
    sec_epoch,
    AVG(Cn0DbHz) AS epoch_mean_cn0,
    STDDEV(Cn0DbHz) AS epoch_std_cn0,
    COUNT(*) AS epoch_n_signals
FROM signals_with_device
GROUP BY session, sec_epoch
HAVING COUNT(*) >= 4;  -- Require at least 4 signals for valid epoch

-- =============================================================================
-- Step 5: Join epoch stats and create centered C/N0
-- =============================================================================

CREATE OR REPLACE TABLE signals_centered AS
SELECT 
    s.*,
    e.epoch_mean_cn0,
    e.epoch_std_cn0,
    s.Cn0DbHz - e.epoch_mean_cn0 AS cn0_centered
FROM signals_with_device s
JOIN epoch_stats e 
    ON s.session = e.session 
    AND s.sec_epoch = e.sec_epoch;

-- =============================================================================
-- Step 6: Create lag and lead features (per signal track)
-- =============================================================================
-- This must be done per (session, ConstellationType, Svid, CarrierFrequencyHz, CodeType)
-- to avoid mixing different signal tracks

-- Note: Window functions over large partitions can be memory-intensive.
-- We process in batches or use row_number + self-join for efficiency.

CREATE OR REPLACE TABLE signal_tracks AS
SELECT 
    *,
    -- Row number within track for self-join approach
    ROW_NUMBER() OVER (
        PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ORDER BY sec_epoch
    ) AS track_row_num,
    -- Future C/N0 at +15s (target)
    LEAD(Cn0DbHz, 15) OVER (
        PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ORDER BY sec_epoch
    ) AS cn0_future_15s,
    -- Past C/N0 for jitter calculation
    LAG(Cn0DbHz, 1) OVER (
        PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ORDER BY sec_epoch
    ) AS cn0_lag_1,
    LAG(Cn0DbHz, 2) OVER (
        PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ORDER BY sec_epoch
    ) AS cn0_lag_2,
    LAG(Cn0DbHz, 3) OVER (
        PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ORDER BY sec_epoch
    ) AS cn0_lag_3,
    LAG(Cn0DbHz, 4) OVER (
        PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ORDER BY sec_epoch
    ) AS cn0_lag_4,
    LAG(Cn0DbHz, 5) OVER (
        PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz, CodeType
        ORDER BY sec_epoch
    ) AS cn0_lag_5
FROM signals_centered;

-- =============================================================================
-- Step 7: Compute rolling 5-second jitter (σ₅ₛ)
-- =============================================================================

CREATE OR REPLACE TABLE signals_with_jitter AS
SELECT 
    *,
    -- Jitter: std of past 5 C/N0 values
    CASE 
        WHEN cn0_lag_1 IS NOT NULL 
             AND cn0_lag_2 IS NOT NULL 
             AND cn0_lag_3 IS NOT NULL 
             AND cn0_lag_4 IS NOT NULL 
             AND cn0_lag_5 IS NOT NULL 
        THEN (
            STDDEV_POP(Cn0DbHz) OVER (
                PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz, CodeType
                ORDER BY sec_epoch
                ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
            )
        )
        ELSE NULL
    END AS sigma_5s,
    -- Recent delta
    Cn0DbHz - cn0_lag_5 AS delta_recent_5s
FROM signal_tracks;

-- =============================================================================
-- Step 8: Compute target (residual Δ C/N0 at 15s horizon)
-- =============================================================================

CREATE OR REPLACE TABLE features_raw AS
SELECT 
    -- Keys
    session,
    sec_epoch,
    device_name,
    recording_date,
    ConstellationType,
    Svid,
    CarrierFrequencyHz,
    CodeType,
    
    -- Temporal features
    Cn0DbHz AS current_cn0,
    cn0_centered,
    delta_recent_5s,
    sigma_5s,
    
    -- Physics features  
    ElevationDegrees,
    SIN(RADIANS(ElevationDegrees)) AS sin_elevation,
    AzimuthDegrees,
    
    -- Context features
    epoch_mean_cn0,
    epoch_std_cn0,
    epoch_n_signals,
    
    -- Target
    cn0_future_15s,
    cn0_future_15s - Cn0DbHz AS delta_cn0_15s,
    
    -- For later: orientation features (will be added separately)
    -- Requires join with OrientationDeg.parquet
    
    -- Quality flags
    CASE WHEN ABS(cn0_future_15s - Cn0DbHz) > 1.5 THEN 1 ELSE 0 END AS is_fading
FROM signals_with_jitter
WHERE 
    -- Require target to exist
    cn0_future_15s IS NOT NULL
    -- Require minimum track history
    AND track_row_num >= 5;

-- =============================================================================
-- Step 9: Add elevation band categorical feature
-- =============================================================================

CREATE OR REPLACE TABLE features_with_bands AS
SELECT 
    *,
    CASE 
        WHEN ElevationDegrees < 20 THEN 'low'
        WHEN ElevationDegrees < 45 THEN 'mid'
        ELSE 'high'
    END AS elevation_band
FROM features_raw;

-- =============================================================================
-- Step 10: Export summary statistics (for verification, not training)
-- =============================================================================

-- Count by device and recording date
SELECT 
    device_name,
    recording_date,
    COUNT(*) as n_samples,
    COUNT(DISTINCT session) as n_sessions,
    AVG(delta_cn0_15s) as mean_delta,
    STDDEV(delta_cn0_15s) as std_delta
FROM features_with_bands
GROUP BY device_name, recording_date
ORDER BY device_name, recording_date;

-- =============================================================================
-- Step 11: Sample for quick verification (1000 rows per device)
-- =============================================================================

-- Use COPY TO for export
-- COPY (
--     SELECT * FROM features_with_bands 
--     WHERE device_name = 'OnePlus CPH2401'
--     USING SAMPLE 1000 ROWS
-- ) TO '${FEATURES_DIR}/sample_oneplus_cph2401.csv';
