-- 022_table_sampling_rates.sql
-- Phase E: Sampling intervals and effective frequencies across all tables
-- SET memory_limit='1GB'; SET threads=2;

-- Compute gap statistics across representative sessions for all tables
-- Table order: GNSS tables (1Hz), Fix (multi-rate), IMU/Orientation sensors (50-250Hz)

WITH raw_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, utcTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet') LIMIT 50000)
),
status_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, UnixTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/Status.parquet') LIMIT 50000)
),
fix_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, UnixTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/Fix.parquet') LIMIT 50000)
),
agc_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, utcTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/Agc.parquet') LIMIT 50000)
),
fop_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, utcTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/FOP.parquet') LIMIT 50000)
),
ori_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, utcTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/OrientationDeg.parquet') LIMIT 50000)
),
grv_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, utcTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/GameRotationVector.parquet') LIMIT 50000)
),
acc_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, utcTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/Accel.parquet') LIMIT 50000)
),
uncal_acc_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, utcTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/UncalAccel.parquet') LIMIT 50000)
),
uncal_gyro_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, utcTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/UncalGyro.parquet') LIMIT 50000)
),
uncal_mag_gaps AS (
    SELECT t - LAG(t) OVER (PARTITION BY session ORDER BY t) as gap_ms
    FROM (SELECT DISTINCT session, utcTimeMillis as t FROM read_parquet('docs/EDA/scratch/typed/UncalMag.parquet') LIMIT 50000)
)
SELECT 'Raw' as table_name,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms) as median_gap_ms,
       PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms) as p5_gap_ms,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms) as p95_gap_ms,
       ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2) as approx_rate_hz
FROM raw_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'Status', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM status_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'Fix', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM fix_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'Agc', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM agc_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'OrientationDeg', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM ori_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'GameRotationVector', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM grv_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'FOP', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM fop_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'Accel', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM acc_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'UncalAccel', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM uncal_acc_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'UncalGyro', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM uncal_gyro_gaps WHERE gap_ms > 0
UNION ALL
SELECT 'UncalMag', PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY gap_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY gap_ms), ROUND(1000.0 / NULLIF(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_ms), 0), 2)
FROM uncal_mag_gaps WHERE gap_ms > 0;
