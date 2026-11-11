-- 024_asof_orientation_sensor_joins.sql
-- Phase E: As-Of Join validation for multi-rate sensor fusion (Raw -> Orientation/IMU)
-- SET memory_limit='1GB'; SET threads=2;

-- 1. As-Of Join: Raw (1Hz) -> FOP (~100Hz) with 200ms causal window
WITH r_epochs AS (
    SELECT DISTINCT session, utcTimeMillis
    FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
fop_epochs AS (
    SELECT session, utcTimeMillis, x, y, z, w, headingDegrees
    FROM read_parquet('docs/EDA/scratch/typed/FOP.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
joined_fop AS (
    SELECT
        r.utcTimeMillis as r_time,
        f.utcTimeMillis as f_time,
        r.utcTimeMillis - f.utcTimeMillis as offset_ms
    FROM r_epochs r
    ASOF JOIN fop_epochs f
      ON r.session = f.session
     AND r.utcTimeMillis >= f.utcTimeMillis
)
SELECT
    'Raw -> FOP (200ms tol)' as join_type,
    COUNT(*) as total_raw_epochs,
    SUM(CASE WHEN offset_ms <= 200 THEN 1 ELSE 0 END) as matched_within_tol,
    ROUND(SUM(CASE WHEN offset_ms <= 200 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_matched,
    MEDIAN(CASE WHEN offset_ms <= 200 THEN offset_ms END) as median_offset_ms,
    MAX(CASE WHEN offset_ms <= 200 THEN offset_ms END) as max_offset_ms
FROM joined_fop;

-- 2. As-Of Join: Raw (1Hz) -> OrientationDeg (~200Hz) with 200ms causal window
WITH r_epochs AS (
    SELECT DISTINCT session, utcTimeMillis
    FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
ori_epochs AS (
    SELECT session, utcTimeMillis, yawDeg, rollDeg, pitchDeg
    FROM read_parquet('docs/EDA/scratch/typed/OrientationDeg.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
joined_ori AS (
    SELECT
        r.utcTimeMillis as r_time,
        o.utcTimeMillis as o_time,
        r.utcTimeMillis - o.utcTimeMillis as offset_ms
    FROM r_epochs r
    ASOF JOIN ori_epochs o
      ON r.session = o.session
     AND r.utcTimeMillis >= o.utcTimeMillis
)
SELECT
    'Raw -> OrientationDeg (200ms tol)' as join_type,
    COUNT(*) as total_raw_epochs,
    SUM(CASE WHEN offset_ms <= 200 THEN 1 ELSE 0 END) as matched_within_tol,
    ROUND(SUM(CASE WHEN offset_ms <= 200 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_matched,
    MEDIAN(CASE WHEN offset_ms <= 200 THEN offset_ms END) as median_offset_ms,
    MAX(CASE WHEN offset_ms <= 200 THEN offset_ms END) as max_offset_ms
FROM joined_ori;

-- 3. As-Of Join: Raw (1Hz) -> Accel (~250Hz) with 100ms causal window
WITH r_epochs AS (
    SELECT DISTINCT session, utcTimeMillis
    FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_19_08_51_20'
),
acc_epochs AS (
    SELECT session, utcTimeMillis, AccelXMps2, AccelYMps2, AccelZMps2
    FROM read_parquet('docs/EDA/scratch/typed/Accel.parquet')
    WHERE session = 'gnss_log_2026_10_19_08_51_20'
),
joined_acc AS (
    SELECT
        r.utcTimeMillis as r_time,
        a.utcTimeMillis as a_time,
        r.utcTimeMillis - a.utcTimeMillis as offset_ms
    FROM r_epochs r
    ASOF JOIN acc_epochs a
      ON r.session = a.session
     AND r.utcTimeMillis >= a.utcTimeMillis
)
SELECT
    'Raw -> Accel (100ms tol)' as join_type,
    COUNT(*) as total_raw_epochs,
    SUM(CASE WHEN offset_ms <= 100 THEN 1 ELSE 0 END) as matched_within_tol,
    ROUND(SUM(CASE WHEN offset_ms <= 100 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_matched,
    MEDIAN(CASE WHEN offset_ms <= 100 THEN offset_ms END) as median_offset_ms,
    MAX(CASE WHEN offset_ms <= 100 THEN offset_ms END) as max_offset_ms
FROM joined_acc;
