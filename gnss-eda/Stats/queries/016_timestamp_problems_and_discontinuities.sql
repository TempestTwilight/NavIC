-- 016_timestamp_problems_and_discontinuities.sql
-- Phase D: Timestamp problems, clock discontinuities, and cross-clock drift
-- Goal: Test TimeNanos monotonicity, table sampling gaps > 2s, hardware clock counter shifts, and elapsedRealtime offset stability.
SET memory_limit='1GB'; SET threads=1; SET preserve_insertion_order=false;

-- 1. Non-monotonic TimeNanos and utcTimeMillis within sessions in Raw
WITH raw_epochs AS (
  SELECT session, utcTimeMillis, TimeNanos
  FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet'
  GROUP BY session, utcTimeMillis, TimeNanos
),
raw_diffs AS (
  SELECT session, utcTimeMillis, TimeNanos,
    TimeNanos - LAG(TimeNanos) OVER (PARTITION BY session ORDER BY utcTimeMillis) as dt_nanos,
    utcTimeMillis - LAG(utcTimeMillis) OVER (PARTITION BY session ORDER BY utcTimeMillis) as dt_utc
  FROM raw_epochs
)
SELECT 'Raw_monotonicity' as check_name,
  COUNT(*) as total_transitions,
  COUNT_IF(dt_nanos < 0) as non_monotonic_nanos,
  COUNT_IF(dt_utc < 0) as non_monotonic_utc,
  COUNT_IF(dt_utc > 2000) as gaps_gt_2s,
  MAX(dt_utc) as max_gap_ms,
  MIN(dt_nanos) as min_dt_nanos
FROM raw_diffs
WHERE dt_nanos IS NOT NULL;

-- 2. HardwareClockDiscontinuityCount within sessions
SELECT session,
  MIN(HardwareClockDiscontinuityCount) as min_disc,
  MAX(HardwareClockDiscontinuityCount) as max_disc,
  COUNT(DISTINCT HardwareClockDiscontinuityCount) as num_distinct_disc
FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet'
GROUP BY session
HAVING COUNT(DISTINCT HardwareClockDiscontinuityCount) > 1;

-- 3. Cross-clock offset stability: utcTimeMillis vs elapsedRealtimeNanos in Accel
WITH sample_accel AS (
  SELECT session, utcTimeMillis, elapsedRealtimeNanos,
    (utcTimeMillis * 1000000 - elapsedRealtimeNanos) as boot_offset_nanos
  FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Accel.parquet'
  WHERE utcTimeMillis IS NOT NULL AND elapsedRealtimeNanos IS NOT NULL
  LIMIT 50000
)
SELECT 'Clock_offset_stability' as check_name,
  MIN(boot_offset_nanos) as min_offset,
  MAX(boot_offset_nanos) as max_offset,
  stddev(boot_offset_nanos) as stddev_offset
FROM sample_accel;
