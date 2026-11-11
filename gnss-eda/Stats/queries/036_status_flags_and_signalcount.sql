-- 036_status_flags_and_signalcount.sql
-- Checks UsedInFix, HasAlmanacData, HasEphemerisData from Status.parquet.
-- Evaluates SignalCount consistency within epochs and against actual Status row count.
-- Documents ML leakage considerations.

SET memory_limit='1GB';
SET threads=2;
SET preserve_insertion_order=false;

-- 1. Status flags overall overview
SELECT
  COUNT(*) AS total_rows,
  ROUND(AVG(CAST(UsedInFix AS INTEGER)), 4) AS frac_used_in_fix,
  ROUND(AVG(CAST(HasAlmanacData AS INTEGER)), 4) AS frac_has_almanac,
  ROUND(AVG(CAST(HasEphemerisData AS INTEGER)), 4) AS frac_has_ephemeris,
  MIN(SignalCount) AS min_signal_count,
  MAX(SignalCount) AS max_signal_count,
  ROUND(AVG(SignalCount), 1) AS mean_signal_count
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet');

-- 2. UsedInFix by ConstellationType
SELECT
  ConstellationType,
  COUNT(*) AS n,
  ROUND(AVG(CAST(UsedInFix AS INTEGER)), 4) AS frac_used_in_fix,
  ROUND(AVG(CAST(HasAlmanacData AS INTEGER)), 4) AS frac_has_almanac,
  ROUND(AVG(CAST(HasEphemerisData AS INTEGER)), 4) AS frac_has_ephemeris
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet')
GROUP BY ConstellationType
ORDER BY ConstellationType;

-- 3. SignalCount consistency within epoch and actual row count comparison
WITH epoch_sc AS (
  SELECT
    session, UnixTimeMillis,
    COUNT(*) AS actual_count,
    MIN(SignalCount) AS min_sc,
    MAX(SignalCount) AS max_sc,
    AVG(SignalCount) AS mean_sc
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet')
  GROUP BY session, UnixTimeMillis
)
SELECT
  COUNT(*) AS n_epochs,
  ROUND(AVG(CASE WHEN min_sc = max_sc THEN 1.0 ELSE 0.0 END), 4) AS frac_sc_constant_within_epoch,
  ROUND(AVG(ABS(mean_sc - actual_count)), 3) AS mean_sc_vs_actual_diff,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY actual_count) AS p50_actual,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY mean_sc) AS p50_sc
FROM epoch_sc;
