-- 037_agc_behavior.sql
-- Analyzes Automatic Gain Control (AgcDb) behavior across Agc.parquet and Raw.parquet.
-- Examines per-device variance, constancy, and correlation with C/N0.

SET memory_limit='1GB';
SET threads=2;
SET preserve_insertion_order=false;

-- 1. Agc.parquet distribution
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT session) AS n_sessions,
  MIN(AgcDb) AS min_agc,
  ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY AgcDb), 2) AS p25_agc,
  ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY AgcDb), 2) AS p50_agc,
  ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY AgcDb), 2) AS p75_agc,
  MAX(AgcDb) AS max_agc,
  ROUND(AVG(AgcDb), 2) AS mean_agc,
  ROUND(STDDEV(AgcDb), 4) AS std_agc
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Agc.parquet');

-- 2. Agc.parquet per-session constancy check (top sessions)
SELECT
  session,
  COUNT(*) AS n,
  COUNT(DISTINCT AgcDb) AS distinct_agc,
  MIN(AgcDb) AS min_agc,
  MAX(AgcDb) AS max_agc,
  ROUND(STDDEV(AgcDb), 4) AS std_agc,
  ROUND(AVG(AgcDb), 2) AS mean_agc
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Agc.parquet')
GROUP BY session
ORDER BY n DESC
LIMIT 10;

-- 3. Raw inline AgcDb distribution across all devices
SELECT
  COUNT(*) AS total_rows,
  COUNT(AgcDb) AS non_null_rows,
  MIN(AgcDb) AS min_agc,
  ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY AgcDb), 2) AS p25_agc,
  ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY AgcDb), 2) AS p50_agc,
  ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY AgcDb), 2) AS p75_agc,
  MAX(AgcDb) AS max_agc,
  ROUND(AVG(AgcDb), 2) AS mean_agc,
  ROUND(STDDEV(AgcDb), 2) AS std_agc
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet');
