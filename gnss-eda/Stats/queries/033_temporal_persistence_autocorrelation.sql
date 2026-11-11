-- 033_temporal_persistence_autocorrelation.sql
-- Evaluates temporal persistence of Cn0DbHz across horizons h = 1s, 5s, 15s.
-- This defines the primary baseline for the 15-second ahead C/N0 prediction task.
-- Uses window LAG partitioned by (session, ConstellationType, Svid, CarrierFrequencyHz)
-- with time matching tolerance ±500ms.

SET memory_limit='1GB';
SET threads=2;
SET preserve_insertion_order=false;

-- Horizon h = 15s (Key ML task horizon)
WITH pairs AS (
  SELECT
    session,
    utcTimeMillis AS t,
    ConstellationType AS ct,
    Svid,
    CarrierFrequencyHz AS freq,
    Cn0DbHz AS cn0
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
  WHERE Cn0DbHz > 0
),
lagged AS (
  SELECT
    ct, svid, freq, session, t, cn0,
    lag(t, 15) OVER w AS t_lag,
    lag(cn0, 15) OVER w AS cn0_lag
  FROM pairs
  WINDOW w AS (PARTITION BY session, ct, svid, freq ORDER BY t)
),
deltas AS (
  SELECT
    ABS(cn0 - cn0_lag) AS abs_delta,
    (cn0 - cn0_lag) AS raw_delta
  FROM lagged
  WHERE ABS(t - t_lag - 15000) <= 500 AND cn0_lag IS NOT NULL
)
SELECT
  COUNT(*) AS n_pairs,
  AVG(abs_delta) AS mean_abs_error,
  STDDEV(abs_delta) AS std_abs_error,
  PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY abs_delta) AS p1,
  PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY abs_delta) AS p5,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY abs_delta) AS p25,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY abs_delta) AS p50_median,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY abs_delta) AS p75,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY abs_delta) AS p95,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY abs_delta) AS p99,
  MAX(abs_delta) AS max_abs_delta,
  AVG(CASE WHEN abs_delta > 1.5 THEN 1.0 ELSE 0.0 END) AS frac_gt_1p5_db,
  AVG(CASE WHEN abs_delta > 3.0 THEN 1.0 ELSE 0.0 END) AS frac_gt_3p0_db
FROM deltas;
