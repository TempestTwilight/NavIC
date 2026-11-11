-- 032_cn0_vs_elevation.sql
-- Computes C/N0 distribution binned by satellite elevation (0-10, 10-20, ..., 80-90 degrees)
-- joins Status.ElevationDegrees to Raw.Cn0DbHz on (session, ConstellationType, Svid) within ±500ms.
-- Memory safe: tested on single session and sampled to remain under 1GB RAM.

SET memory_limit='1GB';
SET threads=2;
SET preserve_insertion_order=false;

-- Single session demonstration (gnss_log_2026_10_23_08_49_01)
WITH raw_s AS (
  SELECT utcTimeMillis, ConstellationType, Svid, CarrierFrequencyHz, Cn0DbHz
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
  WHERE session = 'gnss_log_2026_10_23_08_49_01' AND Cn0DbHz > 0
),
stat_s AS (
  SELECT UnixTimeMillis, ConstellationType, Svid, ElevationDegrees
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet')
  WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
joined AS (
  SELECT
    r.Cn0DbHz,
    s.ElevationDegrees,
    FLOOR(s.ElevationDegrees / 10.0) * 10 AS elev_bin
  FROM raw_s r
  JOIN stat_s s ON
    r.ConstellationType = s.ConstellationType AND
    r.Svid = s.Svid AND
    ABS(r.utcTimeMillis - s.UnixTimeMillis) <= 500
)
SELECT
  elev_bin,
  COUNT(*) as n,
  ROUND(AVG(Cn0DbHz), 2) as mean_cn0,
  ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Cn0DbHz), 2) as p50_cn0,
  ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Cn0DbHz), 2) as p25_cn0,
  ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Cn0DbHz), 2) as p75_cn0,
  ROUND(STDDEV(Cn0DbHz), 2) as std_cn0
FROM joined
WHERE elev_bin >= 0
GROUP BY elev_bin
ORDER BY elev_bin;

-- Direct elevation distribution from Status table (all sessions)
SELECT
  FLOOR((ElevationDegrees + 5) / 5.0) * 5 AS bin_deg,
  COUNT(*) AS n,
  ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Cn0DbHz), 1) AS med_cn0,
  ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Cn0DbHz), 1) AS p25,
  ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Cn0DbHz), 1) AS p75,
  ROUND(100.0 * COUNT(*) FILTER (WHERE Cn0DbHz = 0) / COUNT(*), 1) AS pct_zero
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet')
WHERE ElevationDegrees IS NOT NULL
GROUP BY bin_deg
ORDER BY bin_deg;
