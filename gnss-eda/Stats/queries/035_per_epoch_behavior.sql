-- 035_per_epoch_behavior.sql
-- Computes per-epoch signal count, epoch-mean C/N0 distribution,
-- and decomposes within-epoch variance vs between-epoch variance.

SET memory_limit='1GB';
SET threads=2;
SET preserve_insertion_order=false;

WITH epoch_stats AS (
  SELECT
    session,
    utcTimeMillis,
    COUNT(*) AS n_signals,
    AVG(Cn0DbHz) AS epoch_mean_cn0,
    STDDEV(Cn0DbHz) AS epoch_std_cn0
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
  WHERE Cn0DbHz > 0
  GROUP BY session, utcTimeMillis
)
SELECT
  COUNT(*) AS total_epochs,
  -- Number of signals per epoch quantiles
  PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY n_signals) AS p5_nsig,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY n_signals) AS p25_nsig,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY n_signals) AS p50_nsig,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY n_signals) AS p75_nsig,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY n_signals) AS p95_nsig,
  MAX(n_signals) AS max_nsig,
  ROUND(AVG(n_signals), 2) AS mean_nsig,
  -- Epoch mean C/N0 quantiles
  ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY epoch_mean_cn0), 2) AS p5_emean,
  ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY epoch_mean_cn0), 2) AS p25_emean,
  ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY epoch_mean_cn0), 2) AS p50_emean,
  ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY epoch_mean_cn0), 2) AS p75_emean,
  ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY epoch_mean_cn0), 2) AS p95_emean,
  -- Within-epoch vs between-epoch variance decomposition
  ROUND(AVG(epoch_std_cn0), 2) AS mean_within_epoch_std,
  ROUND(STDDEV(epoch_mean_cn0), 2) AS between_epoch_std
FROM epoch_stats;
