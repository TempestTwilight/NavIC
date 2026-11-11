-- 014_epoch_signal_count_distribution.sql
-- Phase D: Signal count per epoch percentiles in Raw and Status
-- Goal: Characterize satellite density per epoch, test if any epoch exceeds 100 signals.
SET memory_limit='1GB'; SET threads=1; SET preserve_insertion_order=false;

-- Raw epoch counts
WITH raw_ep AS (
  SELECT session, utcTimeMillis, COUNT(*) as cnt
  FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet'
  GROUP BY session, utcTimeMillis
)
SELECT 'Raw' as table_name,
  COUNT(*) as total_epochs,
  quantile_cont(cnt, 0.01) as p1,
  quantile_cont(cnt, 0.25) as p25,
  quantile_cont(cnt, 0.50) as p50,
  quantile_cont(cnt, 0.75) as p75,
  quantile_cont(cnt, 0.99) as p99,
  MAX(cnt) as max_signals,
  COUNT_IF(cnt > 100) as epochs_gt_100
FROM raw_ep;

-- Status epoch counts
WITH stat_ep AS (
  SELECT session, UnixTimeMillis, COUNT(*) as cnt
  FROM '/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Status.parquet'
  WHERE UnixTimeMillis IS NOT NULL
  GROUP BY session, UnixTimeMillis
)
SELECT 'Status' as table_name,
  COUNT(*) as total_epochs,
  quantile_cont(cnt, 0.01) as p1,
  quantile_cont(cnt, 0.25) as p25,
  quantile_cont(cnt, 0.50) as p50,
  quantile_cont(cnt, 0.75) as p75,
  quantile_cont(cnt, 0.99) as p99,
  MAX(cnt) as max_signals,
  COUNT_IF(cnt > 100) as epochs_gt_100
FROM stat_ep;
