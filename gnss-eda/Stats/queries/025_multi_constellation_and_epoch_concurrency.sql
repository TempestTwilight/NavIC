-- 025_multi_constellation_and_epoch_concurrency.sql
-- Phase E: Concurrency within epoch across constellations and Status vs Raw epoch divergence
-- SET memory_limit='1GB'; SET threads=2;

-- 1. Multi-constellation timestamp synchronicity per 1-second batch
WITH epoch_groups AS (
    SELECT
        session,
        ROUND(utcTimeMillis / 1000.0) * 1000 as epoch_sec,
        COUNT(DISTINCT utcTimeMillis) as distinct_utc_in_sec,
        COUNT(DISTINCT ConstellationType) as num_constellations,
        COUNT(*) as num_sats
    FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
    GROUP BY session, 2
)
SELECT
    COUNT(*) as total_sec_bins,
    SUM(CASE WHEN distinct_utc_in_sec = 1 THEN 1 ELSE 0 END) as single_utc_bins,
    SUM(CASE WHEN distinct_utc_in_sec > 1 THEN 1 ELSE 0 END) as multi_utc_bins,
    MAX(distinct_utc_in_sec) as max_distinct_utc_in_sec,
    MAX(num_constellations) as max_constellations_per_epoch,
    MAX(num_sats) as max_sats_per_epoch
FROM epoch_groups;

-- 2. Status vs Raw epoch boundary discrepancy across all 95 sessions
WITH r AS (
    SELECT session, COUNT(DISTINCT utcTimeMillis) as r_epochs
    FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
    GROUP BY session
),
s AS (
    SELECT session, COUNT(DISTINCT UnixTimeMillis) as s_epochs
    FROM read_parquet('docs/EDA/scratch/typed/Status.parquet')
    GROUP BY session
)
SELECT
    COUNT(*) as total_sessions,
    SUM(CASE WHEN r.r_epochs = s.s_epochs THEN 1 ELSE 0 END) as exact_epoch_count_match,
    SUM(CASE WHEN ABS(r.r_epochs - s.s_epochs) <= 5 THEN 1 ELSE 0 END) as within_5_epochs,
    ROUND(AVG(r.r_epochs - s.s_epochs), 2) as avg_diff_raw_minus_status,
    MIN(r.r_epochs - s.s_epochs) as min_diff,
    MAX(r.r_epochs - s.s_epochs) as max_diff
FROM r JOIN s ON r.session = s.session;
