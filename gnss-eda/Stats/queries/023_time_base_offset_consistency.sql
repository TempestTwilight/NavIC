-- 023_time_base_offset_consistency.sql
-- Phase E: Time base verification (utcTimeMillis vs elapsedRealtimeNanos boot offset)
-- SET memory_limit='1GB'; SET threads=2;

-- 1. FOP table boot-time offset consistency check
SELECT
    session,
    COUNT(*) as n_rows,
    MIN(utcTimeMillis - elapsedRealtimeNanos / 1e6) as min_offset_ms,
    MAX(utcTimeMillis - elapsedRealtimeNanos / 1e6) as max_offset_ms,
    AVG(utcTimeMillis - elapsedRealtimeNanos / 1e6) as avg_offset_ms,
    STDDEV(utcTimeMillis - elapsedRealtimeNanos / 1e6) as std_offset_ms
FROM read_parquet('docs/EDA/scratch/typed/FOP.parquet')
WHERE session = 'gnss_log_2026_10_23_08_49_01'
  AND utcTimeMillis IS NOT NULL
  AND elapsedRealtimeNanos IS NOT NULL
GROUP BY session;

-- 2. Accel table boot-time offset consistency check
SELECT
    session,
    COUNT(*) as n_rows,
    MIN(utcTimeMillis - elapsedRealtimeNanos / 1e6) as min_offset_ms,
    MAX(utcTimeMillis - elapsedRealtimeNanos / 1e6) as max_offset_ms,
    AVG(utcTimeMillis - elapsedRealtimeNanos / 1e6) as avg_offset_ms,
    STDDEV(utcTimeMillis - elapsedRealtimeNanos / 1e6) as std_offset_ms
FROM read_parquet('docs/EDA/scratch/typed/Accel.parquet')
WHERE session = 'gnss_log_2026_10_19_08_51_20'
  AND utcTimeMillis IS NOT NULL
  AND elapsedRealtimeNanos IS NOT NULL
GROUP BY session;

-- 3. Raw vs Agc time range consistency (verifying Agc and Raw share identical time base)
SELECT
    'Agc' as table_name,
    MIN(utcTimeMillis) as t_min,
    MAX(utcTimeMillis) as t_max,
    COUNT(*) as row_count
FROM read_parquet('docs/EDA/scratch/typed/Agc.parquet')
WHERE session = 'gnss_log_2026_10_19_08_51_20'
UNION ALL
SELECT
    'Raw' as table_name,
    MIN(utcTimeMillis) as t_min,
    MAX(utcTimeMillis) as t_max,
    COUNT(*) as row_count
FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
WHERE session = 'gnss_log_2026_10_19_08_51_20';
