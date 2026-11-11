-- 021_raw_status_join_alignment.sql
-- Phase E: Cross-table time and signal alignment tests
-- SET memory_limit='1GB'; SET threads=2;

-- 1. Raw <-> Status join on sample session (gnss_log_2026_10_23_08_49_01)
-- Exact match on (utcTimeMillis, ConstellationType, Svid)
WITH r AS (
    SELECT DISTINCT session, utcTimeMillis, ConstellationType, Svid
    FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
s AS (
    SELECT DISTINCT session, UnixTimeMillis, ConstellationType, Svid
    FROM read_parquet('docs/EDA/scratch/typed/Status.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
exact AS (
    SELECT COUNT(*) as n_exact
    FROM r JOIN s
      ON r.utcTimeMillis = s.UnixTimeMillis
     AND r.ConstellationType = s.ConstellationType
     AND r.Svid = s.Svid
),
tol500 AS (
    SELECT COUNT(DISTINCT (r.utcTimeMillis, r.ConstellationType, r.Svid)) as n_500
    FROM r JOIN s
      ON ABS(r.utcTimeMillis - s.UnixTimeMillis) <= 500
     AND r.ConstellationType = s.ConstellationType
     AND r.Svid = s.Svid
)
SELECT
    (SELECT COUNT(*) FROM r) as total_raw_signals,
    (SELECT n_exact FROM exact) as exact_matched,
    ROUND((SELECT n_exact FROM exact) * 100.0 / (SELECT COUNT(*) FROM r), 2) as pct_exact_matched,
    (SELECT n_500 FROM tol500) as matched_500ms,
    ROUND((SELECT n_500 FROM tol500) * 100.0 / (SELECT COUNT(*) FROM r), 2) as pct_matched_500ms;

-- 2. Raw -> Fix: closest Fix within +/- 1000ms
WITH r_epochs AS (
    SELECT DISTINCT utcTimeMillis
    FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
f_epochs AS (
    SELECT DISTINCT UnixTimeMillis
    FROM read_parquet('docs/EDA/scratch/typed/Fix.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
closest_fix AS (
    SELECT r.utcTimeMillis,
           MIN(ABS(r.utcTimeMillis - f.UnixTimeMillis)) as min_diff_ms
    FROM r_epochs r
    CROSS JOIN f_epochs f
    GROUP BY r.utcTimeMillis
)
SELECT
    COUNT(*) as total_raw_epochs,
    SUM(CASE WHEN min_diff_ms <= 1000 THEN 1 ELSE 0 END) as matched_within_1s,
    ROUND(SUM(CASE WHEN min_diff_ms <= 1000 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_matched_1s
FROM closest_fix;

-- 3. Raw -> Agc: exact utcTimeMillis match rate on session with Agc present
WITH r_epochs AS (
    SELECT DISTINCT utcTimeMillis
    FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_19_08_51_20'
),
a_epochs AS (
    SELECT DISTINCT utcTimeMillis
    FROM read_parquet('docs/EDA/scratch/typed/Agc.parquet')
    WHERE session = 'gnss_log_2026_10_19_08_51_20'
)
SELECT
    (SELECT COUNT(*) FROM r_epochs) as total_raw_epochs,
    (SELECT COUNT(*) FROM a_epochs) as total_agc_epochs,
    COUNT(r.utcTimeMillis) as exact_matches,
    ROUND(COUNT(r.utcTimeMillis) * 100.0 / (SELECT COUNT(*) FROM r_epochs), 2) as pct_exact_match
FROM r_epochs r
JOIN a_epochs a ON r.utcTimeMillis = a.utcTimeMillis;

-- 4. Raw -> OrientationDeg: match rate with +/- 100ms tolerance
WITH r_epochs AS (
    SELECT DISTINCT utcTimeMillis as rt
    FROM read_parquet('docs/EDA/scratch/typed/Raw.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
o_epochs AS (
    SELECT DISTINCT utcTimeMillis as ot
    FROM read_parquet('docs/EDA/scratch/typed/OrientationDeg.parquet')
    WHERE session = 'gnss_log_2026_10_23_08_49_01'
),
matched AS (
    SELECT DISTINCT r.rt
    FROM r_epochs r
    JOIN o_epochs o ON ABS(r.rt - o.ot) <= 100
)
SELECT
    (SELECT COUNT(*) FROM r_epochs) as total_raw_epochs,
    COUNT(*) as matched_epochs_within_100ms,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM r_epochs), 2) as pct_matched
FROM matched;
