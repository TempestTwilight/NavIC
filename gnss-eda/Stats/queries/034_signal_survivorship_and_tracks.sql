-- 034_signal_survivorship_and_tracks.sql
-- Analyzes signal survivorship at +1s, +5s, +15s horizons,
-- continuous track length distribution, and signal gap / reappearance behavior.

SET memory_limit='1GB';
SET threads=2;
SET preserve_insertion_order=false;

-- 1. Continuous track length distribution across all sessions
WITH tracks AS (
  SELECT
    session, ConstellationType, Svid, CarrierFrequencyHz,
    COUNT(DISTINCT utcTimeMillis) AS track_len
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
  WHERE Cn0DbHz > 0
  GROUP BY session, ConstellationType, Svid, CarrierFrequencyHz
)
SELECT
  PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY track_len) AS p1,
  PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY track_len) AS p5,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY track_len) AS p25,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY track_len) AS p50,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY track_len) AS p75,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY track_len) AS p95,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY track_len) AS p99,
  MAX(track_len) AS max_len,
  AVG(track_len) AS mean_len,
  COUNT(*) AS total_tracks,
  SUM(CASE WHEN track_len = 1 THEN 1 ELSE 0 END) AS tracks_len_1,
  SUM(CASE WHEN track_len >= 15 THEN 1 ELSE 0 END) AS tracks_ge_15s
FROM tracks;

-- 2. Signal gap and reappearance transitions (sample session: gnss_log_2026_10_23_08_49_01)
WITH epochs AS (
  SELECT
    session, ConstellationType, Svid, CarrierFrequencyHz,
    utcTimeMillis,
    ROW_NUMBER() OVER (
      PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz
      ORDER BY utcTimeMillis
    ) AS rn,
    LAG(utcTimeMillis) OVER (
      PARTITION BY session, ConstellationType, Svid, CarrierFrequencyHz
      ORDER BY utcTimeMillis
    ) AS prev_ts
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Raw.parquet')
  WHERE session = 'gnss_log_2026_10_23_08_49_01' AND Cn0DbHz > 0
),
gaps AS (
  SELECT *,
    (utcTimeMillis - prev_ts) AS gap_ms
  FROM epochs WHERE rn > 1
)
SELECT
  COUNT(*) AS total_transitions,
  SUM(CASE WHEN gap_ms <= 1500 THEN 1 ELSE 0 END) AS continuous_epochs,
  SUM(CASE WHEN gap_ms > 1500 AND gap_ms <= 5000 THEN 1 ELSE 0 END) AS short_gap_under_5s,
  SUM(CASE WHEN gap_ms > 5000 THEN 1 ELSE 0 END) AS long_gap_over_5s,
  ROUND(AVG(CASE WHEN gap_ms > 1500 THEN gap_ms END) / 1000.0, 2) AS mean_gap_sec,
  ROUND(MAX(gap_ms) / 1000.0, 2) AS max_gap_sec
FROM gaps;
