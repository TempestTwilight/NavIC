-- 045_pitch_roll_variation.sql
-- Pitch and roll stability within sessions and distribution across all sessions
SET memory_limit='1GB';
SET threads=2;

-- 1. Within-session pitch and roll std for the 14 orientation sessions
SELECT
  session,
  round(avg(pitchDeg), 1) AS pitch_mean,
  round(stddev(pitchDeg), 2) AS pitch_std,
  round(avg(rollDeg), 1) AS roll_mean,
  round(stddev(rollDeg), 2) AS roll_std,
  CASE
    WHEN abs(avg(pitchDeg)) < 5 AND abs(avg(rollDeg)) < 5 THEN 'FLAT_SCREEN_UP'
    WHEN abs(avg(pitchDeg)) < 5 AND abs(abs(avg(rollDeg)) - 180) < 10 THEN 'FLAT_SCREEN_DOWN'
    WHEN abs(avg(pitchDeg)) > 80 THEN 'VERTICAL_PORTRAIT'
    ELSE 'TILTED_INTERMEDIATE'
  END AS geometric_classification
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/OrientationDeg.parquet')
WHERE session NOT LIKE 'gnss_log_%'
GROUP BY session
ORDER BY session;

-- 2. Across all 89 sessions reporting OrientationDeg: range of mean attitudes
SELECT
  round(min(mean_pitch), 1) AS min_mean_pitch,
  round(max(mean_pitch), 1) AS max_mean_pitch,
  round(min(mean_roll), 1) AS min_mean_roll,
  round(max(mean_roll), 1) AS max_mean_roll
FROM (
  SELECT session, avg(pitchDeg) AS mean_pitch, avg(rollDeg) AS mean_roll
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/OrientationDeg.parquet')
  GROUP BY session
) sub;
