-- 043_angle_conventions_and_conversions.sql
-- Angle conventions, quaternion normalization, and cross-sensor comparisons
SET memory_limit='1GB';
SET threads=2;

-- 1. OrientationDeg global angle bounds
SELECT
  'OrientationDeg' AS sensor,
  min(yawDeg) AS yaw_min, max(yawDeg) AS yaw_max, count(DISTINCT yawDeg) AS yaw_distinct,
  min(pitchDeg) AS pitch_min, max(pitchDeg) AS pitch_max,
  min(rollDeg) AS roll_min, max(rollDeg) AS roll_max
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/OrientationDeg.parquet')

UNION ALL

-- 2. FOP global angle bounds
SELECT
  'FOP' AS sensor,
  min(headingDegrees) AS yaw_min, max(headingDegrees) AS yaw_max, count(DISTINCT headingDegrees) AS yaw_distinct,
  min(headingErrorDegrees) AS pitch_min, max(headingErrorDegrees) AS pitch_max,
  NULL AS roll_min, NULL AS roll_max
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/FOP.parquet');

-- 3. FOP heading vs OrientationDeg yaw alignment across matched epochs (sample)
WITH j AS (
  SELECT f.headingDegrees AS h, o.yawDeg
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/FOP.parquet') f
  JOIN read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/OrientationDeg.parquet') o
    USING(session, elapsedRealtimeNanos)
  WHERE random() < 0.02
),
d AS (
  SELECT *,
    least(abs(((yawDeg - h) % 360 + 360) % 360), abs(((h - yawDeg) % 360 + 360) % 360)) AS circ_diff
  FROM j
)
SELECT
  count(*) AS n_matched,
  round(approx_quantile(circ_diff, 0.5), 2) AS med_absdiff_deg,
  round(avg(CASE WHEN circ_diff < 5 THEN 1.0 ELSE 0.0 END), 3) AS frac_lt5deg,
  round(avg(CASE WHEN circ_diff < 10 THEN 1.0 ELSE 0.0 END), 3) AS frac_lt10deg
FROM d;
