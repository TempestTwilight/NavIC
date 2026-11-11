-- 042_named_orientation_angles.sql
-- Statistical profile of yawDeg, pitchDeg, rollDeg for the 14 orientation-named sessions
SET memory_limit='1GB';
SET threads=2;

SELECT
  session,
  round(avg(yawDeg), 1) AS yaw_mean,
  round(stddev(yawDeg), 1) AS yaw_std,
  min(yawDeg) AS yaw_min,
  max(yawDeg) AS yaw_max,
  round(avg(pitchDeg), 1) AS pitch_mean,
  round(stddev(pitchDeg), 1) AS pitch_std,
  min(pitchDeg) AS pitch_min,
  max(pitchDeg) AS pitch_max,
  round(avg(rollDeg), 1) AS roll_mean,
  round(stddev(rollDeg), 1) AS roll_std,
  min(rollDeg) AS roll_min,
  max(rollDeg) AS roll_max
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/OrientationDeg.parquet')
WHERE session NOT LIKE 'gnss_log_%'
GROUP BY session
ORDER BY session;
