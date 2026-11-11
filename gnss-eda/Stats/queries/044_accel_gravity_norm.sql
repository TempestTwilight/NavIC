-- 044_accel_gravity_norm.sql
-- Gravity norm check per session across calibrated Accel measurements
SET memory_limit='1GB';
SET threads=2;

SELECT
  session,
  round(avg(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) AS mean_norm,
  round(stddev(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 4) AS std_norm,
  round(min(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) AS min_norm,
  round(max(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) AS max_norm,
  count(*) AS n_rows,
  CASE
    WHEN round(avg(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) < 8.0 OR
         round(avg(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)), 3) > 11.0
    THEN 'FLAGGED_OUT_OF_BOUNDS'
    ELSE 'NOMINAL_GRAVITY'
  END AS gravity_status,
  CASE
    WHEN stddev(sqrt(AccelXMps2*AccelXMps2 + AccelYMps2*AccelYMps2 + AccelZMps2*AccelZMps2)) > 0.5
    THEN 'MOVING'
    ELSE 'STATIC'
  END AS motion_class
FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Accel.parquet')
GROUP BY session
ORDER BY session;
