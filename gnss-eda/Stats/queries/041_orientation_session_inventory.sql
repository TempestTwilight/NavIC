-- 041_orientation_session_inventory.sql
-- Inventory of all 95 sessions: duration, timestamps, row counts across sensor tables
SET memory_limit='1GB';
SET threads=2;

WITH meta AS (
  SELECT session, platform, manufacturer, model, gnss_hw
  FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/sessions_meta.csv')
),
ori AS (
  SELECT session, count(*) AS ori_rows,
    min(utcTimeMillis) AS tmin, max(utcTimeMillis) AS tmax,
    round((max(utcTimeMillis) - min(utcTimeMillis)) / 60000.0, 2) AS dur_min
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/OrientationDeg.parquet')
  GROUP BY session
),
grv AS (
  SELECT session, count(*) AS grv_rows
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/GameRotationVector.parquet')
  GROUP BY session
),
fop AS (
  SELECT session, count(*) AS fop_rows
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/FOP.parquet')
  GROUP BY session
),
acc AS (
  SELECT session, count(*) AS accel_rows
  FROM read_parquet('/home/tempest/Documents/ISRO_Project/docs/EDA/scratch/typed/Accel.parquet')
  GROUP BY session
)
SELECT m.session, m.manufacturer || ' ' || m.model AS device,
  ori.dur_min,
  coalesce(ori.ori_rows, 0) AS ori_rows,
  coalesce(grv.grv_rows, 0) AS grv_rows,
  coalesce(fop.fop_rows, 0) AS fop_rows,
  coalesce(acc.accel_rows, 0) AS accel_rows
FROM meta m
LEFT JOIN ori USING(session)
LEFT JOIN grv USING(session)
LEFT JOIN fop USING(session)
LEFT JOIN acc USING(session)
ORDER BY m.session;
