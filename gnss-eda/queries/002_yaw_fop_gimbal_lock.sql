
-- Join OrientationDeg with FOP to quantify yaw vs heading error binned by |pitch|
WITH aligned AS (
    SELECT 
        o.session,
        s.device_name,
        o.pitchDeg,
        o.yawDeg,
        f.headingDegrees,
        -- Circular angular difference in [0, 180]
        180.0 - ABS(ABS(o.yawDeg - f.headingDegrees) - 180.0) as yaw_diff
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/OrientationDeg.parquet') o
    JOIN (
        SELECT session, manufacturer || ' ' || model as device_name
        FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
    ) s ON o.session = s.session
    ASOF JOIN read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/FOP.parquet') f
      ON o.session = f.session
     AND o.utcTimeMillis >= f.utcTimeMillis
    WHERE (o.utcTimeMillis - f.utcTimeMillis) <= 100
)
SELECT 
    device_name,
    FLOOR(ABS(pitchDeg) / 10.0) * 10.0 as pitch_bin,
    COUNT(*) as n_samples,
    ROUND(AVG(yaw_diff), 2) as mean_yaw_diff,
    ROUND(MEDIAN(yaw_diff), 2) as median_yaw_diff,
    ROUND(QUANTILE_CONT(yaw_diff, 0.75) - QUANTILE_CONT(yaw_diff, 0.25), 2) as iqr_yaw_diff,
    ROUND(QUANTILE_CONT(yaw_diff, 0.95), 2) as p95_yaw_diff
FROM aligned
GROUP BY device_name, pitch_bin
ORDER BY device_name, pitch_bin;
