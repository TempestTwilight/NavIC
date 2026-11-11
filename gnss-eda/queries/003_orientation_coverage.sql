
-- 2D histogram of yaw vs pitch across devices
WITH binned AS (
    SELECT 
        s.device_name,
        FLOOR(o.yawDeg / 10.0) * 10.0 as yaw_bin,
        FLOOR(o.pitchDeg / 10.0) * 10.0 as pitch_bin,
        COUNT(*) as count
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/OrientationDeg.parquet') o
    JOIN (
        SELECT session, manufacturer || ' ' || model as device_name
        FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
    ) s ON o.session = s.session
    GROUP BY device_name, yaw_bin, pitch_bin
)
SELECT * FROM binned
ORDER BY device_name, yaw_bin, pitch_bin;
