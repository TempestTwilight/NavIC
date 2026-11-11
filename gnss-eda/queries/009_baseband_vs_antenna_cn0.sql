
-- 2D histogram of Cn0DbHz vs BasebandCn0DbHz and difference statistics
WITH pairs AS (
    SELECT 
        s.device_name,
        FLOOR(Cn0DbHz) as cn0_bin,
        FLOOR(BasebandCn0DbHz) as bb_bin,
        Cn0DbHz - BasebandCn0DbHz as diff_cn0
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN (
        SELECT session, manufacturer || ' ' || model as device_name
        FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
    ) s ON r.session = s.session
    WHERE Cn0DbHz > 0 AND BasebandCn0DbHz > 0
)
SELECT 
    device_name,
    cn0_bin,
    bb_bin,
    COUNT(*) as count,
    ROUND(AVG(diff_cn0), 2) as mean_diff
FROM pairs
GROUP BY device_name, cn0_bin, bb_bin
HAVING count >= 5
ORDER BY device_name, cn0_bin, bb_bin;
