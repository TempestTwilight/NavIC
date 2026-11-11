
-- Compute median and IQR of Cn0DbHz binned by ElevationDegrees across devices
WITH target_sessions AS (
    SELECT session, manufacturer || ' ' || model as device_name
    FROM (
        SELECT session, manufacturer, model,
               ROW_NUMBER() OVER (PARTITION BY model ORDER BY session) as rn
        FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
    ) WHERE rn <= 4
)
SELECT 
    s.device_name,
    FLOOR(st.ElevationDegrees / 5.0) * 5.0 as elev_bin,
    COUNT(*) as n_rows,
    ROUND(AVG(r.Cn0DbHz), 2) as mean_cn0,
    ROUND(MEDIAN(r.Cn0DbHz), 2) as median_cn0,
    ROUND(QUANTILE_CONT(r.Cn0DbHz, 0.25), 2) as p25_cn0,
    ROUND(QUANTILE_CONT(r.Cn0DbHz, 0.75), 2) as p75_cn0
FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
JOIN target_sessions s ON r.session = s.session
JOIN (
    SELECT DISTINCT session, CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
           ConstellationType, Svid, CarrierFrequencyHz, ElevationDegrees
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
) st ON r.session = st.session
   AND CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
   AND r.ConstellationType = st.ConstellationType
   AND r.CarrierFrequencyHz = st.CarrierFrequencyHz
WHERE r.Cn0DbHz > 0 AND st.ElevationDegrees >= 0
GROUP BY s.device_name, elev_bin
ORDER BY s.device_name, elev_bin;
