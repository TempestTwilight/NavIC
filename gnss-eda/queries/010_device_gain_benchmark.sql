
-- Benchmark high-elevation (>60 deg) GPS L1 C/N0 across devices and sessions
SELECT 
    s.device_name,
    r.session,
    COUNT(*) as n_signals,
    ROUND(AVG(r.Cn0DbHz), 2) as mean_cn0,
    ROUND(MEDIAN(r.Cn0DbHz), 2) as median_cn0,
    ROUND(STDDEV(r.Cn0DbHz), 2) as std_cn0
FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
JOIN (
    SELECT session, manufacturer || ' ' || model as device_name
    FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
) s ON r.session = s.session
JOIN (
    SELECT DISTINCT session, CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
           ConstellationType, Svid, CarrierFrequencyHz, ElevationDegrees
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
) st ON r.session = st.session
   AND CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
   AND r.ConstellationType = st.ConstellationType
   AND r.CarrierFrequencyHz = st.CarrierFrequencyHz
WHERE r.Cn0DbHz > 0 
  AND r.ConstellationType = 1 -- GPS
  AND r.CarrierFrequencyHz BETWEEN 1.57e9 AND 1.58e9 -- L1
  AND st.ElevationDegrees >= 60.0
GROUP BY s.device_name, r.session
HAVING n_signals >= 50
ORDER BY s.device_name, median_cn0 DESC;
