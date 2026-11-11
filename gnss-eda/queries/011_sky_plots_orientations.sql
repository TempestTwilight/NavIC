
-- Extract satellite azimuth, elevation, and median C/N0 for 4 distinct orientation poses
SELECT 
    r.session,
    FLOOR(st.AzimuthDegrees / 15.0) * 15.0 as az_bin,
    FLOOR(st.ElevationDegrees / 10.0) * 10.0 as el_bin,
    COUNT(*) as n_obs,
    ROUND(MEDIAN(r.Cn0DbHz), 2) as median_cn0
FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
JOIN (
    SELECT DISTINCT session, CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
           ConstellationType, Svid, CarrierFrequencyHz, AzimuthDegrees, ElevationDegrees
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
) st ON r.session = st.session
   AND CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
   AND r.ConstellationType = st.ConstellationType
   AND r.CarrierFrequencyHz = st.CarrierFrequencyHz
WHERE r.session IN ('0_up_facing', '0_down_facing', '90_front_vertical', '90_back_vertical')
  AND r.Cn0DbHz > 0 AND st.ElevationDegrees >= 0
GROUP BY r.session, az_bin, el_bin
ORDER BY r.session, az_bin, el_bin;
