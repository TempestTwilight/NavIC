
-- Compute relative azimuth (sat_az - phone_yaw) and residual C/N0 at 30-50 deg elevation
WITH rel_data AS (
    SELECT 
        r.session,
        r.Cn0DbHz,
        st.ElevationDegrees as el,
        -- Relative azimuth wrapped to [-180, 180]
        180.0 - ABS(ABS(st.AzimuthDegrees - o.yawDeg) - 180.0) as abs_rel_az,
        r.Cn0DbHz - (20.0 + 20.0 * SIN(RADIANS(st.ElevationDegrees))) as cn0_residual
    FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Raw.parquet') r
    JOIN (
        SELECT DISTINCT session, CAST(ROUND(UnixTimeMillis / 1000.0) AS BIGINT) as sec_epoch,
               ConstellationType, Svid, CarrierFrequencyHz, AzimuthDegrees, ElevationDegrees
        FROM read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/Status.parquet')
        WHERE session IN ('0_up_facing', '0_down_facing', '90_front_vertical', '90_back_vertical')
    ) st ON r.session = st.session
       AND CAST(ROUND(r.utcTimeMillis / 1000.0) AS BIGINT) = st.sec_epoch
       AND r.ConstellationType = st.ConstellationType
       AND r.CarrierFrequencyHz = st.CarrierFrequencyHz
    ASOF JOIN read_parquet('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/typed/OrientationDeg.parquet') o
      ON r.session = o.session
     AND r.utcTimeMillis >= o.utcTimeMillis
    WHERE r.session IN ('0_up_facing', '0_down_facing', '90_front_vertical', '90_back_vertical')
      AND r.Cn0DbHz > 0 AND st.ElevationDegrees BETWEEN 25.0 AND 50.0
      AND (r.utcTimeMillis - o.utcTimeMillis) <= 200
)
SELECT 
    FLOOR(abs_rel_az / 15.0) * 15.0 as rel_az_bin,
    COUNT(*) as n_obs,
    ROUND(AVG(cn0_residual), 2) as mean_residual,
    ROUND(MEDIAN(cn0_residual), 2) as median_residual,
    ROUND(QUANTILE_CONT(cn0_residual, 0.25), 2) as p25_residual,
    ROUND(QUANTILE_CONT(cn0_residual, 0.75), 2) as p75_residual
FROM rel_data
GROUP BY rel_az_bin
ORDER BY rel_az_bin;
