
-- Extract candidate orientation features for correlation audit
WITH sample_data AS (
    SELECT 
        r.Cn0DbHz as cn0,
        st.ElevationDegrees as elev,
        SIN(RADIANS(st.ElevationDegrees)) as sin_elev,
        -- Kasten-Young airmass approximation
        1.0 / (SIN(RADIANS(st.ElevationDegrees)) + 0.50572 * POWER(st.ElevationDegrees + 6.07995, -1.6364)) as airmass_ky,
        -- Simple airmass
        1.0 / NULLIF(SIN(RADIANS(st.ElevationDegrees)), 0) as airmass_simple,
        -- Relative Friis loss in dB: 20 * log10(f / 1.57542e9)
        20.0 * LOG10(r.CarrierFrequencyHz / 1575420000.0) as freq_loss_db,
        st.AzimuthDegrees as az,
        o.yawDeg as yaw,
        o.pitchDeg as pitch,
        o.rollDeg as roll
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
      AND r.Cn0DbHz > 0 AND st.ElevationDegrees >= 5.0
      AND (r.utcTimeMillis - o.utcTimeMillis) <= 200
)
SELECT * FROM sample_data
LIMIT 15000;
